# Star Realms: engine, webapp, assets

Scope, fixed for the life of the project:

- Two-player Star Realms, full rules.
- Trade decks from **Core Set**, **Colony Wars**, and **Frontiers**, playable alone or shuffled together. Nothing else, ever: no heroes, gambits, events, missions, command decks, or Frontiers challenge (solo/co-op) cards.
- One rules engine, used unchanged by the training gym, the heuristic bots, and the webapp.

The engine is written in Rust. Training is in Python. The webapp is a Rust server plus a TypeScript client, with the same engine compiled to WebAssembly for offline and hot-seat play.

## 1. Repository layout

```
starrealms/
  Cargo.toml                 workspace
  crates/
    sr-core/                 rules engine, card data, observation + action encoding
    sr-bots/                 scripted agents (random, greedy, heuristic)
    sr-py/                   PyO3 bindings: vectorized env, replay, bots
    sr-wasm/                 wasm-bindgen wrapper for the browser
    sr-server/               axum: lobbies, WebSocket games, persistence
  data/
    cards.toml               hand-authored card definitions (source of truth)
    sets.toml                per-set card quantities, explorer pile size
    upstream/                official gallery JSON + community spreadsheet export (review inputs)
  python/
    starrealms/              gym wrapper, feature extractors, trainer, eval harness
  web/                       Vite + TypeScript + React client
  models/                    gitignored; exported .srnn networks the server offers as opponents
  tools/
    server_smoke.py          end-to-end check of a running server
    fetch_assets.py          downloads card images into assets/ (not committed)
    import_card_text.py      one-time: merges upstream data into a review file
  assets/                    gitignored; populated by tools/fetch_assets.py
  PLAN.md
```

`sr-core` has no dependency on the server, Python, or wasm crates. Everything above it talks to it through three types: `Game`, `Action`, `Observation`.

## 2. Engine (`sr-core`)

### 2.1 Rules surface

The three sets share one effect language. Everything a card in these sets does reduces to this list (derived from the full card text of all 137 unique cards; Scout, Viper, and Explorer are shared across sets):

| Effect | Cards |
| --- | --- |
| Gain trade / combat / authority | nearly all |
| Draw N | Mothership, Corvette, Nanobot Swarm, ... |
| Choice between effect lists (`OR`) | Patrol Mech, Barter World, Parasite, Outland Station, Moonwurm Hatchling |
| Opponent discards a card | Imperial Fighter, Cargo Craft, Light Cruiser, Jamming Terminal, ... |
| Scrap from hand, discard, or either; may/must; up to N | Trade Bot, Junkyard, Brain World, The Wrecker, Builder Bot, ... |
| Scrap from trade row, up to N | Battle Pod, Ravager, Swarmer, Blob Miner |
| Destroy target base, may/must | Missile Mech, Blob Destroyer, Mech Cruiser, Frontier Ferry (scrap) |
| Acquire for free, with cost cap / type filter and destination (discard, top of deck, hand) | Blob Carrier, Parasite, Leviathan, Moonwurm, Burrower (scrap), Gateship |
| Next acquisition this turn goes to top of deck / hand, with type filter | Central Office, Freighter, Federation Shipyard, Factory World, Long Hauler (scrap) |
| Discard then draw, draw then discard, discard up to N for a resource | Recycling Station, Orbital Platform, Captured Outpost, Supply Depot, Warpgate Cruiser, Farm Ship |
| Count-based: draw per Blob played, draw per card scrapped, combat per card scrapped this turn, combat equal to scrapped card's cost | Blob World, Brain World, Reclamation Station, The Incinerator, Pulverizer, Neural Nexus |
| Conditional on board: opponent has a base, you have ≥2 bases, ≥3 bases | Lancer, Embassy Yacht, Orbital Shuttle, Central Station |
| Copy a ship played this turn / copy a base in play | Stealth Needle, Stealth Tower |
| Counts as every faction for ally purposes | Mech World |
| Persistent trigger: whenever you play a ship / a Star Empire ship | Fleet HQ, Command Center |
| On-acquire trigger: if you played a faction card this turn, may put this into hand | Plasma Vent, Warning Beacon, Emperor's Dreadnaught, Colony Seed Ship |
| Put a base from discard on top of deck | Repair Mech |
| Returns from scrap heap to discard at end of turn | Mobile Market |
| Ally and **Double Ally** abilities (Frontiers: needs two other cards of the faction) | Hive Queen, Light Cruiser, Ion Station, ... |

Rule points that drive the design:

- "Target opponent discards a card" is a counter on the opponent. They choose which cards to discard as the first decision of their own next turn, after drawing. Play never passes to the other seat mid-turn.
- Ships' primary abilities resolve on play. Ally, scrap, and base abilities are activated by the player at any point in the main phase, each once per turn.
- Outposts must be destroyed before the player or their other bases can be attacked.
- Standard setup: 50 authority, 8 Scouts + 2 Vipers, five-card trade row, Explorer pile. First player draws 3 on turn one, then 5.
- If the trade deck runs out, the trade row simply stops refilling (standard two-player rule). The scrap heap is never reshuffled.

### 2.2 Card data

`data/cards.toml` is hand-authored and is the only thing the engine reads. One entry per unique card:

```toml
[[card]]
id       = "patrol_mech"
name     = "Patrol Mech"
set      = ["core"]
faction  = "machine_cult"
kind     = "ship"
cost     = 4
primary  = { choose = [ [{ trade = 3 }], [{ combat = 5 }] ] }
ally     = [{ scrap_from = ["hand", "discard"], may = false }]

[[card]]
id       = "brain_world"
name     = "Brain World"
set      = ["core"]
faction  = "machine_cult"
kind     = "base"
cost     = 8
defense  = 6
outpost  = true
activate = [{ scrap_from = ["hand", "discard"], up_to = 2, then = "draw_per_scrapped" }]
```

A `build.rs` reads the TOML and emits static Rust tables, so card lookups are array indexing and an invalid card file fails the build. Card ids are `u8` (137 unique cards fit comfortably).

The upstream sources used to seed and cross-check this file are described in section 4; they are review inputs, not runtime data.

### 2.3 State

```rust
struct Game {
    rng: Xoshiro256PlusPlus,     // seeded; replay = seed + set config + actions
    turn: u16,
    active: Seat,
    trade_deck: Vec<CardId>,     // order is hidden information
    trade_row: [Option<CardId>; 5],
    explorers: u8,
    scrap_heap: Vec<CardId>,
    players: [PlayerState; 2],
    turn_state: TurnState,       // trade/combat pools, pending decisions, per-turn flags
    outcome: Option<Outcome>,
}

struct PlayerState {
    authority: i16,
    deck: Vec<CardId>,           // order hidden
    hand: SmallVec<[CardId; 8]>,
    discard: Vec<CardId>,
    bases: SmallVec<[InPlay; 6]>,
    played: SmallVec<[InPlay; 10]>, // ships played this turn
    pending_discards: u8,        // owed at start of own turn
}
```

`InPlay` tracks which of a card's abilities (primary, ally, double ally, scrap, activate) have been used this turn, plus the copied card for Stealth Needle / Stealth Tower.

`TurnState` holds a stack of pending decisions. Playing a card pushes zero or more decisions (choose a mode, pick a scrap target, pick a base to destroy). The game is in exactly one of two modes: resolving the top pending decision, or in the free main phase where the player may play, activate, buy, attack, or end. The legal action set is computed from that mode.

Clone is a memcpy-sized operation. No heap allocation in the step path beyond `Vec` growth that is amortized to zero after the first few turns.

### 2.4 Actions

One flat discrete menu, fixed size, with a legal-action mask. Slots are grouped:

| Range | Meaning |
| --- | --- |
| `END` | end turn (remaining combat is auto-applied to the opponent when no outpost stands; strictly dominant in two-player) |
| `PLAY[i]` | play hand slot `i` |
| `ACTIVATE[j]` | use the next unused ability on in-play slot `j` (base activate, ally, double ally) |
| `SCRAP_ABILITY[j]` | use the scrap ability of in-play slot `j` |
| `BUY[k]` | buy trade row slot `k` (0–4) |
| `BUY_EXPLORER` | |
| `ATTACK_PLAYER` | all pool combat to the opponent's authority |
| `ATTACK_BASE[b]` | pay defense to destroy opponent base `b` |
| `CHOOSE[m]` | pick option `m` of the pending decision (mode of an `OR`, trade-row slot, in-play slot, hand slot) |
| `PICK_CARD[c]` | pick card id `c` when the pending decision targets a pile (discard pile scrap, Repair Mech) |
| `SKIP` | decline an optional effect |

Hand slots are capped at 12 and in-play slots at 16; both are above anything the three sets can produce. Total menu is roughly 230 slots, dominated by `PICK_CARD`. The mask is what makes it tractable; a typical main-phase state has 5–15 legal actions.

Cards with no decisions (Scout, Viper, most Blob ships) are still one `PLAY` action each. A `play_all_simple` macro is exposed to the UI and bots but is not part of the learned action space.

### 2.5 Observation

Built in Rust, returned as a fixed-size `f32` vector plus the mask. Always from the perspective of the seat to move. Contents:

- Scalars: own and opponent authority, trade pool, combat pool, turn number, cards remaining in own deck, pending discards.
- Own hand: per-card-id counts.
- Own discard, own deck multiset (known from the play log), own bases, own ships in play with ability-used flags.
- Opponent: bases with remaining defense, discard, full deck multiset (public), hand size.
- Trade row: five slots as card ids plus Explorer availability.
- Pending decision: type and the options on offer.

Card-id counts are the baseline encoding and are enough for a first network. A set-of-tokens view (variable-length list of card ids per zone) is available for an attention encoder later; both come from the same Rust function so they cannot drift.

### 2.6 Determinism, replay, and testing

- All randomness flows through the one seeded generator. `Game::new(seed, config)` plus the ordered list of actions reproduces any game. Replays are the persistence format for the server and the debugging format for the trainer.
- Per-card unit tests: a scripted position, one action, assert the resulting zones. Every card in `cards.toml` must have at least one.
- Invariant tests under random play: card conservation across all zones, authority only changes by declared effects, mask never offers an action that `step` rejects.
- Golden replays: a corpus of recorded games whose final states are hashed; the hash set is a regression test for any rules change.
- Benchmark: random-vs-random games per second per core. Target: at least 10k/s on one core of the i7-14700K. That puts a 20-core self-play run comfortably above 100k games/s before the network is the bottleneck.

### 2.7 Bots (`sr-bots`)

All bots implement `fn act(&mut self, obs: &Observation, mask: &Mask, rng) -> Action`.

1. `Random`: uniform over the mask. Used for invariant tests and benchmarks.
2. `Greedy`: play everything, buy the most expensive affordable card, attack face. The sanity floor.
3. `Heuristic`: the first real opponent. Per-turn plan rather than per-action reaction: score each purchase by cost efficiency, faction synergy with the current deck, and game phase; prefer scrapping Scouts/Vipers/Explorers; kill outposts before face damage when the combat math says so; race when the opponent is low. Tuned by playing it against `Greedy` and against humans until it wins reliably against a casual player.

The heuristic bot is also the first opponent on the server, so friends have someone to play against before anyone else is online.

## 3. Python (`sr-py`, `python/starrealms`)

`sr-py` is a `maturin` project exposing:

- `VecEnv(n, config)`: `n` independent games stepped in parallel with Rayon. `reset()` and `step(actions)` return NumPy arrays: observations `(n, obs_dim)`, masks `(n, act_dim)`, rewards, dones, and a seat index per row. Env groups choose who plays the other seat: both seats from Python (self-play, the default shape), or a bot or exported model played inside Rust so only learner decisions come back.
- `Game`: a single game for scripts and terminal play; load and save replays, render positions as text, ask a bot for its move.
- `arena(a, b, games)`: bot-versus-bot win counts in parallel.
- `srnn_forward`: the Rust network forward pass, for parity tests against PyTorch.

The default reward is terminal `+1 / -1`. Potential-based shaping (authority and deck quality) is opt-in through the trainer's `--shaping` coefficient.

`python/starrealms` holds the trainer (masked PPO, adapted from CleanRL's single-file `ppo.py`), a snapshot league, an evaluation harness against the scripted bots, terminal play, and `export.py`. Models are exported to `.srnn` (magic, JSON header, raw little-endian `f32` tensors), which `sr_bots::Net` evaluates in plain Rust. That keeps ONNX Runtime out of the server and the wasm build.

## 4. Assets and card data

### 4.1 Sources

**Card data and images: the official card gallery.** `https://www.starrealms.com/card-gallery/` is a client-side app that loads `https://www.starrealms.com/card-gallery/data/cards.json`. Verified contents:

- 367 cards and a `sets` array mapping set id to `{cardId, quantity}`. Set ids for our scope are `coreSet`, `colonyWars`, `frontiers`.
- Per card: `id`, `name`, `image`, `factions`, `types`, `cost`, `defense`, `roles`, `text`, `notes`.
- Images resolve at `https://www.starrealms.com/card-gallery/<image>`, i.e. `https://www.starrealms.com/card-gallery/images/content/card-gallery/<slug>.webp`. A sample download returned a 48 KB WebP. Note the gallery base path: the same path under the site root 404s.

Known defect: `defense` is `null` for 33 bases in our three sets, including every outpost (Port of Call, Junkyard, Brain World, Mech World, Blob World, Frontier Station, Stealth Tower, Reclamation Station, ...). Outpost status is not encoded anywhere in that JSON.

**Defense and outpost flags: the community spreadsheet.** The long-running Star Realms card spreadsheet (`docs.google.com/spreadsheets/d/1yqB6Q8CDI7elEjbLmwX6056aBp0xQDQ-C1NtpqVfuI8`) and its JSON export carry `Defense` as `"5"` or `"6 Outpost"`. Colony Wars and Frontiers are complete there; the Core Set is missing Blob World's defense (it is 7, not an outpost).

**Cross-check: the official rulebooks.** The Frontiers rulebook PDF (`https://starrealms.com/files/rules/StarRealms_Frontiers_RuleBook2.pdf`) and the Core Set rulebook list every card with its quantity. Use them to confirm the set contents in `data/sets.toml`.

### 4.2 Process

1. `tools/fetch_assets.py` (done; rerun at dev setup and at deploy) downloads everything external in one pass and records a SHA-256 manifest so reruns are no-ops and `--verify` detects drift:
   - `data/upstream/cards.json`: the official gallery database (committed).
   - `data/upstream/star-realms-cards.json`: the community spreadsheet export (committed).
   - `assets/cards/<cardId>.webp`: the 137 in-scope card images, keyed by the gallery's card id. Ships are 300×420, bases 420×300 (Blob Fighter is served at 214×300). `--all` fetches every card in the gallery instead.
   - `assets/icons/`: the gallery's four faction icons plus the cost and defense icons.
2. `tools/import_card_text.py` (done): joins the two databases by normalized name into `data/upstream/review.csv` with defense and outpost filled in, flagging disagreements. The join has been checked: every in-scope card matches a spreadsheet row, and the only base still lacking a defense value is Blob World (7, not an outpost), which is set by hand. 32 of the 52 in-scope bases are outposts.
3. Hand-author `data/cards.toml` from that review file. Card text goes in as our own structured effect program, not as the upstream string; the string is kept as a `source_text` field for the test suite and the UI tooltip.
4. The UI has a text-only card renderer driven by `cards.toml` that is used when an image is missing. That keeps development possible with an empty `assets/` directory and keeps the project working if the gallery changes.

`assets/` is gitignored. The art belongs to Wise Wizard Games; it is downloaded by whoever runs the project and never redistributed from this repository. The hosted site is for a private group of friends, not a public or commercial service. The server reads only from `assets/` and `data/`, so once the fetch has run, nothing at runtime touches the network.

### 4.3 Things we draw ourselves

Trade, combat, and authority icons; the card back; the playmat; outpost and defense badges. Simple SVGs in `web/src/icons/`. The gallery's faction icons are available in `assets/icons/` if we prefer them to our own.

## 5. Webapp

### 5.1 Shape

- **Server** (`sr-server`): axum on tokio. Serves the built client as static files and a WebSocket per connection. Holds live games in memory, persists every game as a replay (seed, config, action log, seat identities) in SQLite (`rusqlite`, WAL). The server is authoritative: clients send action ids, the server validates against the mask, steps the engine, and broadcasts the new per-seat observation.
- **Client** (`web/`): Vite, TypeScript, React. Renders from the observation and mask: legal cards glow, illegal ones do not, pending decisions open as a prompt with the offered options. Clicking is sending an action id. Nothing in the client knows the rules.
- **WASM** (`sr-wasm`): the engine in the browser. Used for hot-seat, offline play against the heuristic bot, and the replay viewer. It can also load an `.srnn` model (`Engine.load_model`) for offline play against a trained policy; the client does not offer that yet.

Because the opponent never acts during your turn, play is naturally asynchronous. A game can be live (both connected) or correspondence (take your turn whenever). The server treats both identically; only the UI's notification behavior differs.

### 5.2 Features in order

1. Lobby with a display name and an invite link. No accounts. A lobby creates a game; the link joins the other seat; later joiners spectate.
2. Play versus the heuristic bot on the server, and hot-seat locally through WASM.
3. Reconnection by session token stored in the browser; the full replay is the resync.
4. Game list per session: in-progress, awaiting-you, finished. Replay viewer with step controls.
5. Trained-model opponents: every `.srnn` in the server's models directory is listed as an opponent after the heuristic bot.
6. Optional: email or push notification when it is your move; magic-link accounts so a game list follows a person across devices.

### 5.3 Hosting

One Docker image containing the server binary and the built client. Options, cheapest first:

- Home server behind a Cloudflare Tunnel. Zero cost, TLS handled by Cloudflare, no open ports.
- Fly.io or a small Hetzner VPS with Caddy for TLS. A few dollars a month.

SQLite on a persistent volume is enough for a friend group for years. Backups are a copy of one file. Card images are fetched at image build time by `tools/fetch_assets.py`, not stored in the repository or the registry layer that is shared publicly.

## 6. Milestones

Each milestone ends with something runnable.

1. **Core rules, Core Set only.** `cards.toml` for 49 cards, engine, mask, replay, per-card tests, invariants, benchmark. Random-vs-random runs at target speed. Text renderer prints a position.
2. **Bots and Python.** `Random`, `Greedy`, `Heuristic`. `sr-py` `VecEnv`. A notebook plays 10k games of heuristic-vs-greedy and plots win rate and game length.
3. **Colony Wars and Frontiers.** Remaining 94 cards, double ally, on-acquire triggers, Stealth Tower, Mobile Market. Set mixing in `sets.toml`. Golden replays frozen.
4. **First training run.** Masked PPO against `Heuristic`, then self-play with a checkpoint league. Eval harness. The deliverable is a checkpoint that beats `Heuristic` by a clear margin.
5. **Webapp, local.** WASM build, client, hot-seat and vs-bot play with downloaded card art.
6. **Webapp, hosted.** Server, lobbies, invite links, persistence, reconnection, deployed.
7. **Model opponents online.** `.srnn` export, Rust forward pass in the server, opponent selection.

Status (October 2026):

- 1–3 done. 137 cards across the three sets, per-card tests, invariants, golden replays. About 46k random-vs-random Core Set games/s on one core; `Heuristic` beats `Greedy` 97% of the time.
- 4: trainer, league, and evaluation work end to end (a 0.4M-step GPU smoke run, exported and parity-checked). No checkpoint yet that clearly beats `Heuristic`; that needs a real run.
- 5 done: offline vs-bot, hot-seat with a hand-off screen, replay viewer.
- 6: server, lobbies, invite links, spectators, persistence, reconnection, and replays work; `tools/server_smoke.py` covers them. A Dockerfile exists; not yet deployed.
- 7: plumbing done. Waits on a strong checkpoint from milestone 4.

## 7. Decisions already made

- Two players only. The state struct uses a fixed array of two seats; multiplayer variants are not a goal.
- Frontiers challenge cards are out. They are a different game mode with boss rules and a trade-row mulligan.
- End turn auto-applies leftover combat to the opponent. Combat on bases is always an explicit action.
- Card data is hand-authored; upstream sources are review inputs only.
- Reward is terminal by default. Shaping lives in `features()` and is opt-in.
- Rust core, Python training, TypeScript client, WASM for the browser engine. No rules logic outside `sr-core`.
