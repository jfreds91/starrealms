# Star Realms bots: what we've trained, and where it stands

A handoff for whoever continues the model work. It runs chronologically from the scripted bots to the current best network, `phase_f`, and ends with the open problems and the most promising next steps.

**Current state (Oct 20, 2026):** the strongest bot is `models/phase_f.safetensors`, a 4.4M-parameter entity transformer trained with PPO self-play. Against `phase_c`, the long-standing baseline MLP, it wins about **65% on the Core Set** and **60% on mixed sets** (Core + Colony Wars + Frontiers). The target was 70%; nothing tried so far gets there by training longer. The web server plays `phase_f` as an opponent and uses its value head for a win-chance meter.

## Terms

- **Core / mixed:** games with only the Core Set, or with Core + Colony Wars + Frontiers shuffled together. Training uses 30% Core and 70% mixed games (`--sets-mix core:0.3,core+colony_wars+frontiers:0.7`).
- **Decision / step:** one choice by one player (play a card, buy, attack, end turn, pick a target). A game is about 22-24 turns and roughly 150-250 decisions.
- **Win rate:** head-to-head over 2,000 games with alternating seats unless noted. A 95% interval is about ±2 points at 2,000 games, ±1.5 at 4,000.
- **League / self-play:** the learner plays the current network or frozen earlier snapshots of itself. Snapshots are drawn by how often they still beat the learner (prioritized fictitious self-play, weight `(1 - learner score)^2`).
- **Distillation:** training a new network to imitate an existing one's policy and value (`distill.py`). It is how a new architecture or size starts from an existing model's skill instead of from scratch.
- **MLP:** the first network: fully connected layers over a flat observation, with one output logit per action id (622 actions).
- **Entity transformer:** the second architecture. It encodes the board as one token per card plus scalars, and scores each action by pointing at the card that action targets.

## The models

All exported models live in `models/` (gitignored). `.srnn` files are MLPs, run in plain Rust; `.safetensors` files are entity transformers, run in Rust with candle. Training checkpoints are in `runs/<name>/` and logs in `logs/<name>.log`. Both are gitignored too, about 9 GB in total, so **do not delete them**.

| Model | What it is | Params | Key results |
|---|---|---|---|
| random, greedy | Built-in Rust bots | n/a | Floors |
| heuristic | Hand-written Rust strategy | n/a | Beats greedy 968-32 |
| `phase_a` | MLP, PPO vs bots, Core only | 2.2M | 82% vs heuristic |
| `phase_b` | `phase_a` + league self-play, Core only | 2.2M | 85% vs heuristic, 58% vs `phase_a` |
| `phase_c` | `phase_b` + self-play on mixed sets | 2.2M | **The baseline.** 86% vs heuristic (mixed), 82% vs `phase_b` (mixed) |
| `phase_d` | Small entity transformer (distilled from `phase_c`, then PPO) | 0.73M | vs `phase_c`: 57% Core, 55% mixed |
| `entity2` | `phase_d` after 67M more PPO steps | 0.73M | vs `phase_c`: ~59% mixed |
| `cap_m` | Medium transformer distilled from `entity2` (not in `models/`) | 4.4M | ~50% vs `entity2` |
| `phase_e` | `cap_m` + PPO at learning rate 3e-5 | 4.4M | vs `phase_c`: 62.4% Core, 57.7% mixed; 52.9% vs `entity2` |
| **`phase_f`** | `phase_e` + 30M steps with the learning rate decaying to zero | 4.4M | **vs `phase_c`: 64.8% Core, 59.5% mixed; 52.0% vs `phase_e` (4,000 games)** |

`smoke.srnn` and `cap_m.safetensors` in `models/` are test artifacts, not milestones.

## History

### 0. Scripted bots (`crates/sr-bots`)

- **random:** uniform over legal actions.
- **greedy:** plays everything, uses every non-scrap ability, buys the most expensive card, attacks outposts and then the opponent. The sanity floor.
- **heuristic** (`heuristic.rs`): draw first, play everything, thin the deck early, buy for value and faction synergy, clear outposts, race when lethal. It beats greedy 968-32, and it was the yardstick for all early networks.

All bots, and every network, see only what the player to move could know. No agent inspects the opponent's hand or any draw-pile order.

### 1. Training setup

- **Pipeline:** a CleanRL-style single-file masked PPO (`python/starrealms/ppo.py`) over a parallel Rust environment (`crates/sr-py`, built into Python by maturin via `uv sync`). Illegal actions are masked out of the policy.
- **Hardware:** one AMD RX 9070 (16 GB) with ROCm 7.1.
- **Hypothesis:** `torch.compile` and bfloat16 speed up training.
  - **Result:** bfloat16 autocast about doubled throughput with no quality loss, so it's the default (`--bf16`). `torch.compile` was 7× *slower* in the training loop on this GPU, so it was removed.

### 2. Phase A: MLP against the bots, Core Set

- **Test:** about 100M steps against heuristic, greedy and random bots (`--groups heuristic:192,greedy:32,random:32 --sets core`).
- **Problem found:** the original rating system (openskill) drifted. Ratings fell while win rates rose, so `best.pt` pointed at a weak early checkpoint.
  - **Fix:** ratings are now refitted jointly from all results each time (Bradley-Terry, as BayesElo does) in `league.Ratings`, with a regression test.
- **Result:** `phase_a` beats the heuristic bot **82% [80, 83]**.

### 3. Phase B: league self-play, Core Set

- **Hypothesis:** self-play against snapshots beats training only against bots.
- **Result:** confirmed. `phase_b` beats `phase_a` **58%** and the heuristic **85%**.
- **Side finding:** self-play ran at about 25k steps/s versus 60k for bot-only training. The cause was one small GPU forward pass per league opponent per step. Fixed by batching all league opponents into one forward pass.

### 4. Phase C: self-play on mixed sets

- **Result:** on mixed sets, `phase_c` beats `phase_b` **82%** and the heuristic **86%**.
- **Problem:** it got worse on Core-only games (42% vs `phase_b`), because it never saw them. Every later run uses the 30/70 Core/mixed split.

### 5. Phase D: entity transformer with a pointer action head

- **Hypothesis:** the MLP's weakness isn't the 622-way action space but that an action id like "buy trade-row slot 3" means a different card every turn. An architecture that attends to the actual card an action targets should learn faster and generalize across cards.
- **Design** (`tokens.py`, `model.EntityPolicy`; Rust port in `crates/sr-bots/src/entity_net.rs`):
  - A pre-LayerNorm transformer encoder over entity tokens: each card in hand, in play, in discard, in the trade row, and so on, embedded by card id plus zone features, together with scalar tokens.
  - Action logits are dot products between a query per action verb and the key of the targeted entity.
  - There's a value head on a pooled token.
- **Control:** a wide MLP with 5.9M parameters, eight times the transformer's 0.73M, given the same recipe. Both were distilled from `phase_c`, then given 80M PPO steps.
- **Results:**
  - The transformer matched `phase_c` after **0.8M** distillation states; the MLP needed **3.7M**.
  - The wide MLP only got back to `phase_c`'s level (52% Core, 47% mixed).
  - The transformer reached **57%** on Core and **55%** on mixed against `phase_c`, and beat the wide MLP 55-59%. This became `phase_d` (commit `92d7f3a`).
  - **Conclusion:** the gain comes from the architecture, not from parameter count.

### 6. Longer training of the small transformer

- **Hypothesis:** `phase_d` was still improving, so more training would push it higher.
- **Test:** 67M more PPO steps (`runs/ppo_entity2`) with 448 envs (`--groups self:384,heuristic:64`).
- **Result:** it plateaued at **55-60%** vs `phase_c`. The best checkpoint scored 59.1%; it became `entity2`.
- **Later explanation:** this run was scheduled to anneal from 1e-4 over 400M steps but was stopped at 67M, while the learning rate was still about 0.85e-4. It never got a low-learning-rate stretch (see section 10).

### 7. Model-size test (distillation only)

- **Hypothesis:** 0.73M parameters is too small to hold more skill.
- **Test:** distill `entity2` into three sizes on the same 10M states (`runs/cap_s`, `cap_m`, `cap_l`):

| Size | Config | Params | KL to teacher | Same move as teacher | Value MSE |
|---|---|---|---|---|---|
| Small | `--d 128 --tf-layers 3 --ff 256 --heads 4` | 0.73M | 0.054 | 89.5% | 0.0048 |
| Medium | `--d 256 --tf-layers 4 --ff 1024 --heads 8` | 4.4M | 0.020 | 94.8% | 0.0026 |
| Large | `--d 384 --tf-layers 6 --ff 1536 --heads 8` | ~11M | 0.012 | 95.1% | 0.0017 |

- **Result:** supported. A model the teacher's own size can't copy it closely, while the larger ones can and were still improving. Imitation alone doesn't beat the teacher, though: all three score about 50% against it. The large model has never been PPO-trained.

### 8. Lookahead search without peeking (`crates/sr-bots/src/search.rs`, commit `38ed021`)

- **Constraint:** the search must not see hidden information: deck order, the opponent's hand, or which card replaces one bought from the trade row.
- **Method:**
  - `Game::determinize(seat, seed)` builds copies of the game in which every hidden card is re-dealt from what `seat` knows, and the RNG is replaced. A test checks that two games which look the same to the player produce identical copies.
  - `SearchAgent` tries the policy's top-k moves on N copies, plays the rest of the turn greedily with the policy, and scores the result with the value head after the hand-over.
- **Results** (search vs the same network without it):
  - Pure value scoring: 49-52%, which is noise.
  - With a bonus for moves the policy already likes (`prior_weight`): about **53%**.
  - More copies (8, 16), or also playing out the opponent's reply: no better.
  - On the medium model, whose value head is twice as accurate: 50-52%.
  - Scoring mid-turn, before ending the turn: lost 445 of 448. The value head assumes unspent trade and combat will be used well, so "end turn now" looks as good as playing on. Leaves must come after a hand-over.
- **Conclusion:** the value head discriminates between moves no better than the policy does, so shallow search adds little. Search has never been used in training.
- **Usage:** spec `search:<model>@width,samples[,prior_weight[,turns]]` in the `versus` tool.

### 9. PPO on the medium model (`phase_e`)

- **Test 1:** `runs/ppo_m`, starting from `cap_m` with the small model's learning rate of 1e-4.
  - **Result:** it got worse. Elo fell from 310 to 262, entropy rose from 0.39 to 0.53, and per-update KL was about 10× the small model's.
  - **Diagnosis:** the learning rate was too high for the bigger model.
- **Test 2:** `runs/m_lr3e5`, at a learning rate of 3e-5.
  - **Result:** steady improvement, to Elo about 355.
  - Stopped at 169M of 250M planned steps, with the gain down to about one point per 50M steps.
  - The best checkpoint (`best_u02850.pt`) became `phase_e`: **62.4%** Core and **57.7%** mixed vs `phase_c`, and **52.9%** vs `entity2`.

### 10. Hypothesis tests on the small model (`runs/h_*`)

- **Setup:** four 30M-step runs, each continuing `entity2` with the learning rate decaying from 5e-5 to zero:
  - **Control** (`h_ctl3`): no other change.
  - **Weight decay 0.1** (`h_wd`).
  - **Entropy coefficient 0.003 instead of 0.01** (`h_ent2`).
  - **Diverse opponents** (`h_div`): 96 of 448 envs against the fixed MLPs `phase_a`, `phase_b` and the wide MLP.
- **Results** (final checkpoints, mixed sets):

| Run | vs `entity2` (2,000 games) | vs control's final (4,000 games) |
|---|---|---|
| Control | 54.1% | n/a |
| Weight decay 0.1 | 52.8% | 50.5% [49.0, 52.1] |
| Entropy 0.003 | 55.5% | 50.3% [48.8, 51.9] |
| Diverse opponents | 54.0% | 49.6% [48.1, 51.2] |

- **Conclusion:** none of the three changes did anything measurable. **All four**, control included, gained 4-5 points, so the gain came from **decaying the learning rate to zero**.
- Diverse *transformer* opponents remain untested (see the known issues below).
- **Bug found along the way:** occasionally the last minibatch held a single example. The standard deviation of one value is NaN, so advantage normalization wrote NaN into the weights; it struck at update 62 with one seed. Fixed in `788807d`: equal-size minibatches via `torch.tensor_split`, plus guards that skip any minibatch with a non-finite loss or gradient and save the first one to `runs/<run>/nonfinite.pt`.

### 11. Learning-rate decay on the medium model (`phase_f`)

- **Test:** `runs/m_anneal`, continuing `phase_e` with the learning rate decaying from 1e-5 to zero over 30M steps.
- **Result:**

| Checkpoint | vs `phase_e` (4,000, mixed) | vs `phase_c` mixed | vs `phase_c` Core |
|---|---|---|---|
| Final (`final_u523.pt` = `latest.pt`) | **52.0%** [50.4, 53.5] | **59.5%** | **64.8%** |
| Highest Elo during training (update 300) | 50.5% | 58.0% | 63.3% |

- The final checkpoint was exported as `phase_f`. The gain is smaller than the small model's, as expected: the medium run was already at about 1e-5 when it stopped.

## Recipe that produced `phase_f`

From the repo root, after `uv sync --extra train`. Every option is listed by `uv run python -m starrealms.ppo --help`.

```sh
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True   # fewer ROCm OOM retries

# 1. Distill the medium architecture from the best small transformer
uv run python -m starrealms.distill --teacher runs/ppo_entity2/best.pt --arch entity \
  --d 256 --tf-layers 4 --ff 1024 --heads 8 --total-steps 10e6 --eval-every 100000 --target 1.1 \
  --run-name cap_m

# 2. PPO self-play (this run was stopped at 169M steps; best checkpoint at update 2850)
uv run python -m starrealms.ppo --init runs/cap_m/latest.pt --lr 3e-5 --warmup-updates 10 \
  --sets-mix core:0.3,core+colony_wars+frontiers:0.7 --groups self:384,heuristic:64 \
  --total-steps 250e6 --eval-games 400 --eval-opponents heuristic,model:models/phase_c.srnn \
  --run-name m_lr3e5

# 3. Short decay to zero
uv run python -m starrealms.ppo --init runs/m_lr3e5/best_u02850.pt --lr 1e-5 --warmup-updates 10 \
  --sets-mix core:0.3,core+colony_wars+frontiers:0.7 --groups self:384,heuristic:64 \
  --total-steps 30e6 --eval-games 400 --eval-opponents heuristic,model:models/phase_c.srnn \
  --run-name m_anneal

# 4. Export for the Rust server and tools
uv run python -m starrealms.export runs/m_anneal/latest.pt models/phase_f.safetensors
```

PPO details (defaults unless set above):
- 128 decisions per env per update, so 448 envs give about 57k decisions per update, of which about 45k are the learner's own.
- 3 epochs over minibatches of 4,096; clip 0.2; γ 0.995, λ 0.95; entropy 0.01; value coefficient 0.5; target KL 0.05.
- Potential-based shaping on the authority difference with coefficient 0.3, so the value head estimates `return - 0.3·phi`. Add `0.3·phi` back to get a win estimate (`search.rs` and `meter.rs` do this).
- `--anneal-lr` decays linearly to zero over `--total-steps`.
- The league holds 12 snapshots, taken every 100 updates; evaluation also runs every 100 updates, with 400-500 games per opponent.
- The medium model trains at about 5.5k decisions/s. About 63% of wall time is learning, 24% playing, and 13% evaluation pauses (measured at the old eval interval of 25).

## How to evaluate

Head-to-head in Rust, on all cores. This is the most reliable measure, and what every number above uses:

```sh
cargo build --release -p sr-bots --example versus
./target/release/examples/versus model:models/phase_f.safetensors model:models/phase_c.srnn 2000 core 7
./target/release/examples/versus model:models/phase_f.safetensors model:models/phase_c.srnn 2000 core,colony_wars,frontiers 7
```

Arguments are `<a> <b> [games] [sets] [seed]`; it prints A's score with a 95% Wilson interval. Agent specs: `random`, `greedy`, `heuristic`, `model:<path>[@temperature]`, `search:<path>@width,samples[,prior,turns]`. To compare a training checkpoint, export it to `/tmp/x.safetensors` first.

`uv run python -m starrealms.evaluate <ckpt> --games 2000` also works from Python. `uv run pytest` checks that the Rust forward passes (MLP and candle) match PyTorch.

## Lessons

1. **Architecture mattered most.** Pointing actions at the card they target beat an MLP with eight times the parameters.
2. **Bigger models need smaller learning rates.** 1e-4 made the 4.4M model worse; 3e-5 worked.
3. **Finish the learning-rate schedule.** Stopping a run while the learning rate is high leaves several points on the table. A 30M-step decay to zero is a cheap win.
4. **Don't trust `best.pt` blindly.** It's chosen by Elo from 400-game evaluations, which are noisy. After a full decay, the final checkpoint has been at least as good. Confirm with `versus` at 2,000-4,000 games.
5. **Distillation transfers skill to a new shape but doesn't add it.** Students land at about 50% against the teacher.
6. **Small effects need controls.** The hypothesis runs only made sense against a control trained identically. Compare variants head-to-head at 4,000 games.

## Known issues and gotchas

- **Transformers are slow as opponents inside the Rust env.** candle's thread pool fights the environment's rayon workers, which makes transformer opponents (and search agents in Python's `arena`) about 150× slower than MLP opponents. `versus` works around it by setting `RAYON_NUM_THREADS=1`. Until this is fixed, the league can't include strong *fixed* transformer opponents such as `entity2` or `phase_e`.
- **Bootstrapping for the waiting seat is crude.** At the end of a rollout window, the seat that isn't to move reuses its own last value prediction, so its final decision gets almost no policy signal (`compute_gae` in `ppo.py`). That's at most one or two decisions per seat per game per update.
- **The value head only models the player to move.** Asking it about the non-moving player is out of distribution. The win meter (`crates/sr-bots/src/meter.rs`) therefore evaluates the mover, averaging over 8 determinizations when the mover is the opponent, and gives the other player the complement.
- **Mid-turn values are biased.** Unspent trade and combat are valued as if they'll be spent well. This matters for search leaves and makes the meter jump at turn boundaries.
- **Memory:** three small-model runs fit on the GPU at once; medium runs should go one at a time. ROCm logs harmless "memory allocation failed ... OOM" warnings early in a run; `expandable_segments` reduces them.
- **Disk:** checkpoints are large (`runs/` is 9 GB). The root partition is small; keep everything on `/home/bigdrive`.

## Promising next steps

In rough order of expected payoff for effort:

1. **PPO on the large model (~11M).** It's already distilled (`runs/cap_l`) and copies the teacher best. Distill it from `phase_f` instead of `entity2`, then use the medium recipe with a lower learning rate (try 1e-5 to 2e-5) and a full decay to zero. Expect a day or more of GPU time and a few points of gain.
2. **Fix the candle/rayon threading conflict,** then train against diverse fixed *transformer* opponents (`entity2`, `phase_e`, `phase_f`). Diversity has only been tested with weak MLP opponents.
3. **Search-guided training (AlphaZero-style):** choose self-play moves with the search, train the policy on the search's choices and the value head on outcomes, and repeat. This is the most likely route to a large jump, and the biggest project: search is 50-100× slower per move unless batched on the GPU.
4. **Training speed:** profile the learning phase (attention kernel choice on ROCm; whether `torch.compile` helps the transformer), and overlap rollout with learning (actor/learner) to hide most of the 24% spent playing.
5. **Fix the waiting-seat bootstrap** by scoring the cut-off position from the other seat's side, negating the mover's value.

## Where things are

| Path | What |
|---|---|
| `python/starrealms/ppo.py` | PPO trainer, league, GAE, evaluation hooks |
| `python/starrealms/model.py` | `Policy` (MLP) and `EntityPolicy` (transformer) |
| `python/starrealms/tokens.py` | Observation → entity tokens for the transformer |
| `python/starrealms/distill.py` | Teacher → student distillation |
| `python/starrealms/league.py` | Snapshot league and Bradley-Terry ratings |
| `python/starrealms/evaluate.py`, `export.py`, `play.py` | Evaluation, export to Rust formats, terminal play |
| `crates/sr-core/src/obs.rs` | Flat observation encoding (shared by both architectures) |
| `crates/sr-core/src/game.rs` | Rules engine, including `determinize` |
| `crates/sr-bots/src/heuristic.rs` | Heuristic bot |
| `crates/sr-bots/src/net.rs`, `entity_net.rs` | Rust forward passes (MLP; candle transformer) |
| `crates/sr-bots/src/search.rs`, `meter.rs` | Lookahead search; win-chance meter |
| `crates/sr-bots/examples/versus.rs` | Head-to-head evaluation tool |
| `runs/`, `logs/`, `models/` | Checkpoints, training logs, exported models (all gitignored) |
| `PLAN.md` | Original project scope and design |
