# Star Realms

A two-player Star Realms rules engine in Rust, covering the Core Set, Colony Wars, and Frontiers (alone or shuffled together). The same engine powers a Python training environment, scripted and neural-network bots, a multiplayer web server, and an offline browser build.

See [PLAN.md](PLAN.md) for scope, design, and status.

| Path | What |
| --- | --- |
| `crates/sr-core` | Rules engine, card data (`data/cards.toml`), action and observation encoding, replays |
| `crates/sr-bots` | Random, greedy, and heuristic bots; forward pass for exported networks (`.srnn`) |
| `crates/sr-py` | PyO3 bindings: parallel `VecEnv`, `Game`, `arena` |
| `crates/sr-wasm` | The engine for the browser: hot-seat, offline bots, replay viewer |
| `crates/sr-server` | axum server: lobbies, invite links, WebSocket games, SQLite persistence, bot opponents |
| `web/` | React + TypeScript client |
| `python/starrealms` | Masked PPO trainer, evaluation, terminal play, model export |

## Card art

Card images belong to Wise Wizard Games and are not in this repository. Download them once:

```sh
python tools/fetch_assets.py
```

This fills `assets/` (gitignored). Without it, the client draws text-only cards.

## Running the web app

Requires Rust with the `wasm32-unknown-unknown` target, `wasm-bindgen-cli` matching the version pinned in `crates/sr-wasm/Cargo.toml`, and Node 24.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129

(cd web && npm ci && npm run build)
cargo run --release -p sr-server
```

Open <http://127.0.0.1:3030>. Pick a name, then:

- **Online game**: create a game against a bot or a person. For a person, send the invite link; whoever opens it takes the open seat, and anyone after that spectates. Games survive server restarts and browser reloads; your identity is a token stored in the browser.
- **Local game**: runs entirely in the browser through WebAssembly, against a bot or hot-seat with a hand-off screen between turns.
- **Replays**: every finished game can be stepped through from either seat.

For client development, run `npm run dev` in `web/`; Vite proxies `/api` and `/assets` to the server on port 3030.

Server options (flags or environment variables):

| Flag | Env | Default |
| --- | --- | --- |
| `--bind` | `SR_BIND` | `127.0.0.1:3030` |
| `--db` | `SR_DB` | `starrealms.db` |
| `--web` | `SR_WEB` | `web/dist` |
| `--assets` | `SR_ASSETS` | `assets` |
| `--models` | `SR_MODELS` | `models` |
| `--bot-delay-ms` | `SR_BOT_DELAY_MS` | `450` |

Every `*.srnn` file in the models directory is offered as an opponent alongside the built-in bots.

`python tools/server_smoke.py` plays full games against a running server over HTTP and WebSockets and checks hidden information, stale moves, spectators, bots, and replays.

## Docker

```sh
docker build -t starrealms .
docker run -p 3030:3030 -v starrealms-data:/data -v "$PWD/models:/models:ro" starrealms
```

The build downloads the card art, so keep the image private.

## Engine and bots

```sh
cargo test
cargo run -p sr-core --release --example bench          # random-vs-random throughput
cargo run -p sr-bots --release --example arena -- 2000 core,colony_wars,frontiers
```

## Training

```sh
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy pytest torch     # pick the torch build for your GPU
maturin develop --release -m crates/sr-py/Cargo.toml
pytest python/tests

python -m starrealms.ppo --groups self:192,heuristic:64 --sets core --total-steps 20e6
python -m starrealms.evaluate runs/<run>/best.pt --games 2000
python -m starrealms.play --opponent runs/<run>/best.pt          # play it in the terminal
python -m starrealms.export runs/<run>/best.pt models/<name>.srnn # make it a server opponent
```

Exported models run in Rust with no PyTorch dependency; `pytest` checks that the Rust forward pass matches PyTorch.
