# Star Realms

A two-player Star Realms rules engine in Rust, covering the Core Set, Colony Wars, and Frontiers (alone or shuffled together). The same engine powers a Python training environment, scripted and neural-network bots, a multiplayer web server, and an offline browser build.

See [PLAN.md](PLAN.md) for scope, design, and status.

| Path | What |
| --- | --- |
| `crates/sr-core` | Rules engine, card data (`data/cards.toml`), action and observation encoding, replays |
| `crates/sr-bots` | Random, greedy, and heuristic bots; forward passes for exported networks (MLP `.srnn`, entity transformer `.safetensors` via candle) |
| `crates/sr-py` | PyO3 bindings: parallel `VecEnv`, `Game`, `arena` |
| `crates/sr-wasm` | The engine for the browser: replay viewer |
| `crates/sr-server` | axum server: lobbies, invite links, WebSocket games, SQLite persistence, bot opponents |
| `web/` | React + TypeScript client |
| `python/starrealms` | Masked PPO trainer, distillation, evaluation, terminal play, model export |

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

Every `*.srnn` and `*.safetensors` file in the models directory is offered as an opponent alongside the built-in bots.

`uv run tools/server_smoke.py` plays full games against a running server over HTTP and WebSockets and checks hidden information, stale moves, spectators, bots, and replays.

## Docker

```sh
docker compose up -d --build
```

or by hand:

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

The Python side is a [uv](https://docs.astral.sh/uv/) project. `uv sync` compiles the engine with maturin and installs it in editable mode; it rebuilds automatically when Rust sources change.

```sh
uv sync --extra train          # PyTorch from the ROCm 7.1 index; edit [tool.uv.index] in pyproject.toml for CUDA or CPU
uv run pytest                  # fast tests
uv run pytest -m slow          # also checks that a short PPO run learns to beat the random bot
```

Training is a CleanRL-style single-file masked PPO (`python/starrealms/ppo.py`) with self-play against a league of past snapshots. Checkpoints get Elo ratings relative to the heuristic bot (`runs/<run>/ratings.json`), and `best.pt` is the highest-rated one. Configuration is a dataclass; see every option with `--help`.

```sh
uv run python -m starrealms.ppo --help
uv run python -m starrealms.ppo --groups heuristic:192,greedy:32,random:32 --sets core --total-steps 50e6
uv run python -m starrealms.ppo --init runs/<run>/best.pt --groups self:192,heuristic:64
uv run tensorboard --logdir runs              # or add --track to log to Weights & Biases

uv run python -m starrealms.evaluate runs/<run>/best.pt --games 2000   # win rates with 95% intervals
uv run python -m starrealms.play --opponent runs/<run>/best.pt          # play it in the terminal
uv run python -m starrealms.export runs/<run>/best.pt models/<name>.srnn  # make it a server opponent
```

`--sets-mix core:0.3,core+colony_wars+frontiers:0.7` splits every env group across card-set selections and evaluates on each.

There are two architectures (`--arch`): `mlp` (`model.Policy`) and `entity` (`model.EntityPolicy`), a small transformer over card tokens (`tokens.py`) whose action head points at the card each action targets. A new architecture can start from an existing checkpoint's behaviour instead of from scratch:

```sh
uv run python -m starrealms.distill --teacher runs/phase_c/best.pt --arch entity --run-name distill_entity
uv run python -m starrealms.ppo --init runs/distill_entity/best.pt --lr 1e-4 --warmup-updates 20 \
  --sets-mix core:0.3,core+colony_wars+frontiers:0.7 --eval-opponents heuristic,greedy,model:models/phase_c.srnn
uv run python -m starrealms.export runs/<run>/best.pt models/<name>.safetensors
```

Exported models run in Rust with no PyTorch dependency (MLPs in plain Rust, entity models with candle, also in the browser build); `pytest` checks that both Rust forward passes match PyTorch.
