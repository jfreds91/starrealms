# Star Realms bots: training journal

A running log of the bots built and trained for this project, oldest first: hypotheses, tests, results, and what each one taught. Reference notes follow the entries: the recipe behind the current best model, how models are evaluated, lessons, known issues, and ideas not yet tried. New entries go at the end of the journal section.

**Where things stand (Oct 22, 2026):** `phase_f` is still the model to beat. `phase_g` ties it. A privileged value head, which saw the opponent's hand and the next draws during training, explained less of the game than the public head and did not quiet the purchase advantage, so it was not used to train a policy. A medium transformer trained from scratch for 150M steps (`m_scratch`) also failed to pass `phase_f`. On mirrored deals, about 54% of `phase_f` vs `phase_c` pairs are decided by the deal; of the pairs a player decides, `phase_f` wins 81% on Core and 74% on mixed. The web server still plays `phase_f`.

## Terms

- **Core / mixed:** games with only the Core Set, or with Core + Colony Wars + Frontiers shuffled together. Training uses 30% Core and 70% mixed games (`--sets-mix core:0.3,core+colony_wars+frontiers:0.7`).
- **Decision / step:** one choice by one player (play a card, buy, attack, end turn, pick a target). A game is about 22-24 turns and roughly 150-250 decisions.
- **Win rate:** head-to-head over 2,000 games with alternating seats unless noted. A 95% interval is about ±2 points at 2,000 games, ±1.5 at 4,000.
- **League / self-play:** the learner plays the current network or frozen earlier snapshots of itself. Snapshots are drawn by how often they still beat the learner (prioritized fictitious self-play, weight `(1 - learner score)^2`).
- **Distillation:** training a new network to imitate an existing one's policy and value (`distill.py`). It is how a new architecture or size starts from an existing model's skill instead of from scratch.
- **MLP:** the first network: fully connected layers over a flat observation, with one output logit per action id (622 actions).
- **Entity transformer:** the second architecture. It encodes the board as one token per card plus scalars, and scores each action by pointing at the card that action targets.

## The models

All exported models live in `models/` (gitignored). `.srnn` files are MLPs, run in plain Rust; `.safetensors` files are entity transformers, run in Rust with candle. Training checkpoints are in `runs/<name>/` and logs in `logs/<name>.log`. Both are gitignored too, about 9 GB in total, and are the only copies.

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
| `phase_g` | 13.4M distilled from `phase_f`, then PPO with `phase_c` in the mix | 13.4M | vs `phase_c`: 64.0% Core, 62.0% mixed; vs `phase_f`: 52.2% Core, 50.6% mixed (2,000) |

`smoke.srnn` and `cap_m.safetensors` in `models/` are test artifacts, not milestones.

## Journal

### Oct 18: scripted bots (`crates/sr-bots`)

- **random:** uniform over legal actions.
- **greedy:** plays everything, uses every non-scrap ability, buys the most expensive card, attacks outposts and then the opponent. The sanity floor.
- **heuristic** (`heuristic.rs`): draw first, play everything, thin the deck early, buy for value and faction synergy, clear outposts, race when lethal. It beats greedy 968-32, and it was the yardstick for all early networks.

All bots, and every network, see only what the player to move could know. No agent inspects the opponent's hand or any draw-pile order.

### Oct 18: training setup

- **Pipeline:** a CleanRL-style single-file masked PPO (`python/starrealms/ppo.py`) over a parallel Rust environment (`crates/sr-py`, built into Python by maturin via `uv sync`). Illegal actions are masked out of the policy.
- **Hardware:** one AMD RX 9070 (16 GB) with ROCm 7.1.
- **Hypothesis:** `torch.compile` and bfloat16 speed up training.
  - **Result:** bfloat16 autocast about doubled throughput with no quality loss, so it's the default (`--bf16`). `torch.compile` was 7× *slower* in the training loop on this GPU, so it was removed.

### Oct 18: Phase A, an MLP against the bots, Core Set

- **Test:** about 100M steps against heuristic, greedy and random bots (`--groups heuristic:192,greedy:32,random:32 --sets core`).
- **Problem found:** the original rating system (openskill) drifted. Ratings fell while win rates rose, so `best.pt` pointed at a weak early checkpoint.
  - **Fix:** ratings are now refitted jointly from all results each time (Bradley-Terry, as BayesElo does) in `league.Ratings`, with a regression test.
- **Result:** `phase_a` beats the heuristic bot **82% [80, 83]**.

### Oct 18-19: Phase B, league self-play, Core Set

- **Hypothesis:** self-play against snapshots beats training only against bots.
- **Result:** confirmed. `phase_b` beats `phase_a` **58%** and the heuristic **85%**.
- **Side finding:** self-play ran at about 25k steps/s versus 60k for bot-only training. The cause was one small GPU forward pass per league opponent per step. Fixed by batching all league opponents into one forward pass.

### Oct 19: Phase C, self-play on mixed sets

- **Result:** on mixed sets, `phase_c` beats `phase_b` **82%** and the heuristic **86%**.
- **Problem:** it got worse on Core-only games (42% vs `phase_b`), because it never saw them. Every later run uses the 30/70 Core/mixed split.

### Oct 19: Phase D, an entity transformer with a pointer action head

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

### Oct 19: longer training of the small transformer

- **Hypothesis:** `phase_d` was still improving, so more training would push it higher.
- **Test:** 67M more PPO steps (`runs/ppo_entity2`) with 448 envs (`--groups self:384,heuristic:64`).
- **Result:** it plateaued at **55-60%** vs `phase_c`. The best checkpoint scored 59.1%; it became `entity2`.
- **Later explanation:** this run was scheduled to anneal from 1e-4 over 400M steps but was stopped at 67M, while the learning rate was still about 0.85e-4. It never got a low-learning-rate stretch (see the Oct 20 hypothesis tests).

### Oct 19: model-size test (distillation only)

- **Hypothesis:** 0.73M parameters is too small to hold more skill.
- **Test:** distill `entity2` into three sizes on the same 10M states (`runs/cap_s`, `cap_m`, `cap_l`):

| Size | Config | Params | KL to teacher | Same move as teacher | Value MSE |
|---|---|---|---|---|---|
| Small | `--d 128 --tf-layers 3 --ff 256 --heads 4` | 0.73M | 0.054 | 89.5% | 0.0048 |
| Medium | `--d 256 --tf-layers 4 --ff 1024 --heads 8` | 4.4M | 0.020 | 94.8% | 0.0026 |
| Large | `--d 384 --tf-layers 6 --ff 1536 --heads 8` | ~11M | 0.012 | 95.1% | 0.0017 |

- **Result:** supported. A model the teacher's own size can't copy it closely, while the larger ones can and were still improving. Imitation alone doesn't beat the teacher, though: all three score about 50% against it. The large model has never been PPO-trained.

### Oct 19: lookahead search without peeking (`crates/sr-bots/src/search.rs`, commit `38ed021`)

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

### Oct 19-20: PPO on the medium model (`phase_e`)

- **Test 1:** `runs/ppo_m`, starting from `cap_m` with the small model's learning rate of 1e-4.
  - **Result:** it got worse. Elo fell from 310 to 262, entropy rose from 0.39 to 0.53, and per-update KL was about 10× the small model's.
  - **Diagnosis:** the learning rate was too high for the bigger model.
- **Test 2:** `runs/m_lr3e5`, at a learning rate of 3e-5.
  - **Result:** steady improvement, to Elo about 355.
  - Stopped at 169M of 250M planned steps, with the gain down to about one point per 50M steps.
  - The best checkpoint (`best_u02850.pt`) became `phase_e`: **62.4%** Core and **57.7%** mixed vs `phase_c`, and **52.9%** vs `entity2`.

### Oct 20: hypothesis tests on the small model (`runs/h_*`)

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
- Diverse *transformer* opponents remain untested (see the known issues in the notes).
- **Bug found along the way:** occasionally the last minibatch held a single example. The standard deviation of one value is NaN, so advantage normalization wrote NaN into the weights; it struck at update 62 with one seed. Fixed in `788807d`: equal-size minibatches via `torch.tensor_split`, plus guards that skip any minibatch with a non-finite loss or gradient and save the first one to `runs/<run>/nonfinite.pt`.

### Oct 20: learning-rate decay on the medium model (`phase_f`)

- **Test:** `runs/m_anneal`, continuing `phase_e` with the learning rate decaying from 1e-5 to zero over 30M steps.
- **Result:**

| Checkpoint | vs `phase_e` (4,000, mixed) | vs `phase_c` mixed | vs `phase_c` Core |
|---|---|---|---|
| Final (`final_u523.pt` = `latest.pt`) | **52.0%** [50.4, 53.5] | **59.5%** | **64.8%** |
| Highest Elo during training (update 300) | 50.5% | 58.0% | 63.3% |

- The final checkpoint was exported as `phase_f`. The gain is smaller than the small model's, as expected: the medium run was already at about 1e-5 when it stopped.

### Oct 20: does the value head rank close decisions?

- **Question:** search failed because the value head seemed no better than the policy at telling moves apart. Is that true on the decisions where the policy itself is unsure?
- **Test:** 800 positions from greedy self-play of `phase_f` (400 Core, 400 mixed) where the two best legal actions were both plausible (top probability between 0.35 and 0.75, second at least 0.15). For each action, play the rest of the turn greedily and read the shaping-corrected value, and also play the line out to the end on the true deck. Script: `/tmp/value_rank.py`. Raw rows: `logs/value_rank.json`.
- **Result:** the head abstained on **98.6%** of positions whose two actions were not the same printed card (absolute gap under 0.001). The median gap was 0. Only **8 of 800** gaps reached 0.02, all of them buys (Blob Fighter against Ram, Imperial Frigate against Blob Fighter, and similar). On the 95 positions where the two full playouts had different winners, the policy's favorite matched the winner **48%** of the time, a coin flip, and the value head expressed a preference on only 4 of them.
- **Conclusion:** at the decisions a search would actually change, this value head does not rank the moves. One greedy playout is also too noisy to label them. Training the policy to imitate the current search would mostly imitate the policy. A better value target has to show up on purchases; that is the only place a gap appeared at all.

### Oct 21: training against `phase_c`

- **Hypothesis:** self-play snapshots never have to answer `phase_c`'s style. The earlier diversity test used weaker MLPs, and `phase_c` itself was only an evaluation opponent.
- **Run:** `runs/m_vs_c`, from `runs/m_lr3e5/best_u02850.pt` (the pre-anneal `phase_e` checkpoint, so `phase_f` is the control). Same 30M-step decay from 1e-5 to zero, with groups `self:288,model:models/phase_c.srnn:96,heuristic:64`.
- **Result:** a tie with `phase_f`. Final checkpoint (`models/m_vs_c.safetensors`) vs `phase_f`: **51.5%** Core [50.0, 53.1] and **50.0%** mixed [48.5, 51.6], both over 4,000 games. The highest-Elo checkpoint was 49.5% against `phase_f` on mixed (2,000 games). Against `phase_c`: **64.0%** Core and **60.4%** mixed (2,000 games), the same as `phase_f` (64.8% and 59.5%). Playing `phase_c` during training did not add skill.
- **Next:** distill `phase_f` into the 13M model and train that. The first distillation attempt ran out of GPU memory with the 4.4M teacher resident; it was retried with minibatch 1,024.

### Oct 21: the 13M model, with `phase_c` added to self-play

- **Distill** (`runs/cap_l_f`): `phase_f` into the 13.4M model, 10M states. Final KL about 0.08, 90% agreement, **50.1%** Core and **51.0%** mixed against the teacher (1,000 games). A looser copy than the earlier distillation of the small model.
- **Self-play** (`runs/l_hold`): learning rate 1e-5, groups `self:384,heuristic:64`, stopped at update 800 (45.9M of 80M steps) so the rest of training could include `phase_c`. Checkpoint: `runs/l_hold/latest.pt`.
- **Continuation** (`runs/l_hold_c`, then `l_anneal_c`): same learning rate, groups `self:288,model:models/phase_c.srnn:96,heuristic:64`. 35M steps flat, then 25M decaying to zero. Exported as `models/phase_g.safetensors`.
- **Result:** a tie with `phase_f`. Over 2,000 games, `phase_g` scores **52.2%** Core [50.0, 54.4] and **50.6%** mixed [48.4, 52.8] against `phase_f`. Against `phase_c`: **64.0%** Core [61.9, 66.1] and **62.0%** mixed [59.9, 64.1]. The mixed gain over `phase_f`'s 59.5% is about two points and the head-to-head does not show it. The 400-game checks during training stayed in the high 50s to low 60s and were not still climbing at the end. Extra parameters and a slice of games against `phase_c` reproduced `phase_f`.

### Oct 21: a purchase-value head on the medium model

- **Hypothesis:** the policy cannot tell a good purchase from a bad one, and neither can the state value. A head that scores each legal purchase from that card's vector, trained on the return of purchases that were actually made, can rank them. After a warmup, that ranking replaces the usual advantage on buy decisions, so the buy policy follows it. Other actions stay on GAE.
- **Run:** `runs/buy_q`, started from `runs/m_anneal/latest.pt` (`phase_f`). Learning rate 1e-5, held flat, 30M steps. Groups `self:288,model:models/phase_c.srnn:96,heuristic:64`, the same set mix as `phase_f`. The head is fit for 40 updates before it begins to steer buys, and fully replaces the buy advantage by update 80. Checkpoints land in `runs/buy_q`.
- **Result:** it made the policy worse, and the score was not a card ranking. Stopped at update 100 (`runs/buy_q/update_00100.pt`, 5.7M steps). The 400-game check: **53.2%** vs `phase_c` on Core and **49.2%** on mixed, against `phase_f`'s 64.8% and 59.5%. Elo fell from 370 to 284. The within-position gap shrank from about 0.07 to 0.04, so buy advantages collapsed toward zero while entropy rose from 0.38 to 0.43.
- **What the score actually learned:** on 300 greedy Core games, binning the purchases it made by score gives outcomes from **−0.67** to **+0.59** (correlation 0.44), and an untrained head does not. That correlation matches the value head (0.45). Subtracting the mean score of the other legal purchases, then removing the value head's prediction, leaves a correlation of **0.006**. The head expressed "who is winning" through whichever card was bought. The card-to-card difference, which is what replaced the buy advantage, predicts nothing. Script: `/tmp/buy_cal.py`.
- **Follow-up** (`runs/buy_q_readout`): the same head, fit to the GAE advantage instead of the return, with every weight except the new query frozen so the policy stayed `phase_f`. 8M steps, learning rate 3e-4, same groups. KL stayed 0 and the 400-game check stayed at `phase_f` (64.5% Core, 62.5% mixed vs `phase_c`; Elo 369 against a start of 370). The card-to-card score still does not predict the game once the value head is removed: correlation **0.05**, and the five bins of purchase score are not ordered by outcome. Agreement between the top score and the policy's buy is 34%. The training correlation between that score and the advantage never rose above about 0.09.
- **Conclusion:** the label we have, the return of the purchase that was actually made, is mostly the position. A dot product with the bought card spends that label on copying the value head. The part that differs between legal cards does not predict who wins, so it cannot replace the buy advantage. Ranking a purchase needs a target that differs between the cards on offer. The `buy_q` code was reverted; the checkpoints and this entry are what remain.

### Oct 21-22: a privileged critic

- **Hypothesis:** most of the noise in a purchase's advantage is the opponent's current hand and the next few draws, which the value head has to average over. A second value head that sees those cards during training, and only during training, should explain more of the Monte-Carlo return and shrink the advantage on close purchases. The policy stays on public information. The trade deck was left out.
- **Test:** `runs/m_critic`, from `runs/m_lr3e5/best_u02850.pt` (the same start as `phase_f`). The policy weights were frozen. Both value heads trained at 3e-5 for 34M decisions. The third measurement ranked the same 800 close decisions as the Oct 20 test, now labeled with 64 paired playouts (`logs/priv_rank.txt`). A separate regression (`logs/priv_diag.txt`) asked how much of `phase_f`'s existing advantages those hidden cards explain as eight totals (trade, combat, authority, draw, and a count, for the hand and each next-five).
- **Result:** the privileged head is worse.
  - Added R² of the hidden totals, on 2.0M decisions: **0.004** overall and **0.004** on purchases. The public scalars alone explain 0.002.
  - Over the last 50 updates, held-out Monte-Carlo explained variance was **0.535 ± 0.029** for the privileged head and **0.576 ± 0.027** for the public head. Gap **−0.041 ± 0.009**.
  - The standard deviation of the purchase advantage was **1.037 ± 0.018** times as large for the privileged head.
  - Of the 800 close decisions, **12** had a playout majority clear at p < 0.05. Nine were "play Scout" against "play Viper". Three were purchases, out of 38 purchase pairs in the set. The public head abstained on 97.5% of positions and agreed with the playouts on 0 of the 2 it called; the privileged head abstained on 93.0% and agreed on 1 of 3. Those counts are too small to rank the heads.
- **Conclusion:** seeing the hand and the next draws did not make a better critic, even as a lower bound from the raw totals. The logged explained variance of ~0.81 was against bootstrapped returns; against the Monte-Carlo return the public head is at about 0.58. The matched policy runs were not started.

### Oct 22: a medium model from scratch (`m_scratch`)

- **Hypothesis:** `phase_f` sits on one distilled lineage that was polished at 1e-5. A fresh medium transformer, trained at 6e-5 decaying to zero over 150M steps with no teacher, might pass it.
- **Result:** it does not. Final checkpoint vs `phase_c` (2,000 games): **61.4%** Core [59.2, 63.5] and **59.5%** mixed [57.3, 61.6]. Vs `phase_f`: **48.4%** Core [46.3, 50.6] and **45.6%** mixed [43.4, 47.7]. `phase_f` is ahead on mixed by more than the interval. Exported as `models/m_scratch.safetensors` from `runs/m_scratch/latest.pt`.

### Oct 22: how much of the match the deal decides

- **Test:** `versus` grew a `mirror` mode. Each seed is played twice with the seats swapped. A pair is player-decided when the same agent wins both games, and deal-decided when the same seat wins both. `phase_f` vs `phase_c`, 2,000 pairs each.
- **Result:**

| Sets | Pair score for `phase_f` | Deal-decided | `phase_f`'s share of player-decided pairs |
|---|---|---|---|
| Core | **64.2%** [62.1, 66.3] | **53.9%** (1,079 / 2,000) | **80.9%** (745 / 921) |
| Mixed | **60.9%** [58.7, 63.0] | **54.9%** (1,098 / 2,000) | **74.2%** (669 / 902) |

- The deal-decided share is a lower bound: only the opening deal is shared, and later shuffles follow the play. Winning every pair that is already player-decided would put the score near 73% if that share stayed put. The 70% target is that region, and `phase_f` already takes most of the pairs a player decides.

## Notes: the recipe that produced `phase_f`

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

## Notes: how models are evaluated

Head-to-head in Rust, on all cores. This is the most reliable measure, and what every number above uses:

```sh
cargo build --release -p sr-bots --example versus
./target/release/examples/versus model:models/phase_f.safetensors model:models/phase_c.srnn 2000 core 7
./target/release/examples/versus model:models/phase_f.safetensors model:models/phase_c.srnn 2000 core,colony_wars,frontiers 7
```

Arguments are `<a> <b> [games] [sets] [seed]`; it prints A's score with a 95% Wilson interval. Agent specs: `random`, `greedy`, `heuristic`, `model:<path>[@temperature]`, `search:<path>@width,samples[,prior,turns]`. To compare a training checkpoint, export it to `/tmp/x.safetensors` first.

`uv run python -m starrealms.evaluate <ckpt> --games 2000` also works from Python. `uv run pytest` checks that the Rust forward passes (MLP and candle) match PyTorch.

## Notes: lessons so far

1. **Architecture mattered most.** Pointing actions at the card they target beat an MLP with eight times the parameters.
2. **Bigger models need smaller learning rates.** 1e-4 made the 4.4M model worse; 3e-5 worked.
3. **Finish the learning-rate schedule.** Stopping a run while the learning rate is high leaves several points on the table. A 30M-step decay to zero is a cheap win.
4. **Don't trust `best.pt` blindly.** It's chosen by Elo from 400-game evaluations, which are noisy. After a full decay, the final checkpoint has been at least as good. Confirm with `versus` at 2,000-4,000 games.
5. **Distillation transfers skill to a new shape but doesn't add it.** Students land at about 50% against the teacher.
6. **Small effects need controls.** The hypothesis runs only made sense against a control trained identically. Compare variants head-to-head at 4,000 games.

## Notes: known issues

- **Transformers are slow as opponents inside the Rust env.** candle's thread pool fights the environment's rayon workers, which makes transformer opponents (and search agents in Python's `arena`) about 150× slower than MLP opponents. `versus` works around it by setting `RAYON_NUM_THREADS=1`. Until this is fixed, the league can't include strong *fixed* transformer opponents such as `entity2` or `phase_e`.
- **Bootstrapping for the waiting seat is crude.** At the end of a rollout window, the seat that isn't to move reuses its own last value prediction, so its final decision gets almost no policy signal (`compute_gae` in `ppo.py`). That's at most one or two decisions per seat per game per update.
- **The value head only models the player to move.** Asking it about the non-moving player is out of distribution. The win meter (`crates/sr-bots/src/meter.rs`) therefore evaluates the mover, averaging over 8 determinizations when the mover is the opponent, and gives the other player the complement.
- **Mid-turn values are biased.** Unspent trade and combat are valued as if they'll be spent well. This matters for search leaves and makes the meter jump at turn boundaries.
- **Memory:** three small-model runs fit on the GPU at once; medium runs should go one at a time. ROCm logs harmless "memory allocation failed ... OOM" warnings early in a run; `expandable_segments` reduces them.
- **Disk:** checkpoints are large (`runs/` is 9 GB). The root partition is small; keep everything on `/home/bigdrive`.

## Notes: ideas not yet tried

In rough order of expected payoff for effort:

1. **PPO on the large model (~11M).** Tried as `phase_g` (distilled from `phase_f`, then PPO with `phase_c` in the mix). It tied `phase_f`. A longer decay, or the same recipe without `phase_c` in the mix, was not run.
2. **Fix the candle/rayon threading conflict,** then train against diverse fixed *transformer* opponents (`entity2`, `phase_e`, `phase_f`). Diversity has only been tested with weak MLP opponents.
3. **Label split purchases with paired playouts, then train the policy on that preference.** The value head cannot rank these moves, and the privileged critic did not fix it, so the label has to be who won when both buys were played out on the same re-deal. In progress (Oct 22). A full search over every move in the turn is the larger version of the same idea, and it stays blocked until a purchase label is shown to exist and to be learnable.
4. **Training speed:** profile the learning phase (attention kernel choice on ROCm; whether `torch.compile` helps the transformer), and overlap rollout with learning (actor/learner) to hide most of the 24% spent playing.
5. **Fix the waiting-seat bootstrap** by scoring the cut-off position from the other seat's side, negating the mover's value.

## Notes: where things are

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
