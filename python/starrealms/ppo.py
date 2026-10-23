"""Masked PPO with self-play and a snapshot league.

    python -m starrealms.ppo --groups self:192,heuristic:64 --total-steps 20e6

Env groups: in ``self`` envs both seats are decided here. Each env is assigned
an opponent for the seat it does not train: the current policy (then both seats
produce training data) with probability ``--p-self``, else a frozen snapshot
from the league, drawn by prioritized fictitious self-play. Bot names
(``heuristic``, ``greedy``, ``random``) put the opponent inside Rust, so only
learner decisions come back.

Evaluation: every ``--eval-every`` updates the policy plays the eval bots and a
few league snapshots; Elo ratings for every checkpoint and bot are refitted
jointly from all those results (see ``league.Ratings``), relative to the
heuristic bot. ``best.pt`` is the highest-rated checkpoint; ``ratings.json``
holds the ladder.

Credit assignment: decisions of the two seats interleave inside one env, so
GAE is computed per (env, seat) chain, where a step's successor is the same
seat's next decision. The terminal reward (+1/-1/0) lands on each seat's last
decision of the game. Optional potential-based shaping uses the authority
difference.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from . import N_ACTIONS, OBS_DIM, PRIV_IDS
from . import tokens as tk
from .env import SCALAR, MultiEnv, parse_sets_mix, sets_tag
from .evaluate import evaluate
from .league import FixedPool, League, Ratings, snapshot_name
from .model import EntityConfig, EntityPolicy, ModelConfig, Policy, load

PHI_INDEX = SCALAR["authority_diff"]  # (my authority - opp authority) / 50
BOTS = ("random", "greedy", "heuristic")


@dataclass
class Args:
    run_name: str | None = None
    """defaults to ppo_<sets>_<timestamp>"""
    out: str = "runs"
    """run directory parent (checkpoints and TensorBoard events)"""
    track: bool = False
    """also log to Weights & Biases"""
    wandb_project: str = "starrealms"
    wandb_entity: str | None = None

    groups: str = "self:192,heuristic:64"
    """env groups as name:count; 'self' is self-play, other names are Rust bots or model:<path>"""
    sets: str = "core"
    """comma separated: core,colony_wars,frontiers"""
    sets_mix: str | None = None
    """split every env group across card-set selections, e.g. 'core:0.3,core+colony_wars+frontiers:0.7';
    overrides --sets, and evaluation then reports each selection"""
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = torch.cuda.is_available()
    """bfloat16 autocast for forward passes (about 2x throughput on the RX 9070)"""

    total_steps: float = 20e6
    """env decisions, all rows"""
    num_steps: int = 128
    """rollout length per env"""
    lr: float = 2.5e-4
    anneal_lr: bool = True
    warmup_updates: int = 0
    """linear learning-rate warmup"""
    gamma: float = 0.995
    gae_lambda: float = 0.95
    epochs: int = 3
    minibatch: int = 4096
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    """decoupled (AdamW) weight decay; 0 is plain Adam"""
    target_kl: float = 0.05
    """stop the epoch loop once approx KL exceeds this (0 = never)"""
    shaping: float = 0.3
    """potential-based shaping coefficient (0 = off)"""
    privileged: bool = False
    """entity models: critic sees the opponent's hand and the next draws; the policy does not"""
    vf_coef_pub: float = 0.25
    """weight of the public value head when --privileged; it keeps serving the exported model"""
    critic_only: bool = False
    """fit the value heads and privileged tokens, and leave the policy weights untouched"""
    critic_patience: int = 0
    """critic-only: stop after this many updates with no MC explained-variance gain (0 = off)"""

    p_self: float = 0.5
    """self-play envs: P(opponent = current policy)"""
    snapshot_every: int = 100
    """updates between league snapshots; a multiple of eval_every, since snapshots are rated checkpoints"""
    league_size: int = 12
    pfsp_power: float = 2.0
    """league opponents are drawn with weight (1 - learner score) ** pfsp_power; 0 = uniform"""
    eval_every: int = 100
    """updates between evaluations; each one rates a checkpoint and may replace best.pt"""
    eval_games: int = 500
    eval_opponents: str = "heuristic,greedy"
    """bot names or model:<path> (exported models, played in Rust); all enter the Elo fit"""

    arch: Literal["mlp", "entity"] = "mlp"
    """mlp: `Policy` (--hidden/--layers/--embed); entity: `EntityPolicy` (--d/--heads/--tf-layers/--ff)"""
    hidden: int = 512
    layers: int = 2
    embed: int = 16
    d: int = 128
    heads: int = 4
    tf_layers: int = 3
    ff: int = 256
    init: str | None = None
    """checkpoint to start from (its architecture wins over --arch)"""


def opp_name(spec: str) -> str:
    """Rating/log name of an eval opponent: a bot name, or the file stem of `model:<path>[@t]`."""
    if spec.startswith("model:"):
        return Path(spec[len("model:") :].rsplit("@", 1)[0]).stem
    return spec


def parse_groups(s: str) -> dict[str, int]:
    out = {}
    for part in s.split(","):
        name, n = part.rsplit(":", 1)
        out[name.strip()] = int(n)
    return out


def compute_gae(values, seats, dones, rewards, phi, v_last, seat_last, phi_last, gamma, lam, shaping):
    """Per-(env, seat) GAE. Arrays are [T, N] (rewards [T, N, 2]); *_last are [N]."""
    T, N = values.shape
    rows = np.arange(N)
    adv = np.zeros((T, N), np.float32)
    next_val = np.zeros((N, 2), np.float32)
    next_adv = np.zeros((N, 2), np.float32)
    next_phi = np.zeros((N, 2), np.float32)
    valid = np.zeros((N, 2), bool)
    term = np.zeros((N, 2), bool)
    term_r = np.zeros((N, 2), np.float32)
    next_val[rows, seat_last] = v_last
    next_phi[rows, seat_last] = phi_last
    valid[rows, seat_last] = True
    for t in range(T - 1, -1, -1):
        d = dones[t]
        term[d] = True
        term_r[d] = rewards[t][d]
        s = seats[t]
        v = values[t]
        is_term = term[rows, s]
        ok = valid[rows, s] & ~is_term
        nv = np.where(is_term, 0.0, np.where(ok, next_val[rows, s], v))
        nphi = np.where(is_term, 0.0, np.where(ok, next_phi[rows, s], phi[t]))
        r = np.where(is_term, term_r[rows, s], 0.0) + shaping * (gamma * nphi - phi[t])
        na = np.where(ok, next_adv[rows, s], 0.0)
        a = r + gamma * nv - v + gamma * lam * na
        adv[t] = a
        next_val[rows, s] = v
        next_adv[rows, s] = a
        next_phi[rows, s] = phi[t]
        valid[rows, s] = True
        term[rows, s] = False
    return adv


def explained_var(pred, target) -> float:
    var = float(np.var(target))
    if var == 0:
        return float("nan")
    return float(1 - np.var(target - pred) / var)


_PURCHASE: np.ndarray | None = None


def purchase_index() -> np.ndarray:
    global _PURCHASE
    if _PURCHASE is None:
        idx = []
        for name in ("buy", "buy_explorer"):
            start, n = tk.ACTIONS[name]
            idx.extend(range(start, start + n))
        _PURCHASE = np.asarray(idx, np.int64)
    return _PURCHASE


def buy_advantage_std(adv, actions, mask, train) -> float:
    """Std of `adv` on learner decisions with at least two legal purchases."""
    cols = purchase_index()
    act = np.asarray(actions).reshape(-1)
    keep = np.asarray(train).reshape(-1) & np.isin(act, cols)
    keep &= np.asarray(mask).reshape(-1, mask.shape[-1])[:, cols].sum(1) >= 2
    if int(keep.sum()) < 2:
        return float("nan")
    return float(np.asarray(adv).reshape(-1)[keep].std())


_CRITIC_TRAIN = ("priv_type_emb.", "critic_tok", "value_priv1.", "value_priv2.", "value1.", "value2.")


def main(args: Args) -> Path:
    if args.snapshot_every % args.eval_every:
        raise ValueError("snapshot_every must be a multiple of eval_every")
    groups = parse_groups(args.groups)
    sets = [s.strip() for s in args.sets.split(",")]
    mix = parse_sets_mix(args.sets_mix) if args.sets_mix else None
    eval_sets = [s for s, _ in mix] if mix else [sets]
    run = args.run_name or f"ppo_{'-'.join(sets)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    env = MultiEnv(groups, sets=sets, seed=args.seed, sets_mix=mix, privileged=args.privileged)
    N, T = env.num_envs, args.num_steps
    if args.init:
        policy = load(args.init, dev)
    elif args.arch == "entity":
        policy = EntityPolicy(EntityConfig(d=args.d, layers=args.tf_layers, heads=args.heads, ff=args.ff)).to(dev)
    else:
        policy = Policy(ModelConfig(embed=args.embed, hidden=args.hidden, layers=args.layers)).to(dev)
    if args.privileged and not isinstance(policy, EntityPolicy):
        raise ValueError("--privileged requires an entity model")
    if args.critic_only and not args.privileged:
        raise ValueError("--critic-only requires --privileged")
    if args.critic_only:
        for name, p in policy.named_parameters():
            p.requires_grad_(name.startswith(_CRITIC_TRAIN))
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, eps=1e-5, weight_decay=args.weight_decay)
    def amp():
        return torch.autocast(dev.type, torch.bfloat16, enabled=args.bf16)

    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"run {run}: {N} envs {dict(zip(env.labels, env.sizes))}, device {dev}, "
          f"{policy.ARCH} {n_params/1e6:.2f}M params, {n_train/1e6:.2f}M trainable"
          + (" privileged" if args.privileged else "")
          + (" critic-only" if args.critic_only else ""))

    league = League(args.league_size, args.pfsp_power)
    fixed_pool = FixedPool(env.fixed_paths, dev) if env.fixed_paths else None
    ratings = Ratings()
    for s in eval_sets:
        ratings.calibrate_bots(list(BOTS), s)
    eval_opponents = [o for o in args.eval_opponents.split(",") if o]

    def tagged(key: str, s: list[str]) -> str:
        return f"{key}@{sets_tag(s)}" if mix else key

    def rate_checkpoint(update: int) -> dict[str, float]:
        """Evaluate the current policy against the eval bots and a few league snapshots, and
        record the results for `u<update>`."""
        name = snapshot_name(update)
        stats = {}
        policy.eval()
        with amp():
            for s in eval_sets:
                for opp in eval_opponents:
                    r = evaluate(policy, opp, args.eval_games, s, dev, greedy=True, seed=update)
                    ratings.games(name, opp_name(opp), r.wins, r.draws, r.losses)
                    stats[tagged(f"eval/{opp_name(opp)}", s)] = r.winrate
                    stats[tagged(f"eval_lower95/{opp_name(opp)}", s)] = r.interval()[0]
                league_scores = []
                for tag in league.eval_members():
                    r = evaluate(policy, league.members[tag], args.eval_games, s, dev, greedy=True, seed=update)
                    ratings.games(name, snapshot_name(tag), r.wins, r.draws, r.losses)
                    league_scores.append(r.winrate)
                if league_scores:
                    stats[tagged("eval/league", s)] = float(np.mean(league_scores))
        ratings.ensure(name)
        stats["rating/learner"] = ratings.rating(name)
        for b in BOTS:
            stats[f"rating/{b}"] = ratings.rating(b)
        ratings.save(out / "ratings.json")
        return stats

    first = rate_checkpoint(0)
    print(f"initial policy: rating {first['rating/learner']:.1f}; bots "
          + " ".join(f"{b}={ratings.rating(b):.1f}" for b in BOTS))
    league.add(policy, 0)

    config = dataclasses.asdict(args)
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run, config=config,
                   sync_tensorboard=True, save_code=True)
    writer = SummaryWriter(str(out))
    writer.add_text("args", "\n".join(f"- `{k}`: {v}" for k, v in config.items()))

    obs_buf = torch.zeros((T, N, OBS_DIM), device=dev)
    mask_buf = torch.zeros((T, N, N_ACTIONS), dtype=torch.bool, device=dev)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_buf = torch.zeros((T, N), device=dev)
    val_np = np.zeros((T, N), np.float32)
    seat_np = np.zeros((T, N), np.int64)
    train_np = np.zeros((T, N), bool)
    done_np = np.zeros((T, N), bool)
    rew_np = np.zeros((T, N, 2), np.float32)
    phi_np = np.zeros((T, N), np.float32)
    val_pub_np = np.zeros((T, N), np.float32) if args.privileged else None
    priv_buf = torch.zeros((T, N, PRIV_IDS), dtype=torch.int16, device=dev) if args.privileged else None

    opp_of = np.full(N, -1)
    for i in np.flatnonzero(env.self_play):
        opp_of[i] = league.sample(rng, args.p_self)
    recent = defaultdict(lambda: deque(maxlen=500))
    lengths = deque(maxlen=500)

    res = env.reset()
    num_updates = int(args.total_steps // (N * T))
    steps = 0
    t_start = time.time()
    best: str | None = None  # name of the checkpoint saved as best.pt; its rating moves with each refit
    best_mc = {"priv": -1e9, "pub": -1e9}
    since_mc = 0
    halt = False
    for update in range(1, num_updates + 1):
        lr = args.lr * (1.0 - (update - 1) / num_updates) if args.anneal_lr else args.lr
        if update <= args.warmup_updates:
            lr *= update / args.warmup_updates
        opt.param_groups[0]["lr"] = lr
        t0 = time.time()
        policy.eval()
        for t in range(T):
            obs = torch.from_numpy(res.obs).to(dev, non_blocking=True)
            mask = torch.from_numpy(res.mask).to(dev, non_blocking=True)
            priv_t = None
            if args.privileged:
                priv_t = torch.from_numpy(np.ascontiguousarray(res.priv)).to(dev, non_blocking=True)
                priv_buf[t] = priv_t
            with torch.no_grad(), amp():
                if args.privileged:
                    logits, v_pub, v_priv = policy(obs, mask, priv_t)
                    logits = logits.float()
                    a = torch.distributions.Categorical(logits=logits).sample()
                    logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
                    v = v_priv.float()
                    val_pub_np[t] = v_pub.float().cpu().numpy()
                else:
                    a, logp, v = policy.act(obs, mask)
                opp_rows = env.self_play & (res.seat != res.learner_seat) & (opp_of >= 0)
                fixed_rows = env.fixed & (res.seat != res.learner_seat)
                if opp_rows.any():
                    idx = torch.from_numpy(np.flatnonzero(opp_rows)).to(dev)
                    a[idx] = league.act(obs[idx], mask[idx], opp_of[opp_rows])
                if fixed_rows.any():
                    idx = torch.from_numpy(np.flatnonzero(fixed_rows)).to(dev)
                    a[idx] = fixed_pool.act(obs[idx], mask[idx], env.fixed_id[fixed_rows])
            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = a
            logp_buf[t] = logp
            val_np[t] = v.float().cpu().numpy()
            seat_np[t] = res.seat
            train_np[t] = ~(opp_rows | fixed_rows)
            phi_np[t] = res.obs[:, PHI_INDEX]
            res = env.step(a.cpu().numpy())
            done_np[t] = res.done
            rew_np[t] = res.rewards
            for i in np.flatnonzero(res.done):
                ls = res.done_learner[i]
                w = 1.0 if res.winner[i] == ls else (0.5 if res.winner[i] == -1 else 0.0)
                if env.self_play[i]:
                    k = opp_of[i]
                    key = env.labels[env.group[i]].replace("self", "self" if k < 0 else "league", 1)
                    if k >= 0:
                        league.record(k, w)
                    opp_of[i] = league.sample(rng, args.p_self)
                else:
                    key = env.labels[env.group[i]]
                recent[key].append(w)
                lengths.append(int(res.turns[i]))
        steps += N * T
        t_roll = time.time() - t0

        with torch.no_grad(), amp():
            obs = torch.from_numpy(res.obs).to(dev)
            seat_last = res.seat.astype(np.int64)
            phi_last = res.obs[:, PHI_INDEX]
            if args.privileged:
                priv_last = torch.from_numpy(np.ascontiguousarray(res.priv)).to(dev)
                h_last = policy.encode(obs, priv_last)[0]
                v_last = policy._value_priv(h_last).float().cpu().numpy()
                v_last_pub = policy._value(h_last).float().cpu().numpy()
            else:
                v_last = policy.value(obs).float().cpu().numpy()
                v_last_pub = None
        def gae(values, v_boot, lam):
            return compute_gae(
                values, seat_np, done_np, rew_np, phi_np, v_boot, seat_last, phi_last,
                args.gamma, lam, args.shaping,
            )

        adv_np = gae(val_np, v_last, args.gae_lambda)
        ret_np = adv_np + val_np
        # Held-out critic diagnostics, from the values predicted before this update.
        if args.privileged:
            train = train_np
            adv_pub = gae(val_pub_np, v_last_pub, args.gae_lambda)
            mc_priv = gae(val_np, v_last, 1.0) + val_np
            mc_pub = gae(val_pub_np, v_last_pub, 1.0) + val_pub_np
            ev_priv = explained_var(val_np[train], (adv_np + val_np)[train])
            ev_pub = explained_var(val_pub_np[train], (adv_pub + val_pub_np)[train])
            ev_mc_priv = explained_var(val_np[train], mc_priv[train])
            ev_mc_pub = explained_var(val_pub_np[train], mc_pub[train])
            acts_np = act_buf.cpu().numpy()
            masks_np = mask_buf.cpu().numpy()
            std_priv = buy_advantage_std(adv_np, acts_np, masks_np, train)
            std_pub = buy_advantage_std(adv_pub, acts_np, masks_np, train)
        else:
            ev_priv = ev_pub = ev_mc_priv = ev_mc_pub = std_priv = std_pub = float("nan")

        sel = torch.from_numpy(train_np.reshape(-1)).to(dev)
        b_obs = obs_buf.view(T * N, OBS_DIM)[sel]
        b_mask = mask_buf.view(T * N, N_ACTIONS)[sel]
        b_act = act_buf.view(-1)[sel]
        b_logp = logp_buf.view(-1)[sel]
        b_adv = torch.from_numpy(adv_np.reshape(-1)).to(dev)[sel]
        b_ret = torch.from_numpy(ret_np.reshape(-1)).to(dev)[sel]
        b_val = torch.from_numpy(val_np.reshape(-1)).to(dev)[sel]
        b_priv = priv_buf.view(T * N, PRIV_IDS)[sel] if args.privileged else None
        B = b_obs.shape[0]

        policy.train()
        t1 = time.time()
        clipfracs, approx_kl, skipped = [], 0.0, 0
        stop = False
        for epoch in range(args.epochs):
            perm = torch.randperm(B, device=dev)
            # equal chunks of at least `minibatch` rows; a leftover single row would make the
            # advantage std NaN
            for mb in torch.tensor_split(perm, max(1, B // args.minibatch)):
                with amp():
                    if args.privileged:
                        logits, v_pub_mb, newv = policy(b_obs[mb], b_mask[mb], b_priv[mb])
                        v_pub_mb = v_pub_mb.float()
                    else:
                        logits, newv = policy(b_obs[mb], b_mask[mb])
                        v_pub_mb = None
                logits, newv = logits.float(), newv.float()
                logp_all = torch.log_softmax(logits, -1)
                newlogp = logp_all.gather(1, b_act[mb, None]).squeeze(1)
                probs = logp_all.exp()
                entropy = -(probs * logp_all.clamp(min=-1e4)).sum(-1)
                # unbounded, a near-impossible sampled action can overflow the ratio to inf
                logratio = (newlogp - b_logp[mb]).clamp(-20.0, 20.0)
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip).float().mean().item())
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_clip = b_val[mb] + (newv - b_val[mb]).clamp(-args.clip, args.clip)
                v_loss = 0.5 * torch.max((newv - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = args.vf_coef * v_loss
                if v_pub_mb is not None:
                    loss = loss + args.vf_coef_pub * (v_pub_mb - b_ret[mb]).pow(2).mean()
                if not args.critic_only:
                    loss = loss + pg - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    gnorm = nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                    skipped += 1
                    dump = out / "nonfinite.pt"
                    if not dump.exists():
                        torch.save({"obs": b_obs[mb].cpu(), "mask": b_mask[mb].cpu(), "act": b_act[mb].cpu(),
                                    "logp": b_logp[mb].cpu(), "adv": b_adv[mb].cpu(), "ret": b_ret[mb].cpu(),
                                    "val": b_val[mb].cpu(), "update": update, "loss": loss.item(),
                                    "state": policy.state_dict()}, dump)
                    continue
                opt.step()
                if not args.critic_only and args.target_kl and approx_kl > args.target_kl:
                    stop = True
                    break
            if stop:
                break
        t_train = time.time() - t1

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(N * T / (t_roll + t_train))
        scalars = {
            "charts/learning_rate": opt.param_groups[0]["lr"],
            "charts/sps": sps,
            "charts/rollout_s": t_roll,
            "charts/train_s": t_train,
            "charts/batch": B,
            "charts/game_turns": float(np.mean(lengths)) if lengths else 0.0,
            "losses/policy": pg.item(),
            "losses/value": v_loss.item(),
            "losses/entropy": ent.item(),
            "losses/approx_kl": approx_kl,
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": float(explained),
            "losses/skipped_minibatches": skipped,
        }
        if args.privileged:
            scalars.update({
                "losses/ev_priv": ev_priv,
                "losses/ev_pub": ev_pub,
                "losses/ev_mc_priv": ev_mc_priv,
                "losses/ev_mc_pub": ev_mc_pub,
                "losses/adv_std_buy_priv": std_priv,
                "losses/adv_std_buy_pub": std_pub,
            })
            if args.critic_only and args.critic_patience:
                improved = False
                for key, val in (("priv", ev_mc_priv), ("pub", ev_mc_pub)):
                    if np.isfinite(val) and val > best_mc[key] + 1e-4:
                        best_mc[key] = val
                        improved = True
                since_mc = 0 if improved else since_mc + 1
                if since_mc >= args.critic_patience:
                    halt = True
        for k, q in sorted(recent.items()):
            scalars[f"winrate/{k}"] = float(np.mean(q))

        if update % args.eval_every == 0 or update == num_updates or halt:
            scalars.update(rate_checkpoint(update))
            rating = scalars["rating/learner"]
            policy.save(out / "latest.pt", update=update, steps=steps, rating=rating, args=config)
            if best is None or rating >= ratings.rating(best):
                best = snapshot_name(update)
                policy.save(out / "best.pt", update=update, steps=steps, rating=rating, args=config)
            policy.save(out / f"update_{update:05d}.pt", update=update, steps=steps, rating=rating)
            if update % args.snapshot_every == 0:
                evicted = league.add(policy, update)
                if evicted is not None:
                    for i in np.flatnonzero(opp_of == evicted):
                        opp_of[i] = league.sample(rng, args.p_self)
            scalars["league/size"] = len(league.members)
            scalars["league/hardest_score"] = min(league.score.values())

        for k, v in scalars.items():
            writer.add_scalar(k, v, steps)
        def summary(prefix):
            return " ".join(f"{k[len(prefix):]}={v:.3f}" for k, v in scalars.items() if k.startswith(prefix))

        wr, ev = summary("winrate/"), summary("eval/")
        if ev:
            ev += f" elo={scalars['rating/learner']:.0f} best={best}@{ratings.rating(best):.0f}"
        print(
            f"[{update}/{num_updates}] steps {steps/1e6:.2f}M sps {sps} ent {ent.item():.3f} "
            f"kl {approx_kl:.4f} ev {explained:.3f} turns {scalars['charts/game_turns']:.1f}"
            + (f" ev_mc {ev_mc_priv:.3f}/{ev_mc_pub:.3f} buy_std {std_priv:.3f}/{std_pub:.3f}"
               if args.privileged else "")
            + (f" skipped {skipped}" if skipped else "")
            + f" | train {wr}"
            + (f" | eval {ev}" if ev else ""),
            flush=True,
        )
        if halt:
            print(f"critic patience {args.critic_patience}: stopping at update {update}", flush=True)
            break
    writer.close()
    if args.track:
        wandb.finish()
    print(f"done: {out}")
    return out


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main(tyro.cli(Args, description=__doc__))
