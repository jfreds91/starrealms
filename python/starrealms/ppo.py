"""Masked PPO with self-play and a snapshot league.

    python -m starrealms.ppo --groups self:192,heuristic:64 --total-steps 20_000_000

Env groups
----------
* ``self``: both seats are decided here. Each env is assigned an opponent for
  the seat it does not train: the current policy (then both seats produce
  training data) with probability ``--p-self``, else a frozen snapshot from the
  league.
* bot names (``heuristic``, ``greedy``, ``random``): the opponent is played in
  Rust; only learner decisions come back.

Credit assignment
-----------------
Decisions of the two seats interleave inside one env, so GAE is computed per
(env, seat) chain: a step's successor is the same seat's next decision. The
terminal reward (+1/-1/0) lands on each seat's last decision of the game.
Optional potential-based shaping uses the authority difference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import N_ACTIONS, OBS_DIM
from .env import LAYOUT, MultiEnv
from .evaluate import evaluate
from .model import ModelConfig, Policy

PHI_INDEX = LAYOUT["scalars"].start + 29  # (my authority - opp authority) / 50


def parse_groups(s: str) -> dict[str, int]:
    out = {}
    for part in s.split(","):
        name, n = part.split(":")
        out[name.strip()] = int(n)
    return out


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None)
    p.add_argument("--groups", default="self:192,heuristic:64")
    p.add_argument("--sets", default="core", help="comma separated: core,colony_wars,frontiers")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total-steps", type=float, default=20e6, help="env decisions (all rows)")
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--shaping", type=float, default=0.3, help="potential-based shaping coefficient (0 = off)")
    p.add_argument("--p-self", type=float, default=0.5, help="self-play envs: P(opponent = current policy)")
    p.add_argument("--snapshot-every", type=int, default=25, help="updates between league snapshots")
    p.add_argument("--league-size", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--eval-opponents", default="heuristic,greedy")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--embed", type=int, default=16)
    p.add_argument("--init", default=None, help="checkpoint to start from")
    p.add_argument("--out", default="runs")
    return p


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


class League:
    def __init__(self, size: int, device):
        self.size = size
        self.device = device
        self.members: list[Policy] = []
        self.tags: list[int] = []
        self.games = defaultdict(int)
        self.wins = defaultdict(float)

    def add(self, policy: Policy, tag: int) -> None:
        m = copy.deepcopy(policy).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.members.append(m)
        self.tags.append(tag)
        if len(self.members) > self.size:
            # keep the first snapshot as an anchor, drop the oldest after it
            del self.members[1], self.tags[1]

    def sample(self, rng: np.random.Generator, p_self: float) -> int:
        if not self.members or rng.random() < p_self:
            return -1
        return int(rng.integers(len(self.members)))


def main(argv=None) -> None:
    args = args_parser().parse_args(argv)
    groups = parse_groups(args.groups)
    sets = [s.strip() for s in args.sets.split(",")]
    run = args.run_name or f"ppo_{'-'.join(sets)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    env = MultiEnv(groups, sets=sets, seed=args.seed)
    N, T = env.num_envs, args.num_steps
    if args.init:
        policy = Policy.load(args.init, dev)
    else:
        policy = Policy(ModelConfig(embed=args.embed, hidden=args.hidden, layers=args.layers)).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    league = League(args.league_size, dev)
    league.add(policy, 0)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"run {run}: {N} envs {groups}, sets {sets}, device {dev}, {n_params/1e6:.2f}M params")

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

    opp_of = np.full(N, -1)
    for i in np.flatnonzero(env.self_play):
        opp_of[i] = league.sample(rng, args.p_self)
    recent = defaultdict(lambda: deque(maxlen=500))
    lengths = deque(maxlen=500)

    log_f = open(out / "log.csv", "w", newline="")
    log_w = None
    res = env.reset()
    num_updates = int(args.total_steps // (N * T))
    steps = 0
    t_start = time.time()
    best = -1.0
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            opt.param_groups[0]["lr"] = args.lr * (1.0 - (update - 1) / num_updates)
        t0 = time.time()
        policy.eval()
        for t in range(T):
            obs = torch.from_numpy(res.obs).to(dev, non_blocking=True)
            mask = torch.from_numpy(res.mask).to(dev, non_blocking=True)
            a, logp, v = policy.act(obs, mask)
            opp_rows = env.self_play & (res.seat != res.learner_seat) & (opp_of >= 0)
            if opp_rows.any():
                for k in np.unique(opp_of[opp_rows]):
                    idx = torch.from_numpy(np.flatnonzero(opp_rows & (opp_of == k))).to(dev)
                    a[idx] = league.members[k].act(obs[idx], mask[idx])[0]
            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = a
            logp_buf[t] = logp
            val_np[t] = v.float().cpu().numpy()
            seat_np[t] = res.seat
            train_np[t] = ~opp_rows
            phi_np[t] = res.obs[:, PHI_INDEX]
            res = env.step(a.cpu().numpy())
            done_np[t] = res.done
            rew_np[t] = res.rewards
            for i in np.flatnonzero(res.done):
                ls = res.done_learner[i]
                w = 1.0 if res.winner[i] == ls else (0.5 if res.winner[i] == -1 else 0.0)
                if env.self_play[i]:
                    k = opp_of[i]
                    key = "self" if k < 0 else "league"
                    if k >= 0:
                        league.games[k] += 1
                        league.wins[k] += w
                    opp_of[i] = league.sample(rng, args.p_self)
                else:
                    key = env.names[env.group[i]]
                recent[key].append(w)
                lengths.append(int(res.turns[i]))
        steps += N * T
        t_roll = time.time() - t0

        with torch.no_grad():
            obs = torch.from_numpy(res.obs).to(dev)
            v_last = policy.value(obs).float().cpu().numpy()
        adv_np = compute_gae(
            val_np, seat_np, done_np, rew_np, phi_np, v_last, res.seat.astype(np.int64),
            res.obs[:, PHI_INDEX], args.gamma, args.gae_lambda, args.shaping,
        )
        ret_np = adv_np + val_np

        sel = torch.from_numpy(train_np.reshape(-1)).to(dev)
        b_obs = obs_buf.view(T * N, OBS_DIM)[sel]
        b_mask = mask_buf.view(T * N, N_ACTIONS)[sel]
        b_act = act_buf.view(-1)[sel]
        b_logp = logp_buf.view(-1)[sel]
        b_adv = torch.from_numpy(adv_np.reshape(-1)).to(dev)[sel]
        b_ret = torch.from_numpy(ret_np.reshape(-1)).to(dev)[sel]
        b_val = torch.from_numpy(val_np.reshape(-1)).to(dev)[sel]
        B = b_obs.shape[0]

        policy.train()
        t1 = time.time()
        clipfracs, approx_kl = [], 0.0
        stop = False
        for epoch in range(args.epochs):
            perm = torch.randperm(B, device=dev)
            for start in range(0, B, args.minibatch):
                mb = perm[start : start + args.minibatch]
                logits, newv = policy(b_obs[mb], b_mask[mb])
                logp_all = torch.log_softmax(logits, -1)
                newlogp = logp_all.gather(1, b_act[mb, None]).squeeze(1)
                probs = logp_all.exp()
                entropy = -(probs * logp_all.clamp(min=-1e4)).sum(-1)
                logratio = newlogp - b_logp[mb]
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
                loss = pg - args.ent_coef * ent + args.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                if args.target_kl and approx_kl > args.target_kl:
                    stop = True
                    break
            if stop:
                break
        t_train = time.time() - t1

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        row = {
            "update": update,
            "steps": steps,
            "time": round(time.time() - t_start, 1),
            "sps": int(N * T / (t_roll + t_train)),
            "roll_s": round(t_roll, 2),
            "train_s": round(t_train, 2),
            "batch": B,
            "pg_loss": round(pg.item(), 4),
            "v_loss": round(v_loss.item(), 4),
            "entropy": round(ent.item(), 3),
            "kl": round(approx_kl, 4),
            "clipfrac": round(float(np.mean(clipfracs)), 3),
            "explained_var": round(float(explained), 3),
            "turns": round(float(np.mean(lengths)), 1) if lengths else 0,
        }
        for k, q in sorted(recent.items()):
            row[f"wr_{k}"] = round(float(np.mean(q)), 3)

        if update % args.snapshot_every == 0:
            league.add(policy, update)
        if update % args.eval_every == 0 or update == num_updates:
            policy.eval()
            for opp in [o for o in args.eval_opponents.split(",") if o]:
                wr = evaluate(policy, opp, args.eval_games, sets, dev, greedy=True, seed=update)
                row[f"eval_{opp}"] = round(wr, 3)
            policy.save(out / "latest.pt", update=update, steps=steps, args=vars(args))
            score = row.get("eval_heuristic", -1)
            if score >= best:
                best = score
                policy.save(out / "best.pt", update=update, steps=steps, args=vars(args))
            policy.save(out / f"update_{update:05d}.pt", update=update, steps=steps)

        if log_w is None:
            keys = ["self", "league"] + [n for n in env.names if n != "self"]
            fields = [k for k in row if not k.startswith(("wr_", "eval_"))]
            fields += [f"wr_{k}" for k in keys] + [f"eval_{o}" for o in args.eval_opponents.split(",") if o]
            log_w = csv.DictWriter(log_f, fieldnames=fields, restval="", extrasaction="ignore")
            log_w.writeheader()
        log_w.writerow(row)
        log_f.flush()
        wr = " ".join(f"{k}={v}" for k, v in row.items() if k.startswith(("wr_", "eval_")))
        print(
            f"[{update}/{num_updates}] steps {steps/1e6:.2f}M sps {row['sps']} "
            f"ent {row['entropy']} kl {row['kl']} ev {row['explained_var']} turns {row['turns']} {wr}",
            flush=True,
        )
    log_f.close()
    print(f"done: {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
