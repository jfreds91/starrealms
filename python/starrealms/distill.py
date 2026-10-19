"""Train a new network to imitate an existing checkpoint (policy KL + value regression).

    python -m starrealms.distill --teacher runs/phase_c/best.pt --arch entity --run-name distill_entity

States come from self-play games on `--sets-mix`. Moves are made by the teacher with
probability `beta` (annealed from `--beta-start` to 0 over the first half of the run) and by
the student otherwise, so the student also learns the teacher's answer in the positions its
own mistakes lead to (DAgger). Every state is labelled with the teacher's full action
distribution and value. Every `--eval-every` iterations the student plays the teacher head
to head; training stops once it scores `--target` (or after `--total-steps` states).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .env import MultiEnv, parse_sets_mix
from .evaluate import evaluate
from .model import EntityConfig, EntityPolicy, ModelConfig, Policy, load


@dataclass
class Args:
    teacher: str = "runs/phase_c/best.pt"
    run_name: str = "distill"
    out: str = "runs"
    arch: Literal["mlp", "entity"] = "entity"
    hidden: int = 1024
    layers: int = 3
    embed: int = 16
    d: int = 128
    heads: int = 4
    tf_layers: int = 3
    ff: int = 256

    sets_mix: str = "core:0.3,core+colony_wars+frontiers:0.7"
    envs: int = 512
    num_steps: int = 32
    """states collected per env per iteration"""
    epochs: int = 2
    minibatch: int = 4096
    lr: float = 3e-4
    vf_coef: float = 0.5
    beta_start: float = 1.0
    total_steps: float = 30e6
    eval_every: int = 25
    eval_games: int = 1000
    target: float = 0.47
    """stop once the student's head-to-head score against the teacher reaches this"""
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = torch.cuda.is_available()


def main(args: Args) -> Path:
    out = Path(args.out) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    mix = parse_sets_mix(args.sets_mix)
    env = MultiEnv({"self": args.envs}, seed=args.seed, sets_mix=mix)
    N, T = env.num_envs, args.num_steps

    teacher = load(args.teacher, dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if args.arch == "entity":
        student = EntityPolicy(EntityConfig(d=args.d, layers=args.tf_layers, heads=args.heads, ff=args.ff))
    else:
        student = Policy(ModelConfig(embed=args.embed, hidden=args.hidden, layers=args.layers))
    student = student.to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr, eps=1e-5)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"distill {args.teacher} -> {student.ARCH} {n_params / 1e6:.2f}M params, {N} envs", flush=True)
    writer = SummaryWriter(str(out))

    def amp():
        return torch.autocast(dev.type, torch.bfloat16, enabled=args.bf16)

    iters = int(args.total_steps // (N * T))
    res = env.reset()
    obs_buf = torch.zeros((T, N, res.obs.shape[1]), device=dev)
    mask_buf = torch.zeros((T, N, res.mask.shape[1]), dtype=torch.bool, device=dev)
    best, steps, t_start = -1.0, 0, time.time()
    for it in range(1, iters + 1):
        beta = max(0.0, args.beta_start * (1 - 2 * (it - 1) / iters))
        lr = args.lr * (1 - (it - 1) / iters)
        opt.param_groups[0]["lr"] = lr
        student.eval()
        for t in range(T):
            obs = torch.from_numpy(res.obs).to(dev, non_blocking=True)
            mask = torch.from_numpy(res.mask).to(dev, non_blocking=True)
            with amp():
                a = teacher.act(obs, mask)[0]
                use_student = torch.from_numpy(rng.random(N) >= beta).to(dev)
                if use_student.any():
                    a = torch.where(use_student, student.act(obs, mask)[0], a)
            obs_buf[t], mask_buf[t] = obs, mask
            res = env.step(a.cpu().numpy())
        steps += N * T

        b_obs = obs_buf.view(N * T, -1)
        b_mask = mask_buf.view(N * T, -1)
        with torch.no_grad(), amp():
            t_logits, t_value = [], []
            for s in range(0, N * T, 8192):
                lg, v = teacher(b_obs[s : s + 8192], b_mask[s : s + 8192])
                t_logits.append(lg.float())
                t_value.append(v.float())
            t_logp = torch.log_softmax(torch.cat(t_logits), -1)
            t_value = torch.cat(t_value)
        student.train()
        for _ in range(args.epochs):
            perm = torch.randperm(N * T, device=dev)
            for s in range(0, N * T, args.minibatch):
                mb = perm[s : s + args.minibatch]
                with amp():
                    logits, v = student(b_obs[mb], b_mask[mb])
                logp = torch.log_softmax(logits.float(), -1)
                tl = t_logp[mb]
                kl = (tl.exp() * (tl - logp)).masked_fill(~b_mask[mb], 0.0).sum(-1).mean()
                v_loss = F.mse_loss(v.float(), t_value[mb])
                loss = kl + args.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            agree = (logp.argmax(-1) == tl.argmax(-1)).float().mean().item()
        scalars = {"distill/kl": kl.item(), "distill/value_mse": v_loss.item(), "distill/argmax_agree": agree,
                   "distill/beta": beta, "distill/lr": lr}
        line = (f"[{it}/{iters}] states {steps / 1e6:.1f}M kl {kl.item():.4f} vmse {v_loss.item():.4f} "
                f"agree {agree:.3f} beta {beta:.2f} {steps / (time.time() - t_start):.0f} states/s")
        if it % args.eval_every == 0 or it == iters:
            student.eval()
            scores = []
            with amp():
                for s, _ in mix:
                    r = evaluate(student, teacher, args.eval_games, s, dev, greedy=True, seed=it)
                    scalars[f"eval/teacher@{'+'.join(s)}"] = r.winrate
                    scores.append(r.winrate)
                    line += f" | vs teacher {'+'.join(s)} {r}"
            score = float(np.mean(scores))
            student.save(out / "latest.pt", steps=steps, score_vs_teacher=score, teacher=args.teacher)
            if score > best:
                best = score
                student.save(out / "best.pt", steps=steps, score_vs_teacher=score, teacher=args.teacher)
            if score >= args.target:
                print(line, flush=True)
                print(f"reached {score:.3f} >= {args.target} against the teacher", flush=True)
                break
        for k, v in scalars.items():
            writer.add_scalar(k, v, steps)
        print(line, flush=True)
    writer.close()
    print(f"done: {out} (best score vs teacher {best:.3f})")
    return out


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main(tyro.cli(Args, description=__doc__))
