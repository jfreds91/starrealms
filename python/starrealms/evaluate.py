"""Evaluate a checkpoint against the scripted bots.

    python -m starrealms.evaluate runs/<run>/best.pt --opponents heuristic,greedy,random --games 2000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ._engine import VecEnv


@torch.no_grad()
def evaluate(policy, opponent: str, games: int, sets, device, greedy: bool = True, seed: int = 0) -> float:
    """Win rate (draws count 1/2) of `policy` vs a Rust bot, alternating seats."""
    n = min(games, 256)
    env = VecEnv(n, list(sets), 10_000_019 + seed, opponent)
    obs, mask, _, _ = env.reset()
    score, finished = 0.0, 0
    started = n
    active = np.ones(n, bool)  # count only the first `games` games
    while finished < games:
        o = torch.from_numpy(obs).to(device)
        m = torch.from_numpy(mask).to(device)
        a, _, _ = policy.act(o, m, greedy=greedy)
        obs, mask, _, _, _, done, winner, _, done_learner = env.step(a.cpu().numpy())
        for i in np.flatnonzero(done & active):
            finished += 1
            score += 1.0 if winner[i] == done_learner[i] else (0.5 if winner[i] == -1 else 0.0)
            if started < games:
                started += 1
            else:
                active[i] = False
    return score / finished


def main(argv=None) -> None:
    from .model import Policy

    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--opponents", default="heuristic,greedy,random")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--sets", default="core")
    p.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    policy = Policy.load(args.checkpoint, args.device).eval()
    sets = args.sets.split(",")
    for opp in args.opponents.split(","):
        wr = evaluate(policy, opp, args.games, sets, args.device, greedy=not args.sample)
        print(f"{opp:>10}: {wr:.3f} over {args.games} games")


if __name__ == "__main__":
    main()
