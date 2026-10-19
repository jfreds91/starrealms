"""Evaluate a checkpoint against the scripted bots.

    python -m starrealms.evaluate runs/<run>/best.pt --opponents heuristic,greedy,random --games 2000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from ._engine import VecEnv


@dataclass
class EvalResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def winrate(self) -> float:
        """Draws count 1/2."""
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for `winrate` (95% by default)."""
        n, p = self.games, self.winrate
        if n == 0:
            return 0.0, 1.0
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

    def __str__(self) -> str:
        lo, hi = self.interval()
        return f"{self.winrate:.3f} [{lo:.3f}, {hi:.3f}] over {self.games} games ({self.wins}W {self.draws}D {self.losses}L)"


@torch.no_grad()
def evaluate(policy, opponent, games: int, sets, device, greedy: bool = True, seed: int = 0) -> EvalResult:
    """Play `games` games of `policy`, alternating seats, against `opponent`: a Rust bot name,
    `model:<path.srnn>`, or another `Policy` (which also plays greedily if `greedy`)."""
    n = min(games, 256)
    bot = isinstance(opponent, str)
    env = VecEnv(n, list(sets), 10_000_019 + seed, opponent if bot else None)
    obs, mask, seat, learner = env.reset()
    res = EvalResult(0, 0, 0)
    started = n
    active = np.ones(n, bool)  # count only the first `games` games
    while res.games < games:
        o = torch.from_numpy(obs).to(device)
        m = torch.from_numpy(mask).to(device)
        a, _, _ = policy.act(o, m, greedy=greedy)
        if not bot:
            theirs = torch.from_numpy(np.flatnonzero(seat != learner)).to(device)
            if len(theirs):
                a[theirs] = opponent.act(o[theirs], m[theirs], greedy=greedy)[0]
        obs, mask, seat, learner, _, done, winner, _, done_learner = env.step(a.cpu().numpy())
        for i in np.flatnonzero(done & active):
            if winner[i] == done_learner[i]:
                res.wins += 1
            elif winner[i] == -1:
                res.draws += 1
            else:
                res.losses += 1
            if started < games:
                started += 1
            else:
                active[i] = False
    return res


def main(argv=None) -> None:
    from .model import load

    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--opponents", default="heuristic,greedy,random", help="bot names or model:<path.srnn>")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--sets", default="core")
    p.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    policy = load(args.checkpoint, args.device).eval()
    sets = args.sets.split(",")
    for opp in args.opponents.split(","):
        r = evaluate(policy, opp, args.games, sets, args.device, greedy=not args.sample)
        print(f"{opp:>10}: {r}")


if __name__ == "__main__":
    main()
