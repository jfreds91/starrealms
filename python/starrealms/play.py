"""Play in the terminal against a bot or a trained checkpoint.

    python -m starrealms.play --opponent heuristic
    python -m starrealms.play --opponent runs/<run>/best.pt --sets core,colony_wars
"""

from __future__ import annotations

import argparse
import random

from ._engine import Game


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="heuristic", help="bot name or checkpoint path")
    p.add_argument("--sets", default="core")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seat", type=int, default=None, help="0 goes first")
    args = p.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    seat = args.seat if args.seat is not None else random.randrange(2)
    g = Game(args.sets.split(","), seed)

    policy = None
    if args.opponent.endswith(".pt"):
        import torch

        from .model import Policy

        policy = Policy.load(args.opponent).eval()

        def opp_act() -> int:
            o = torch.from_numpy(g.observe())[None]
            m = torch.from_numpy(g.mask())[None]
            return int(policy.act(o, m, greedy=True)[0][0])
    else:

        def opp_act() -> int:
            return g.bot_action(args.opponent, random.randrange(2**32))

    print(f"seed {seed}; you are seat {seat} ({'first' if seat == 0 else 'second'})")
    while g.winner is None:
        legal = g.legal()
        if g.active != seat:
            a = opp_act()
            print(f"  opponent: {g.describe(a)}")
            g.step(a)
            continue
        if len(legal) == 1:
            g.step(legal[0])
            continue
        print()
        print(g.render(seat))
        while True:
            raw = input("action> ").strip()
            if raw in ("q", "quit"):
                print(f"replay: {g.replay()}")
                return
            if raw.isdigit() and int(raw) in legal:
                g.step(int(raw))
                break
            print("enter one of the numbers in brackets (q to quit)")
    w = g.winner
    print(g.render(seat))
    print("draw" if w == -1 else ("you win!" if w == seat else "you lose"))
    print(f"replay: {g.replay()}")


if __name__ == "__main__":
    main()
