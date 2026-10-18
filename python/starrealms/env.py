"""Environment helpers on top of the Rust `VecEnv`.

Conventions
-----------
* Observations are always from the perspective of the seat to move (`seat`).
* `learner_seat` says which seat the training policy controls in each env;
  in self-play groups the other seat is driven by an opponent policy chosen
  by the trainer, in bot groups it is played inside Rust and never surfaces.
* When a game ends, `done[i]` is set, `rewards[i, s]` holds the terminal
  reward for seat `s` (+1 win, -1 loss, 0 draw) and the env has already been
  reset: `obs[i]` belongs to the next game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ._engine import VecEnv, card_catalog, obs_layout

LAYOUT: dict[str, slice] = {name: slice(off, off + n) for name, off, n in obs_layout()}


@lru_cache(maxsize=1)
def cards() -> list[dict]:
    """Card catalog in engine index order (index = id used in observations - 1)."""
    return json.loads(card_catalog())


@dataclass
class StepResult:
    obs: np.ndarray  # [n, OBS_DIM] f32
    mask: np.ndarray  # [n, N_ACTIONS] bool
    seat: np.ndarray  # [n] i8, seat to move
    learner_seat: np.ndarray  # [n] i8
    rewards: np.ndarray  # [n, 2] f32
    done: np.ndarray  # [n] bool
    winner: np.ndarray  # [n] i8: seat, -1 draw, -2 not done
    turns: np.ndarray  # [n] u16
    done_learner: np.ndarray  # [n] i8: learner seat of the game that just ended

    @property
    def learner_to_move(self) -> np.ndarray:
        return self.seat == self.learner_seat


class MultiEnv:
    """Concatenation of several `VecEnv` groups that share one action batch.

    `groups` maps an opponent name to an env count, e.g.
    ``{"self": 192, "heuristic": 64}``. "self" means both seats are decided
    from Python; anything else names a Rust bot.
    """

    def __init__(self, groups: dict[str, int], sets=("core",), seed: int = 0, max_turns: int = 200):
        self.names: list[str] = []
        self.envs: list[VecEnv] = []
        for k, (name, n) in enumerate(groups.items()):
            if n <= 0:
                continue
            opp = None if name == "self" else name
            self.envs.append(VecEnv(n, list(sets), seed * 7919 + k, opp, max_turns))
            self.names.append(name)
        self.sizes = [e.num_envs for e in self.envs]
        self.num_envs = sum(self.sizes)
        self.group = np.concatenate([np.full(n, i, np.int8) for i, n in enumerate(self.sizes)])
        self.self_play = np.isin(self.group, [i for i, n in enumerate(self.names) if n == "self"])

    def reset(self) -> StepResult:
        parts = [e.reset() for e in self.envs]
        obs, mask, seat, learner = (np.concatenate(x) for x in zip(*parts))
        n = self.num_envs
        return StepResult(
            obs, mask, seat, learner,
            np.zeros((n, 2), np.float32), np.zeros(n, bool), np.full(n, -2, np.int8),
            np.zeros(n, np.uint16), learner.copy(),
        )

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64)
        parts, start = [], 0
        for e, n in zip(self.envs, self.sizes):
            parts.append(e.step(actions[start : start + n]))
            start += n
        return StepResult(*(np.concatenate(x) for x in zip(*parts)))

    def features(self) -> dict[str, np.ndarray]:
        fs = [e.features() for e in self.envs]
        return {k: np.concatenate([f[k] for f in fs]) for k in fs[0]}

    def locate(self, i: int) -> tuple[VecEnv, int]:
        for e, n in zip(self.envs, self.sizes):
            if i < n:
                return e, i
            i -= n
        raise IndexError(i)

    def render(self, i: int) -> str:
        e, j = self.locate(i)
        return e.render(j)
