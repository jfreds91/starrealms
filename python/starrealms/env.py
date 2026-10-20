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
from pathlib import Path

import numpy as np

from ._engine import SCALAR_NAMES, VecEnv, card_catalog, obs_layout

LAYOUT: dict[str, slice] = {name: slice(off, off + n) for name, off, n in obs_layout()}
SCALAR: dict[str, int] = {name: LAYOUT["scalars"].start + i for i, name in enumerate(SCALAR_NAMES)}
"""Observation column of each named scalar feature."""


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


def parse_sets_mix(s: str) -> list[tuple[list[str], float]]:
    """``"core:0.3,core+colony_wars+frontiers:0.7"`` -> [(["core"], 0.3), ([...], 0.7)]."""
    out = []
    for part in s.split(","):
        sets, frac = part.rsplit(":", 1)
        out.append(([x.strip() for x in sets.split("+")], float(frac)))
    total = sum(f for _, f in out)
    return [(sets, f / total) for sets, f in out]


def sets_tag(sets) -> str:
    return "+".join(sets)


class MultiEnv:
    """Concatenation of several `VecEnv` groups that share one action batch.

    `groups` maps an opponent name to an env count, e.g.
    ``{"self": 192, "heuristic": 64}``. "self" means both seats are decided
    from Python; anything else names a Rust bot. With `sets_mix` (see
    `parse_sets_mix`) every group is split across those card-set selections in
    proportion, instead of all playing `sets`; `labels` then name each part
    ``<opponent>@<sets>``.
    """

    def __init__(self, groups: dict[str, int], sets=("core",), seed: int = 0, max_turns: int = 200,
                 sets_mix: list[tuple[list[str], float]] | None = None):
        self.names: list[str] = []
        self.labels: list[str] = []
        self.envs: list[VecEnv] = []
        mix = sets_mix or [(list(sets), 1.0)]
        k = 0
        for name, n in groups.items():
            counts = [round(n * f) for _, f in mix]
            counts[-1] = n - sum(counts[:-1])
            for (s, _), m in zip(mix, counts):
                if m <= 0:
                    continue
                opp = None if name == "self" else name
                self.envs.append(VecEnv(m, list(s), seed * 7919 + k, opp, max_turns))
                self.names.append(name)
                label = Path(name.split(":", 1)[1]).stem if name.startswith("model:") else name
                self.labels.append(f"{label}@{sets_tag(s)}" if sets_mix else label)
                k += 1
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
