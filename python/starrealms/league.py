"""Opponent league for self-play, and Elo ratings for checkpoints and bots."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


def snapshot_name(update: int) -> str:
    return f"u{update:05d}"


class League:
    """Frozen policy snapshots keyed by the update they were taken at.

    The first snapshot is an anchor and is never evicted; beyond `size` members the
    oldest of the others goes. Opponents are drawn by prioritized fictitious self-play:
    weight `(1 - p) ** pfsp_power` where `p` is a moving average of the learner's score
    against that member, so opponents it still loses to come up more often.
    """

    def __init__(self, size: int, pfsp_power: float = 2.0, ema: float = 0.02):
        self.size = size
        self.pfsp_power = pfsp_power
        self.ema = ema
        self.members: dict[int, nn.Module] = {}
        self.score: dict[int, float] = {}
        self._stack: Stack | None = None

    def add(self, policy: nn.Module, tag: int) -> int | None:
        """Snapshot `policy`; returns the evicted tag, if any."""
        m = copy.deepcopy(policy).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.members[tag] = m
        self.score[tag] = 0.5
        self._stack = None
        if len(self.members) > self.size:
            evict = sorted(self.members)[1]
            del self.members[evict], self.score[evict]
            return evict
        return None

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, tags: np.ndarray) -> torch.Tensor:
        """Sampled actions for rows whose opponent is `tags[i]`, all members in one batched pass."""
        if self._stack is None:
            self._stack = Stack(self.members)
        return self._stack.act(obs, mask, tags)

    def weights(self) -> tuple[list[int], np.ndarray]:
        tags = list(self.members)
        w = np.array([(1.0 - self.score[t]) ** self.pfsp_power + 1e-3 for t in tags])
        return tags, w / w.sum()

    def sample(self, rng: np.random.Generator, p_self: float) -> int:
        """A member tag, or -1 for the current policy."""
        if not self.members or rng.random() < p_self:
            return -1
        tags, w = self.weights()
        return int(tags[rng.choice(len(tags), p=w)])

    def eval_members(self) -> list[int]:
        """Snapshots a new checkpoint is rated against: the anchor, the newest, and the hardest."""
        if not self.members:
            return []
        picks = [min(self.members), max(self.members), min(self.score, key=self.score.get)]
        return list(dict.fromkeys(picks))

    def record(self, tag: int, score: float) -> None:
        """Learner's result (1 win, 0.5 draw, 0 loss) in a finished game against `tag`."""
        if tag in self.score:
            self.score[tag] += self.ema * (score - self.score[tag])


class Stack:
    """Parameters of same-architecture policies stacked along a new leading dim, so all of
    them run as one vmapped forward pass instead of one small pass each."""

    def __init__(self, members: dict[int, nn.Module]):
        self.tags = np.array(list(members))
        mods = list(members.values())
        self.params, _ = torch.func.stack_module_state(mods)
        base = copy.deepcopy(mods[0])
        if hasattr(base, "compact"):
            base.compact = False  # data-dependent shapes are not allowed under vmap

        def forward(params, obs, mask):
            return torch.func.functional_call(base, params, (obs, mask))[0]

        self.forward = torch.vmap(forward)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, tags: np.ndarray) -> torch.Tensor:
        k = np.searchsorted(self.tags, tags)  # member index per row (tags are sorted ints)
        order = np.argsort(k, kind="stable")
        counts = np.bincount(k, minlength=len(self.tags))
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.empty_like(k)
        pos[order] = np.arange(len(k)) - starts[k[order]]
        width = int(counts.max())
        dev = obs.device
        kt, pt = torch.from_numpy(k).to(dev), torch.from_numpy(pos).to(dev)
        o = obs.new_zeros((len(self.tags), width, obs.shape[1]))
        m = mask.new_zeros((len(self.tags), width, mask.shape[1]))
        o[kt, pt] = obs
        m[kt, pt] = mask
        logits = self.forward(self.params, o, m)[kt, pt].float()
        return torch.distributions.Categorical(logits=logits).sample()


class FixedPool:
    """Frozen `.pt` checkpoints played from Python, one `Stack` per architecture.

    A `model:<path>.pt` opponent cannot run inside the Rust env. Members are never evicted,
    so a run can train against fixed transformers such as `phase_f` without the candle thread
    pool fighting the environment's workers.
    """

    def __init__(self, paths: list[str], device):
        import json
        from dataclasses import asdict

        from .model import load

        self.paths = list(paths)
        models = []
        for path in self.paths:
            model = load(path, device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            if hasattr(model, "compact"):
                model.compact = False
            models.append(model)
        groups: dict[str, dict[int, nn.Module]] = {}
        for i, model in enumerate(models):
            key = model.ARCH + json.dumps(asdict(model.cfg), sort_keys=True)
            groups.setdefault(key, {})[i] = model
        self.stacks = [Stack(members) for members in groups.values()]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, ids: np.ndarray) -> torch.Tensor:
        """Sampled actions; `ids[row]` indexes `paths`."""
        out = torch.empty(obs.shape[0], dtype=torch.long, device=obs.device)
        for stack in self.stacks:
            sel = np.isin(ids, stack.tags)
            if not np.any(sel):
                continue
            chosen = torch.as_tensor(sel, device=obs.device)
            out[chosen] = stack.act(obs[chosen], mask[chosen], ids[sel])
        return out


class Ratings:
    """Elo ratings by name (bots, and checkpoints named `u<update>`), fitted jointly.

    Every recorded result is kept, and after each batch all ratings are refitted as the
    Bradley-Terry maximum-likelihood estimate (as BayesElo does), so no rating depends on
    the order games were played in. Each player also gets `prior` virtual games, half won,
    against a virtual opponent, which keeps ratings finite for players that never lose
    or never win. Elo is reported relative to `anchor`.
    """

    def __init__(self, anchor: str = "heuristic", prior: float = 1.0):
        self.anchor = anchor
        self.prior = prior
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.score: dict[tuple[int, int], float] = defaultdict(float)  # (i, j): i's points vs j
        self.gamma = np.ones(0)
        self.elo: dict[str, float] = {}

    def ensure(self, name: str) -> None:
        if name not in self.index:
            self.index[name] = len(self.names)
            self.names.append(name)
            self.gamma = np.append(self.gamma, 1.0)
            self._fit()

    def games(self, a: str, b: str, wins: int, draws: int, losses: int) -> None:
        """Record a's results against b and refit."""
        self.ensure(a)
        self.ensure(b)
        i, j = self.index[a], self.index[b]
        self.score[i, j] += wins + 0.5 * draws
        self.score[j, i] += losses + 0.5 * draws
        self._fit()

    def _fit(self, tol: float = 1e-10, max_iter: int = 20_000) -> None:
        n = len(self.names)
        s = np.zeros((n, n))
        for (i, j), v in self.score.items():
            s[i, j] = v
        played = s + s.T
        points = s.sum(1) + self.prior / 2
        g = self.gamma.copy()
        for _ in range(max_iter):  # minorization-maximization (Hunter 2004)
            denom = (played / (g[:, None] + g[None, :])).sum(1) + self.prior / (g + 1.0)
            new = points / denom
            done = np.abs(np.log(new) - np.log(g)).max() < tol
            g = new
            if done:
                break
        self.gamma = g
        elo = 400 * np.log10(g)
        if self.anchor in self.index:
            elo -= elo[self.index[self.anchor]]
        self.elo = dict(zip(self.names, elo.tolist()))

    def calibrate_bots(self, bots: list[str], sets: list[str], games: int = 300) -> None:
        """Rate the scripted bots from round-robin games between them (played in Rust)."""
        from ._engine import arena

        for i, a in enumerate(bots):
            for b in bots[i + 1 :]:
                wa, wb, d = arena(a, b, games, sets, 7)
                self.games(a, b, wa, d, wb)

    def rating(self, name: str) -> float:
        return self.elo.get(name, float("-inf"))

    def table(self) -> list[dict]:
        rows = []
        for name, i in self.index.items():
            played = sum(v for (a, b), v in self.score.items() if i in (a, b))
            rows.append({"name": name, "elo": round(self.elo[name], 1), "games": played})
        return sorted(rows, key=lambda x: -x["elo"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.table(), indent=1))
