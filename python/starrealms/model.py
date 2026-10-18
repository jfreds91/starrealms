"""Policy/value network.

Input handling (kept simple so the forward pass can be re-implemented in Rust):

* `ids` segment -> card embedding lookup (index 0 = empty), flattened.
* `counts` segment ([9 zones x N_CARDS]) -> per-zone bag of card embeddings
  (`counts @ E[1:]`), plus the raw counts scaled by 1/4.
* everything else is fed as is.

Then a ReLU MLP trunk with separate policy (N_ACTIONS logits) and value heads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from . import COUNT_ZONES, N_ACTIONS, N_CARDS, N_IDS, OBS_DIM
from .env import LAYOUT

COUNTS = LAYOUT["counts"]
IDS = LAYOUT["ids"]
N_ZONES = len(COUNT_ZONES)
DENSE_IDX = np.r_[0 : COUNTS.start, COUNTS.stop : IDS.start]  # all non-count, non-id features
COUNT_SCALE = 0.25


@dataclass
class ModelConfig:
    embed: int = 16
    hidden: int = 512
    layers: int = 2


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Policy(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d, h = self.cfg.embed, self.cfg.hidden
        self.embed = nn.Embedding(N_CARDS + 1, d, padding_idx=0)
        nn.init.normal_(self.embed.weight, std=0.1)
        with torch.no_grad():
            self.embed.weight[0].zero_()
        self.register_buffer("dense_idx", torch.as_tensor(DENSE_IDX, dtype=torch.long), persistent=False)
        in_dim = len(DENSE_IDX) + N_ZONES * N_CARDS + N_IDS * d + N_ZONES * d
        trunk: list[nn.Module] = []
        for i in range(self.cfg.layers):
            trunk += [layer_init(nn.Linear(in_dim if i == 0 else h, h)), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)
        self.pi = layer_init(nn.Linear(h, N_ACTIONS), std=0.01)
        self.v = layer_init(nn.Linear(h, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        dense = obs.index_select(1, self.dense_idx)
        counts = obs[:, COUNTS].view(b, N_ZONES, N_CARDS)
        ids = obs[:, IDS].long()
        e = self.embed.weight
        zone = torch.matmul(counts, e[1:]).view(b, -1) * COUNT_SCALE
        idv = self.embed(ids).view(b, -1)
        return torch.cat([dense, counts.view(b, -1) * COUNT_SCALE, idv, zone], dim=1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked logits, value)."""
        h = self.trunk(self.features(obs))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.trunk(self.features(obs))).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, greedy: bool = False):
        logits, value = self(obs, mask)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, value

    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), "obs_dim": OBS_DIM, **extra}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "Policy":
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("obs_dim", OBS_DIM) != OBS_DIM:
            raise ValueError(f"checkpoint obs_dim {ck['obs_dim']} != engine OBS_DIM {OBS_DIM}")
        m = cls(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        return m.to(device)
