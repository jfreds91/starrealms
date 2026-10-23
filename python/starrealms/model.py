"""Policy/value networks.

`Policy` (MLP; Rust: `sr_bots::Net`, `.srnn` files):

* `ids` segment -> card embedding lookup (index 0 = empty), flattened.
* `counts` segment ([9 zones x N_CARDS]) -> per-zone bag of card embeddings
  (`counts @ E[1:]`), plus the raw counts scaled by 1/4.
* everything else is fed as is.

Then a ReLU MLP trunk with separate policy (N_ACTIONS logits) and value heads.

`EntityPolicy` (transformer; Rust: `sr_bots::EntityNet`, `.safetensors` files): the
observation is split into entity tokens (`tokens.py`), a pre-LayerNorm transformer encoder
mixes them, and each action's logit is a dot product between a per-verb query computed from
the global token and the key of the token the action targets (a card in hand, a trade-row
slot, an opponent base, ...). Card ids absent from the hand/discard tokens (overflow) fall
back to a context-free key of the card itself. End turn, skip, choices and attacking the
opponent are scored from the global token directly. An optional privileged critic reads three
extra zone tokens (opponent hand, own upcoming draws, opponent upcoming draws) plus a learned
critic token that public tokens are not allowed to attend to. `value_priv` is that critic;
`value` stays the public head the exported model serves. Rust never sees the privileged weights.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import COUNT_ZONES, N_ACTIONS, N_CARDS, N_IDS, OBS_DIM
from . import tokens as tk
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


@dataclass
class EntityConfig:
    d: int = 128
    layers: int = 3
    heads: int = 4
    ff: int = 256


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


class Base(nn.Module):
    """Shared sampling and checkpoint I/O. Subclasses define `ARCH`, `cfg`, `forward` and `value`."""

    ARCH = ""

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, greedy: bool = False, priv: torch.Tensor | None = None):
        out = self(obs, mask) if priv is None else self(obs, mask, priv)
        # With privileged input the last value is the critic; otherwise it is the only value head.
        logits, value = out[0].float(), out[-1].float()
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, value

    def save(self, path, **extra) -> None:
        torch.save({"arch": self.ARCH, "config": asdict(self.cfg), "state_dict": self.state_dict(),
                    "obs_dim": OBS_DIM, **extra}, path)

    @staticmethod
    def load(path, device="cpu") -> "Base":
        """Load a checkpoint of either architecture."""
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck.get("obs_dim", OBS_DIM) != OBS_DIM:
            raise ValueError(f"checkpoint obs_dim {ck['obs_dim']} != engine OBS_DIM {OBS_DIM}")
        if ck.get("arch", Policy.ARCH) == EntityPolicy.ARCH:
            m: Base = EntityPolicy(EntityConfig(**ck["config"]))
        else:
            m = Policy(ModelConfig(**ck["config"]))
        missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
        if unexpected or not _missing_is_privileged(list(missing)):
            raise RuntimeError(f"checkpoint {path} missing {missing} unexpected {unexpected}")
        return m.to(device)


# Added with the privileged critic. Checkpoints from before it omit them; a fresh init is correct.
_PRIV_PARAMS = ("priv_type_emb.", "critic_tok", "value_priv1.", "value_priv2.")


def _missing_is_privileged(keys: list[str]) -> bool:
    return all(k.startswith(_PRIV_PARAMS) for k in keys)


load = Base.load


class Policy(Base):
    ARCH = "mlp"

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


class Block(nn.Module):
    """Pre-LayerNorm transformer encoder layer."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.ff2.weight)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """`key_mask` is True where attention is allowed: `[B, 1, 1, N]` or `[B, 1, N, N]`."""
        b, n, d = x.shape
        h, dh = self.heads, d // self.heads
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, h, dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(y)
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class EntityPolicy(Base):
    ARCH = "entity"

    def __init__(self, cfg: EntityConfig | None = None):
        super().__init__()
        from ._engine import card_features

        self.cfg = cfg or EntityConfig()
        self.compact = True
        """Trim padding per batch (see `tokens.tokenize`); off under vmap, which needs static shapes."""
        self._types: dict = {}
        d = self.cfg.d
        feats = torch.as_tensor(card_features())
        self.register_buffer("card_feats", torch.cat([torch.zeros(1, feats.shape[1]), feats]))
        self.card_id = nn.Embedding(N_CARDS + 1, d)
        nn.init.normal_(self.card_id.weight, std=0.02)
        self.card_feat = nn.Linear(feats.shape[1], d, bias=False)
        self.type_emb = nn.Embedding(tk.N_TOKEN_TYPES, d)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.global_in = nn.Linear(tk.N_DENSE_GLOBAL, d)
        self.zone_in = nn.Linear(d + 1, d)
        self.copy_in = nn.Linear(d, d, bias=False)
        self.own_in = nn.Linear(tk.OWN_FEATS, d)
        self.opp_in = nn.Linear(tk.OPP_FEATS, d)
        self.row_in = nn.Linear(tk.ROW_FEATS, d)
        self.explorer_in = nn.Linear(1, d)
        self.hand_in = nn.Linear(1, d)
        self.discard_in = nn.Linear(1, d)
        self.blocks = nn.ModuleList(Block(d, self.cfg.heads, self.cfg.ff) for _ in range(self.cfg.layers))
        self.ln_f = nn.LayerNorm(d)
        self.value1 = nn.Linear(d, d)
        self.value2 = layer_init(nn.Linear(d, 1), std=1.0)
        self.global_out = layer_init(nn.Linear(d, tk.N_GLOBAL_LOGITS), std=0.01)
        self.query = layer_init(nn.Linear(d, len(tk.VERBS) * d), std=0.01)
        self.key = nn.Linear(d, d)
        self.priv_type_emb = nn.Embedding(len(tk.PRIV_ZONES), d)
        nn.init.normal_(self.priv_type_emb.weight, std=0.02)
        self.critic_tok = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.critic_tok, std=0.02)
        self.value_priv1 = nn.Linear(d, d)
        self.value_priv2 = layer_init(nn.Linear(d, 1), std=1.0)

    def card_table(self) -> torch.Tensor:
        """[N_CARDS + 1, d] card vectors; row 0 (empty) is zero."""
        t = self.card_id.weight + self.card_feat(self.card_feats)
        return torch.cat([torch.zeros_like(t[:1]), t[1:]])

    def encode(self, obs: torch.Tensor, priv: torch.Tensor | None = None) -> tuple[torch.Tensor, tk.Tokens, torch.Tensor]:
        """Final token states [B, N, d], the public tokens, and the card table.

        `priv` `[B, PRIV_IDS]` appends the privileged zones and the critic token. Public tokens
        cannot attend to those, so their states (and therefore the logits and public value)
        do not depend on `priv`.
        """
        t = tk.tokenize(obs, self.compact)
        e = self.card_table()
        b = obs.shape[0]
        zone_bag = torch.matmul(t.zones, e[1:])
        zone_total = t.zones.sum(-1, keepdim=True)
        groups = [t.source[:, None], t.own_ids, t.own_copy, t.opp_ids, t.row_ids, t.hand_ids, t.discard_ids]
        # one lookup for every card id: indexing backward is very slow on ROCm, embedding backward is not
        src, own, copy, opp, row, hand, disc = F.embedding(torch.cat(groups, 1), e).split(
            [g.shape[1] for g in groups], 1)
        x = torch.cat([
            self.global_in(t.dense)[:, None] + src,
            self.zone_in(torch.cat([zone_bag, zone_total], -1)),
            own + self.copy_in(copy) + self.own_in(t.own_feats),
            opp + self.opp_in(t.opp_feats),
            row + self.row_in(t.row_feats),
            (e[tk.EXPLORER_ID] + self.explorer_in(t.explorers[:, None]))[:, None],
            hand + self.hand_in(t.hand_counts[..., None]),
            disc + self.discard_in(t.discard_counts[..., None]),
        ], 1)
        key = (tuple(t.sizes.values()), x.device)
        if key not in self._types:
            self._types[key] = tk.token_types(t.sizes).to(x.device)
        x = x + self.type_emb(self._types[key])
        if priv is None:
            key_mask = t.valid.view(b, 1, 1, -1)
        else:
            counts = tk.priv_zones(priv)
            bag = torch.matmul(counts, e[1:])
            total = counts.sum(-1, keepdim=True)
            zones = self.zone_in(torch.cat([bag, total], -1)) + self.priv_type_emb.weight
            critic = self.critic_tok.view(1, 1, -1).expand(b, -1, -1)
            n_pub = x.shape[1]
            x = torch.cat([x, zones, critic], 1)
            n = x.shape[1]
            valid = torch.cat([t.valid, torch.ones(b, n - n_pub, dtype=torch.bool, device=x.device)], 1)
            is_priv = torch.zeros(n, dtype=torch.bool, device=x.device)
            is_priv[n_pub:] = True
            # query q may see key k when k is valid and (q is privileged or k is public)
            allow = valid[:, None, :] & (is_priv[None, :, None] | ~is_priv[None, None, :])
            key_mask = allow[:, None]
        for blk in self.blocks:
            x = blk(x, key_mask)
        return self.ln_f(x), t, e

    def _value(self, h: torch.Tensor) -> torch.Tensor:
        return self.value2(F.gelu(self.value1(h[:, 0]))).squeeze(-1)

    def _value_priv(self, h: torch.Tensor) -> torch.Tensor:
        return self.value_priv2(F.gelu(self.value_priv1(h[:, -1]))).squeeze(-1)

    def logits(self, h: torch.Tensor, t: tk.Tokens, e: torch.Tensor) -> torch.Tensor:
        """Unmasked [B, N_ACTIONS] logits."""
        b, n, d = h.shape
        g = h[:, 0]
        q = self.query(g).view(b, len(tk.VERBS), d)
        scale = 1.0 / math.sqrt(d)
        scores = torch.einsum("bvd,bnd->bvn", q, self.key(h)) * scale
        card_scores = torch.einsum("bvd,cd->bvc", q, self.key(self.ln_f(e[1:]))) * scale
        glob = self.global_out(g)
        slots = {"hand": t.hand_slot, "discard": t.discard_slot}
        parts, gi = [], 0
        for name, (_, count) in sorted(tk.ACTIONS.items(), key=lambda kv: kv[1][0]):
            if name in tk.GLOBAL_ACTIONS:
                parts.append(glob[:, gi : gi + count])
                gi += count
                continue
            group = tk.POINTER_ACTIONS[name]
            v = tk.VERBS.index(name)
            s = scores[:, v, t.group(group)]
            if group in tk.CARD_ADDRESSED:
                slot = slots[group]
                s = torch.where(slot >= 0, s.gather(1, slot.clamp(min=0)), card_scores[:, v])
            elif s.shape[1] < count:  # trimmed padding: those slots are empty, so never legal
                s = F.pad(s, (0, count - s.shape[1]))
            parts.append(s)
        return torch.cat(parts, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, priv: torch.Tensor | None = None):
        """`(masked logits, public value)` or, with `priv`, `(logits, public value, privileged value)`."""
        h, t, e = self.encode(obs, priv)
        logits = self.logits(h, t, e).masked_fill(~mask, -1e9)
        v_pub = self._value(h)
        if priv is None:
            return logits, v_pub
        return logits, v_pub, self._value_priv(h)

    def value(self, obs: torch.Tensor, priv: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encode(obs, priv)[0]
        return self._value(h) if priv is None else self._value_priv(h)
