"""Observation -> entity tokens for `EntityPolicy`, built from the flat observation with torch ops.

Token groups, in order (`GROUPS`, with their maximum sizes): one global token, one summary
token per count zone, the own in-play slots, the opponent's bases, the trade-row slots, the
Explorer pile, then the distinct card ids in hand and in the discard pile (ascending id, at
most `HAND` / `DISCARD`). Empty slots are padding and are masked out of attention.

In-play slots, opponent bases and the hand/discard ids always fill a prefix of their group,
so with `compact=True` each of those groups is cut to the longest prefix used in the batch.
Padding is masked, so this changes nothing but the cost.

`POINTER_ACTIONS` says, for each action family, which group its logits point at. The Rust
`EntityNet` mirrors this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import COUNT_ZONES, N_ACTIONS, N_CARDS, cards
from ._engine import action_layout
from .env import LAYOUT, SCALAR

ACTIONS: dict[str, tuple[int, int]] = {name: (start, n) for name, start, n in action_layout()}
N_OWN = ACTIONS["primary"][1]
N_OPP = ACTIONS["attack_base"][1]
N_ROW = ACTIONS["buy"][1]
N_ZONES = len(COUNT_ZONES)
HAND = 12
DISCARD = 32
EXPLORER_ID = next(c["idx"] for c in cards() if c["id"] == "explorer") + 1

GROUPS: dict[str, int] = {"global": 1, "zones": N_ZONES, "own": N_OWN, "opp": N_OPP, "row": N_ROW,
                          "explorer": 1, "hand": HAND, "discard": DISCARD}
"""Token groups in order, with their full sizes."""
N_TOKENS = sum(GROUPS.values())
N_TOKEN_TYPES = 1 + N_ZONES + len(GROUPS) - 2
"""Global, one per zone, one per remaining group."""

GLOBAL_ACTIONS = ("end", "skip", "choose", "attack_player")
"""Families scored from the global token."""
POINTER_ACTIONS: dict[str, str] = {
    "play": "hand",
    "primary": "own",
    "ally": "own",
    "ally2": "own",
    "scrap": "own",
    "buy": "row",
    "buy_explorer": "explorer",
    "attack_base": "opp",
    "pick_hand": "hand",
    "pick_discard": "discard",
    "pick_row": "row",
    "pick_explorer": "explorer",
    "pick_opp_base": "opp",
    "pick_own": "own",
}
"""Families scored by pointing at a token group; each family is its own verb."""
VERBS = list(POINTER_ACTIONS)
N_GLOBAL_LOGITS = sum(ACTIONS[f][1] for f in GLOBAL_ACTIONS)
CARD_ADDRESSED = ("hand", "discard")
"""Groups addressed by card id rather than by slot."""

assert set(GLOBAL_ACTIONS) | set(POINTER_ACTIONS) == set(ACTIONS)
assert sum(n for _, n in ACTIONS.values()) == N_ACTIONS

N_DENSE_GLOBAL = LAYOUT["scalars"].stop - LAYOUT["scalars"].start + LAYOUT["pending"].stop - LAYOUT["pending"].start
OWN_FEATS = (LAYOUT["own_slots"].stop - LAYOUT["own_slots"].start) // N_OWN
OPP_FEATS = (LAYOUT["opp_bases"].stop - LAYOUT["opp_bases"].start) // N_OPP
ROW_FEATS = (LAYOUT["row"].stop - LAYOUT["row"].start) // N_ROW
COUNT_SCALE = 0.25
_EXPLORERS = SCALAR["explorers"]
_ZONE = {z: i for i, z in enumerate(COUNT_ZONES)}


def token_types(sizes: dict[str, int]) -> torch.Tensor:
    """Type id of every token for the given group sizes."""
    types, nxt = [], 1 + N_ZONES
    for g, n in sizes.items():
        if g == "global":
            types.append(0)
        elif g == "zones":
            types += range(1, 1 + N_ZONES)
        else:
            types += [nxt] * n
            nxt += 1
    return torch.tensor(types)


@dataclass
class Tokens:
    dense: torch.Tensor  # [B, N_DENSE_GLOBAL] scalars + pending one-hot
    source: torch.Tensor  # [B] pending source card id (0 = none)
    zones: torch.Tensor  # [B, N_ZONES, N_CARDS] counts (scaled)
    own_ids: torch.Tensor  # [B, n_own]
    own_copy: torch.Tensor  # [B, n_own]
    own_feats: torch.Tensor  # [B, n_own, OWN_FEATS]
    opp_ids: torch.Tensor  # [B, n_opp]
    opp_feats: torch.Tensor  # [B, n_opp, OPP_FEATS]
    row_ids: torch.Tensor  # [B, N_ROW]
    row_feats: torch.Tensor  # [B, N_ROW, ROW_FEATS]
    explorers: torch.Tensor  # [B] explorer pile size (scaled)
    hand_ids: torch.Tensor  # [B, n_hand] distinct card ids in hand, ascending, 0 = padding
    hand_counts: torch.Tensor  # [B, n_hand] (scaled)
    discard_ids: torch.Tensor  # [B, n_discard]
    discard_counts: torch.Tensor  # [B, n_discard]
    valid: torch.Tensor  # [B, n_tokens] bool
    hand_slot: torch.Tensor  # [B, N_CARDS] position within the hand group per card id, -1 = none
    discard_slot: torch.Tensor  # [B, N_CARDS]
    sizes: dict[str, int]  # group sizes in this batch, in `GROUPS` order

    def group(self, name: str) -> slice:
        start = 0
        for g, n in self.sizes.items():
            if g == name:
                return slice(start, start + n)
            start += n
        raise KeyError(name)


def _distinct(counts: torch.Tensor, k: int):
    """First `k` card ids (ascending) with a nonzero count; their counts; per-card position."""
    b = counts.shape[0]
    idx = torch.arange(N_CARDS, device=counts.device).expand(b, -1)
    key = torch.where(counts > 0, idx, N_CARDS)
    first = key.sort(dim=1).values[:, :k]
    ok = first < N_CARDS
    ids = torch.where(ok, first + 1, 0)
    cnt = counts.gather(1, first.clamp(max=N_CARDS - 1)) * ok
    slot = torch.full((b, N_CARDS + 1), -1, dtype=torch.long, device=counts.device)
    pos = torch.arange(k, device=counts.device).expand(b, -1)
    slot = slot.scatter(1, torch.where(ok, first, N_CARDS), torch.where(ok, pos, -1))
    return ids, cnt, slot[:, :N_CARDS]


def tokenize(obs: torch.Tensor, compact: bool = False) -> Tokens:
    b = obs.shape[0]
    dense = torch.cat([obs[:, LAYOUT["scalars"]], obs[:, LAYOUT["pending"]]], 1)
    counts = obs[:, LAYOUT["counts"]].view(b, N_ZONES, N_CARDS)
    ids = obs[:, LAYOUT["ids"]].long()
    own_ids, own_copy = ids[:, :N_OWN], ids[:, N_OWN : 2 * N_OWN]
    opp_ids = ids[:, 2 * N_OWN : 2 * N_OWN + N_OPP]
    row_ids = ids[:, 2 * N_OWN + N_OPP : 2 * N_OWN + N_OPP + N_ROW]
    own_feats = obs[:, LAYOUT["own_slots"]].view(b, N_OWN, OWN_FEATS)
    opp_feats = obs[:, LAYOUT["opp_bases"]].view(b, N_OPP, OPP_FEATS)
    hand_ids, hand_counts, hand_slot = _distinct(counts[:, _ZONE["my_hand"]], HAND)
    discard_ids, discard_counts, discard_slot = _distinct(counts[:, _ZONE["my_discard"]], DISCARD)
    sizes = dict(GROUPS)
    if compact:
        used = torch.stack([(x > 0).sum(1).max() for x in (own_ids, opp_ids, hand_ids, discard_ids)]).tolist()
        n_own, n_opp, n_hand, n_disc = (max(1, n) for n in used)
        own_ids, own_copy, own_feats = own_ids[:, :n_own], own_copy[:, :n_own], own_feats[:, :n_own]
        opp_ids, opp_feats = opp_ids[:, :n_opp], opp_feats[:, :n_opp]
        hand_ids, hand_counts = hand_ids[:, :n_hand], hand_counts[:, :n_hand]
        discard_ids, discard_counts = discard_ids[:, :n_disc], discard_counts[:, :n_disc]
        sizes.update(own=n_own, opp=n_opp, hand=n_hand, discard=n_disc)
    one = torch.ones((b, 1), dtype=torch.bool, device=obs.device)
    valid = torch.cat([one.expand(b, 1 + N_ZONES), own_ids > 0, opp_ids > 0, row_ids > 0, one,
                       hand_ids > 0, discard_ids > 0], 1)
    return Tokens(
        dense=dense,
        source=ids[:, -1],
        zones=counts * COUNT_SCALE,
        own_ids=own_ids,
        own_copy=own_copy,
        own_feats=own_feats,
        opp_ids=opp_ids,
        opp_feats=opp_feats,
        row_ids=row_ids,
        row_feats=obs[:, LAYOUT["row"]].view(b, N_ROW, ROW_FEATS),
        explorers=obs[:, _EXPLORERS],
        hand_ids=hand_ids,
        hand_counts=hand_counts * COUNT_SCALE,
        discard_ids=discard_ids,
        discard_counts=discard_counts * COUNT_SCALE,
        valid=valid,
        hand_slot=hand_slot,
        discard_slot=discard_slot,
        sizes=sizes,
    )
