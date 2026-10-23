"""Export a PyTorch checkpoint for the Rust side (`sr_bots::Net`).

    python -m starrealms.export runs/<run>/best.pt models/<name>.srnn          # MLP Policy
    python -m starrealms.export runs/<run>/best.pt models/<name>.safetensors   # EntityPolicy

`.srnn` layout: ``b"SRNN0001"``, u32 LE header length, JSON header, then little-endian
f32 tensors at the byte offsets (relative to the data start) in the header.

Entity models are written as safetensors (f32), with the model config and the tokenizer
constants the Rust `EntityNet` checks in the ``__metadata__`` map.

The server, the browser build, and ``model:<path>`` opponents in `VecEnv` all load these
files without PyTorch.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import N_ACTIONS, N_CARDS, OBS_DIM

MAGIC = b"SRNN0001"


# The privileged critic is training-only. The Rust net loads tensors by name and never runs it.
_PRIV_EXPORT = ("critic_tok", "priv_type_emb.", "value_priv1.", "value_priv2.")


def _arrays(policy) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype="<f4")
            for name, t in policy.state_dict().items()
            if not name.startswith(_PRIV_EXPORT)}


def export_srnn(policy) -> bytes:
    tensors, blobs, offset = [], [], 0
    for name, a in _arrays(policy).items():
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        blobs.append(a.tobytes())
        offset += a.nbytes
    cfg = policy.cfg
    header = json.dumps({
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "n_cards": N_CARDS,
        "embed": cfg.embed,
        "hidden": cfg.hidden,
        "layers": cfg.layers,
        "tensors": tensors,
    }).encode()
    return MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)


def export_safetensors(policy) -> bytes:
    from . import tokens as tk

    meta = {
        "arch": policy.ARCH,
        "config": json.dumps(asdict(policy.cfg)),
        "obs_dim": str(OBS_DIM),
        "n_actions": str(N_ACTIONS),
        "n_cards": str(N_CARDS),
        "hand": str(tk.HAND),
        "discard": str(tk.DISCARD),
        "verbs": ",".join(tk.VERBS),
    }
    header: dict = {"__metadata__": meta}
    blobs, offset = [], 0
    for name, a in _arrays(policy).items():
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        blobs.append(a.tobytes())
        offset += a.nbytes
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(blobs)


def export(policy, path: str | Path) -> int:
    """Write `policy` to `path` (`.srnn` for `Policy`, `.safetensors` for `EntityPolicy`).
    Returns the file size."""
    path = Path(path)
    want = ".safetensors" if policy.ARCH == "entity" else ".srnn"
    if path.suffix != want:
        raise ValueError(f"{policy.ARCH} models export to {want} files, not {path.name}")
    data = export_safetensors(policy) if policy.ARCH == "entity" else export_srnn(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def main() -> None:
    from .model import load

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a .pt file saved by the trainer")
    p.add_argument("out", help="destination .srnn (MLP) or .safetensors (entity) file")
    args = p.parse_args()
    size = export(load(args.checkpoint), args.out)
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
