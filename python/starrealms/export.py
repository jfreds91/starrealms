"""Export a PyTorch checkpoint to the `.srnn` format read by `sr_bots::Net`.

    python -m starrealms.export runs/<run>/best.pt models/<name>.srnn

Layout: ``b"SRNN0001"``, u32 LE header length, JSON header, then little-endian
f32 tensors at the byte offsets (relative to the data start) in the header.
The server, the browser build, and ``model:<path>`` opponents in `VecEnv` all
load this file without PyTorch.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from . import N_ACTIONS, N_CARDS, OBS_DIM

MAGIC = b"SRNN0001"


def export(policy, path: str | Path) -> int:
    """Write `policy` (a `starrealms.model.Policy`) to `path`. Returns the file size."""
    tensors, blobs, offset = [], [], 0
    for name, t in policy.state_dict().items():
        a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4")
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
    data = MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def main() -> None:
    from .model import Policy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a .pt file saved by the trainer")
    p.add_argument("out", help="destination .srnn file")
    args = p.parse_args()
    size = export(Policy.load(args.checkpoint), args.out)
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
