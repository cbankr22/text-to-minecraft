"""Convert kept 64^3 objects to 32^3 (data/objaverse/vox32/<uid>.npz). Skips existing.

    .venv/Scripts/python scripts/make_vox32.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.dataset import load_manifest  # noqa: E402
from ttmc.downsample import down2  # noqa: E402

DATA = ROOT / "data" / "objaverse"


def main() -> None:
    out = DATA / "vox32"
    out.mkdir(exist_ok=True)
    rows = load_manifest(DATA / "manifest.jsonl")
    done = 0
    for r in rows:
        dst = out / f"{r['uid']}.npz"
        if dst.exists():
            continue
        b = np.load(DATA / "vox64" / f"{r['uid']}.npz")["blocks"]
        np.savez_compressed(dst, blocks=down2(b))
        done += 1
    print(f"vox32: converted {done}, total {len(rows)}")


if __name__ == "__main__":
    main()
