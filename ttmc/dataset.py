"""Voxel datasets for training. Grids are (Y, Z, X) uint8 block tokens."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path: Path, keep_only: bool = True, licenses: set[str] | None = None) -> list[dict]:
    """Processed objects, one row per uid (the latest; a retried object appears twice)."""
    latest: dict[str, dict] = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l:
            r = json.loads(l)
            latest[r["uid"]] = r
    rows = [r for r in latest.values() if r.get("status") == "ok"]
    if licenses is not None:
        rows = [r for r in rows if r.get("license") in licenses]
    return [r for r in rows if r.get("keep")] if keep_only else rows


def is_val(uid: str, val_frac: float = 0.03) -> bool:
    """Stable per-object split: adding data never moves an object between train and val."""
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 10_000 < val_frac * 10_000


def split(rows: list[dict], val_frac: float = 0.03,
          val_uids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """val_uids pins an explicit validation set (e.g. a run's saved val_uids.json)."""
    if val_uids is not None:
        in_val = lambda r: r["uid"] in val_uids  # noqa: E731
    else:
        in_val = lambda r: is_val(r["uid"], val_frac)  # noqa: E731
    return [r for r in rows if not in_val(r)], [r for r in rows if in_val(r)]


def augment(blocks: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random 90-degree turn around Y plus optional mirror; gravity stays down."""
    blocks = np.rot90(blocks, rng.randrange(4), axes=(1, 2))
    if rng.random() < 0.5:
        blocks = blocks[:, :, ::-1]
    return np.ascontiguousarray(blocks)


class VoxelDataset(Dataset):
    def __init__(self, rows: list[dict], vox_dir: Path, augment: bool = False):
        self.rows, self.vox_dir, self.augment = rows, vox_dir, augment
        self.rng = random.Random()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        blocks = np.load(self.vox_dir / f"{r['uid']}.npz")["blocks"]
        if self.augment:
            blocks = augment(blocks, self.rng)
        return {"blocks": torch.from_numpy(blocks.astype(np.int64)), "caption": r["caption"],
                "uid": r["uid"]}
