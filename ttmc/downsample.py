"""Downsample block grids by 2x: a cell is solid if any of its 8 voxels is, and takes
the most common non-air block among them (ties -> first in scan order)."""
from __future__ import annotations

import numpy as np


def down2(blocks: np.ndarray) -> np.ndarray:
    Y, Z, X = blocks.shape
    c = blocks.reshape(Y // 2, 2, Z // 2, 2, X // 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)
    solid = c > 0
    # count how often each of the 8 values appears among the solid ones
    counts = ((c[:, :, None] == c[:, None, :]) & solid[:, None, :]).sum(-1)
    counts[~solid] = -1
    mode = c[np.arange(len(c)), counts.argmax(1)]
    out = np.where(solid.any(1), mode, 0).astype(blocks.dtype)
    return out.reshape(Y // 2, Z // 2, X // 2)
