"""Build size control: the model always fills its 32^3 canvas, and we scale the result.

    "a coffee mug, 8 blocks tall"  ->  ("a coffee mug", Size(8, "height"))
    "a small house"                ->  ("a small house", Size(16, "longest"))

Generating at full canvas and shrinking afterwards keeps all 512 tokens for detail; a
model asked to build small would spend most of them on air. Size words stay in the prompt
(CLIP knows "small"), explicit "N blocks" phrases are removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ttmc.schem import crop

# Longest side in blocks for size words. None of these is "right"; they are defaults.
SIZE_WORDS = {"tiny": 8, "little": 12, "small": 16, "medium": 24, "big": 32, "large": 32,
              "huge": 48, "giant": 64, "gigantic": 64, "massive": 64}
DEFAULT = 32
MAX = 64                           # longest side cap; past this, upscaling only adds bulk
_NUM = re.compile(r",?\s*\b(\d{1,3})\s*(?:blocks?|b)\b\s*(tall|high|wide|long|big|across)?",
                  re.IGNORECASE)


@dataclass
class Size:
    blocks: int
    axis: str = "longest"          # "longest", "height" or "width" (longest horizontal side)


def parse_size(prompt: str) -> tuple[str, Size]:
    m = _NUM.search(prompt)
    if m:
        axis = {"tall": "height", "high": "height", "wide": "width", "long": "width",
                "across": "width"}.get((m.group(2) or "").lower(), "longest")
        clean = (prompt[:m.start()] + prompt[m.end():]).strip(" ,")
        return clean, Size(max(1, min(int(m.group(1)), MAX)), axis)
    for w in re.findall(r"[a-z]+", prompt.lower()):
        if w in SIZE_WORDS:
            return prompt, Size(SIZE_WORDS[w])
    return prompt, Size(DEFAULT)


def resize(blocks: np.ndarray, size: Size, min_fill: float = 0.1) -> np.ndarray:
    """Crop to the build and scale it uniformly so `size.axis` measures `size.blocks`.

    Shrinking: a target cell is solid if at least `min_fill` of the source voxels it covers
    are, and takes their most common block. Thin parts (chair legs) can vanish below about
    a third of the original size; that is the price of fewer blocks. Growing: nearest neighbor.
    """
    b = crop(blocks)
    if not b.any():
        return b
    Y, Z, X = b.shape
    ref = {"longest": max(Y, Z, X), "height": Y, "width": max(Z, X)}[size.axis]
    s = min(size.blocks / ref, MAX / max(Y, Z, X))
    shape = tuple(max(1, round(n * s)) for n in (Y, Z, X))
    if s >= 1:
        idx = [np.minimum((np.arange(n) / s).astype(int), m - 1) for n, m in zip(shape, b.shape)]
        return b[np.ix_(*idx)]
    # each source voxel goes to target cell floor(i * s)
    ty, tz, tx = (np.minimum((np.arange(n) * s).astype(int), m - 1) for n, m in zip(b.shape, shape))
    cell = (ty[:, None, None] * shape[1] + tz[None, :, None]) * shape[2] + tx[None, None, :]
    cell, src = cell.ravel(), b.ravel().astype(np.int64)
    n_cells, n_blocks = int(np.prod(shape)), int(src.max()) + 1
    volume = np.bincount(cell, minlength=n_cells)
    counts = np.bincount(cell * n_blocks + src, minlength=n_cells * n_blocks).reshape(n_cells, n_blocks)
    counts[:, 0] = 0                                       # air never wins the vote
    solid = counts.sum(1) >= min_fill * volume
    out = np.where(solid, counts.argmax(1), 0).astype(b.dtype)
    return out.reshape(shape)


def place(b: np.ndarray, res: int) -> np.ndarray:
    """Put a cropped build in a res^3 canvas: centered in X/Z, standing on the floor."""
    g = np.zeros((res, res, res), b.dtype)
    y, z, x = b.shape
    oz, ox = (res - z) // 2, (res - x) // 2
    g[:y, oz:oz + z, ox:ox + x] = b
    return g
