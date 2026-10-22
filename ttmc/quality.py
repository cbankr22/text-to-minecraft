"""Geometry-based quality checks for voxelized objects.

All checks work on a (Y,Z,X) token grid (0 = air) and return metrics plus a list
of reject reasons; an empty list means keep.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

MIN_VOXELS = 300
MIN_MID_EXTENT = 6        # 2nd-largest bbox side; below this the object is a stick
MIN_SMALL_EXTENT = 2      # smallest bbox side; 1-voxel-thick objects are tiles/planes
BOX_FACE_FILL = 0.9       # bbox face this full counts as a wall
BOX_MIN_WALLS = 5         # this many walls plus hidden content -> boxed (skybox, room shell)
BOX_HIDDEN_FRAC = 0.02    # share of voxels strictly inside the walls
SCATTER_LARGEST_FRAC = 0.5
SCATTER_MIN_COMPONENTS = 15


def assess(blocks: np.ndarray) -> tuple[dict, list[str]]:
    occ = blocks > 0
    n = int(occ.sum())
    metrics: dict = {"n_voxels": n}
    if n == 0:
        return metrics, ["empty"]

    ys, zs, xs = np.nonzero(occ)
    lo = np.array([ys.min(), zs.min(), xs.min()])
    hi = np.array([ys.max(), zs.max(), xs.max()]) + 1
    ext = hi - lo
    small, mid, _ = sorted(ext.tolist())
    box = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    walls = [box[0], box[-1], box[:, 0], box[:, -1], box[:, :, 0], box[:, :, -1]]
    n_walls = sum(float(w.mean()) >= BOX_FACE_FILL for w in walls)
    hidden = int(box[1:-1, 1:-1, 1:-1].sum()) if min(ext) > 2 else 0

    labels, n_comp = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    largest_frac = float(sizes.max() / n)

    metrics.update(extent_yzx=ext.tolist(), fill=round(n / box.size, 3), box_walls=n_walls, hidden=hidden,
                   components=int(n_comp), largest_component=round(largest_frac, 3))

    reasons = []
    if n < MIN_VOXELS:
        reasons.append("sparse")
    if mid < MIN_MID_EXTENT:
        reasons.append("thin")
    elif small < MIN_SMALL_EXTENT:
        reasons.append("flat")
    if n_walls >= BOX_MIN_WALLS and hidden >= BOX_HIDDEN_FRAC * n:
        reasons.append("boxed")
    if largest_frac < SCATTER_LARGEST_FRAC and n_comp >= SCATTER_MIN_COMPONENTS:
        reasons.append("scattered")
    return metrics, reasons
