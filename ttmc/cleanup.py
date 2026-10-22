"""Post-processing for generated builds: drop floating specks.

Generated grids sometimes have a few stray blocks hovering around the main build. A build
is kept as its connected pieces (26-neighbourhood: blocks touching by a face, edge or
corner belong together); pieces much smaller than the largest one are removed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def remove_specks(grid: np.ndarray, rel: float = 0.01, min_size: int = 8) -> np.ndarray:
    """grid (Y, Z, X) uint8 block tokens, 0 = air. Removes pieces smaller than
    max(min_size, rel * largest piece). The largest piece always stays."""
    labels, n = ndimage.label(grid > 0, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return grid
    sizes = np.bincount(labels.ravel())[1:]
    keep = sizes >= max(min_size, rel * sizes.max())
    keep[sizes.argmax()] = True
    out = grid.copy()
    out[~np.concatenate([[True], keep])[labels]] = 0
    return out
