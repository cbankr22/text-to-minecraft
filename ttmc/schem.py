"""Write block grids as Sponge schematic v2 (.schem), loadable with WorldEdit
(//schem load) or Litematica."""
from __future__ import annotations

from pathlib import Path

import nbtlib
import numpy as np
from nbtlib.tag import ByteArray, Compound, Int, IntArray, Short

DATA_VERSION_1_21_4 = 4189


def _varints(values: np.ndarray) -> bytes:
    out = bytearray()
    for v in values.tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def write_schem(blocks: np.ndarray, names: list[str], path: Path,
                data_version: int = DATA_VERSION_1_21_4) -> None:
    """blocks: (Y, Z, X) token grid; names[token] = 'minecraft:...'. Air is kept so the
    schematic pastes as a clean box; crop first if you want it tight."""
    used = np.unique(blocks)
    palette = {names[t]: i for i, t in enumerate(used)}
    lut = np.zeros(int(blocks.max()) + 1, dtype=np.int64)
    for i, t in enumerate(used):
        lut[t] = i
    h, l, w = blocks.shape
    data = _varints(lut[blocks].ravel())                  # index = x + z*W + y*W*L
    root = Compound({
        "Version": Int(2), "DataVersion": Int(data_version),
        "Width": Short(w), "Height": Short(h), "Length": Short(l),
        "Offset": IntArray([0, 0, 0]),
        "PaletteMax": Int(len(palette)),
        "Palette": Compound({k: Int(v) for k, v in palette.items()}),
        "BlockData": ByteArray(np.frombuffer(data, dtype=np.uint8).astype(np.int8)),
    })
    nbtlib.File({"Schematic": root}, gzipped=True).save(path) if False else \
        nbtlib.File(root, gzipped=True, root_name="Schematic").save(path)


def crop(blocks: np.ndarray) -> np.ndarray:
    ys, zs, xs = np.nonzero(blocks)
    if not len(ys):
        return blocks
    return blocks[ys.min():ys.max() + 1, zs.min():zs.max() + 1, xs.min():xs.max() + 1]
