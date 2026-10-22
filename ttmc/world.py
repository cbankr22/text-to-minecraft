"""Random access to a Java world's overworld blocks (1.13+ region files)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ttmc.formats import AIR, AIR_LIKE, decode_chunk_sections, read_region_chunk


class World:
    """Chunk access with a cache of region-file bytes and of parsed chunks
    (call clear() now and then to free the parsed ones)."""

    def __init__(self, path: Path):
        self.region_dir = path / "dimensions" / "minecraft" / "overworld" / "region"  # 26.x
        if not self.region_dir.exists():
            self.region_dir = path / "region"                                       # older
        self.raw: dict[tuple[int, int], bytes] = {}
        self.parsed: dict[tuple[int, int], object] = {}

    def chunk(self, cx: int, cz: int):
        if (cx, cz) not in self.parsed:
            key = (cx >> 5, cz >> 5)
            if key not in self.raw:
                f = self.region_dir / f"r.{key[0]}.{key[1]}.mca"
                self.raw[key] = f.read_bytes() if f.exists() else b""
            self.parsed[cx, cz] = read_region_chunk(self.raw[key], (cz & 31) * 32 + (cx & 31))
        return self.parsed[cx, cz]

    def clear(self) -> None:
        self.parsed.clear()

    def regions(self) -> list[tuple[int, int]]:
        out = []
        for f in sorted(self.region_dir.glob("r.*.*.mca")):
            _, rx, rz, _ = f.name.split(".")
            out.append((int(rx), int(rz)))
        return out

    def region_groups(self) -> list[list[tuple[int, int]]]:
        """Regions split into groups of neighbours (worlds can have far-off islands of
        explored land; one dense map spanning them all would not fit in memory)."""
        left, groups = set(self.regions()), []
        while left:
            stack, group = [left.pop()], []
            while stack:
                rx, rz = stack.pop()
                group.append((rx, rz))
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nb = (rx + d[0], rz + d[1])
                    if nb in left:
                        left.remove(nb)
                        stack.append(nb)
            groups.append(sorted(group))
        return sorted(groups, key=len, reverse=True)

    def iter_chunks(self, regions: list[tuple[int, int]] | None = None
                    ) -> Iterator[tuple[int, int, object]]:
        """Every stored chunk as (cx, cz, nbt); keeps one region in memory at a time."""
        for rx, rz in regions or self.regions():
            for i in range(1024):
                cx, cz = rx * 32 + (i & 31), rz * 32 + (i >> 5)
                ch = self.chunk(cx, cz)
                if ch is not None:
                    yield cx, cz, ch
            self.clear()
            self.raw.pop((rx, rz), None)


def is_full(chunk) -> bool:
    return chunk is not None and str(chunk.get("Status", "minecraft:full")).endswith("full")


def cut_box(world: World, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int):
    """Blocks in [x0,x1) x [y0,y1) x [z0,z1) as (Y,Z,X) uint16 + palette (0 = air).
    Missing chunks stay air; the count of missing chunks is returned too."""
    out = np.zeros((y1 - y0, z1 - z0, x1 - x0), dtype=np.uint16)
    lookup = {AIR: 0}
    missing = 0
    for cx in range(x0 >> 4, ((x1 - 1) >> 4) + 1):
        for cz in range(z0 >> 4, ((z1 - 1) >> 4) + 1):
            ch = world.chunk(cx, cz)
            if not is_full(ch):
                missing += 1
                continue
            ax0, az0 = max(x0, cx * 16), max(z0, cz * 16)
            ax1, az1 = min(x1, cx * 16 + 16), min(z1, cz * 16 + 16)
            for sy, idx, pal in decode_chunk_sections(ch):
                ay0, ay1 = max(y0, sy * 16), min(y1, sy * 16 + 16)
                if ay0 >= ay1:
                    continue
                lut = np.array([lookup.setdefault(AIR if p in AIR_LIKE else p, len(lookup))
                                for p in pal], dtype=np.uint16)
                part = idx[ay0 - sy * 16: ay1 - sy * 16,
                           az0 - cz * 16: az1 - cz * 16,
                           ax0 - cx * 16: ax1 - cx * 16]
                out[ay0 - y0: ay1 - y0, az0 - z0: az1 - z0, ax0 - x0: ax1 - x0] = lut[part]
    return out, list(lookup), missing
