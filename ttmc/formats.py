"""Readers for Minecraft build formats.

Every reader returns a Build: a dense (Y, Z, X) uint16 array of indices into a
palette of block-state strings like "minecraft:oak_stairs[facing=north]".
Index 0 of every palette is always "minecraft:air".

Supported:
  .schem       Sponge schematic v1/v2/v3 (WorldEdit, modern)
  .litematic   Litematica (all regions merged into one grid)
  .schematic   MCEdit/legacy (numeric IDs kept as "legacy:<id>:<data>")
  worlds       Anvil region files (.mca), Java 1.13+ (DataVersion >= 1451)
"""
from __future__ import annotations

import io
import math
import struct
import zlib
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import nbtlib
import numpy as np

AIR = "minecraft:air"
AIR_LIKE = {AIR, "minecraft:cave_air", "minecraft:void_air"}

# 20w17a (1.16 snapshot): block-state longs stopped spanning across longs.
DATA_VERSION_NO_SPAN = 2529
DATA_VERSION_1_13 = 1451


@dataclass
class Build:
    blocks: np.ndarray                      # (Y, Z, X) uint16
    palette: list[str]                      # palette[0] == AIR
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.blocks.shape)

    def block_counts(self) -> dict[str, int]:
        counts = np.bincount(self.blocks.ravel(), minlength=len(self.palette))
        return {self.palette[i]: int(c) for i, c in enumerate(counts) if c}

    def save(self, path: Path) -> None:
        np.savez_compressed(path, blocks=self.blocks, palette=np.array(self.palette))


def load_build(path: Path) -> Build:
    data = np.load(path)
    return Build(data["blocks"], [str(p) for p in data["palette"]])


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _load_nbt(path: Path) -> nbtlib.Compound:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nbtlib.File.parse(io.BytesIO(raw))


def block_state_string(entry) -> str:
    """Palette compound {Name, Properties} -> 'name[k=v,...]'."""
    name = str(entry["Name"])
    props = entry.get("Properties")
    if props:
        inner = ",".join(f"{k}={props[k]}" for k in sorted(props))
        return f"{name}[{inner}]"
    return name


def _remap_air_first(indices: np.ndarray, palette: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reorder palette so all air variants collapse to index 0."""
    new_palette = [AIR]
    lut = np.zeros(len(palette), dtype=np.uint16)
    seen: dict[str, int] = {AIR: 0}
    for i, name in enumerate(palette):
        key = AIR if name in AIR_LIKE else name
        if key not in seen:
            seen[key] = len(new_palette)
            new_palette.append(key)
        lut[i] = seen[key]
    return lut[indices], new_palette


def unpack_longs(longs, count: int, bits: int, spanning: bool) -> np.ndarray:
    """Unpack `count` values of `bits` width from a Minecraft long array."""
    arr = np.asarray(longs, dtype=np.int64).view(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    if not spanning:
        per_long = 64 // bits
        shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
        vals = (arr[:, None] >> shifts[None, :]) & mask
        return vals.ravel()[:count].astype(np.int64)
    # Values may straddle two longs.
    arr = np.concatenate([arr, np.zeros(1, dtype=np.uint64)])
    bit_idx = np.arange(count, dtype=np.int64) * bits
    li = bit_idx // 64
    off = (bit_idx % 64).astype(np.uint64)
    lo = arr[li] >> off
    hi_shift = np.uint64(64) - off
    straddle = (off + np.uint64(bits)) > np.uint64(64)
    hi = np.where(straddle, arr[li + 1] << np.where(straddle, hi_shift, np.uint64(0)), np.uint64(0))
    return ((lo | hi) & mask).astype(np.int64)


def _read_varints(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    i = pos = 0
    n = len(data)
    while i < count and pos < n:
        value = shift = 0
        while True:
            b = data[pos]
            pos += 1
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        out[i] = value
        i += 1
    if i != count:
        raise ValueError(f"expected {count} varints, got {i}")
    return out


# --------------------------------------------------------------------------- #
# Sponge .schem
# --------------------------------------------------------------------------- #

def read_sponge(path: Path) -> Build:
    root = _load_nbt(path)
    if "Schematic" in root:            # v3 wraps everything
        root = root["Schematic"]
    version = int(root.get("Version", 1))
    w, h, l = int(root["Width"]), int(root["Height"]), int(root["Length"])
    if version >= 3:
        blocks = root["Blocks"]
        pal_tag, data = blocks["Palette"], blocks["Data"]
    else:
        pal_tag, data = root["Palette"], root["BlockData"]
    palette = [""] * len(pal_tag)
    for name, idx in pal_tag.items():
        palette[int(idx)] = str(name)
    raw = np.asarray(data, dtype=np.int8).view(np.uint8).tobytes()
    idx = _read_varints(raw, w * h * l).reshape(h, l, w)
    blocks, palette = _remap_air_first(idx, palette)
    return Build(blocks, palette, {"format": f"sponge_v{version}",
                                   "data_version": int(root.get("DataVersion", 0))})


# --------------------------------------------------------------------------- #
# Litematica .litematic
# --------------------------------------------------------------------------- #

def read_litematic(path: Path) -> Build:
    root = _load_nbt(path)
    regions = []
    for name, reg in root["Regions"].items():
        pos = [int(reg["Position"][k]) for k in "xyz"]
        size = [int(reg["Size"][k]) for k in "xyz"]
        # Negative size means the region extends backwards from Position.
        origin = [p + s + 1 if s < 0 else p for p, s in zip(pos, size)]
        sx, sy, sz = (abs(s) for s in size)
        pal = [block_state_string(e) for e in reg["BlockStatePalette"]]
        bits = max(2, math.ceil(math.log2(len(pal)))) if len(pal) > 1 else 2
        idx = unpack_longs(reg["BlockStates"], sx * sy * sz, bits, spanning=True)
        regions.append((origin, (sy, sz, sx), idx.reshape(sy, sz, sx), pal))

    lo = np.min([r[0] for r in regions], axis=0)
    hi = np.max([[o[0] + s[2], o[1] + s[0], o[2] + s[1]] for o, s, *_ in regions], axis=0)
    X, Y, Z = (hi - lo).tolist()
    lookup = {AIR: 0}
    out = np.zeros((Y, Z, X), dtype=np.uint16)
    for origin, (sy, sz, sx), idx, pal in regions:
        lut = np.array([lookup.setdefault(AIR if p in AIR_LIKE else p, len(lookup)) for p in pal],
                       dtype=np.uint16)
        ox, oy, oz = (np.array(origin) - lo).tolist()
        region = lut[idx]
        target = out[oy:oy + sy, oz:oz + sz, ox:ox + sx]
        np.copyto(target, region, where=region != 0)
    return Build(out, list(lookup), {"format": "litematic",
                                "data_version": int(root.get("MinecraftDataVersion", 0)),
                                "regions": len(regions)})


# --------------------------------------------------------------------------- #
# MCEdit .schematic (pre-1.13 numeric IDs)
# --------------------------------------------------------------------------- #

def read_mcedit(path: Path) -> Build:
    root = _load_nbt(path)
    w, h, l = int(root["Width"]), int(root["Height"]), int(root["Length"])
    ids = np.asarray(root["Blocks"], dtype=np.int8).view(np.uint8).astype(np.int32)
    if "AddBlocks" in root:   # ids above 255, packed as nibbles
        add = np.asarray(root["AddBlocks"], dtype=np.int8).view(np.uint8)
        nib = np.empty(add.size * 2, dtype=np.int32)
        nib[0::2], nib[1::2] = add >> 4, add & 0x0F
        ids |= nib[: ids.size] << 8
    meta = np.asarray(root["Data"], dtype=np.int8).view(np.uint8).astype(np.int32) & 0x0F
    combined = (ids << 4) | meta
    uniq, inv = np.unique(combined, return_inverse=True)
    palette = [AIR if (u >> 4) == 0 else f"legacy:{u >> 4}:{u & 15}" for u in uniq]
    blocks, palette = _remap_air_first(inv.reshape(h, l, w), palette)
    return Build(blocks, palette, {"format": "mcedit", "legacy_ids": True})


# --------------------------------------------------------------------------- #
# Worlds: level.dat + Anvil region files
# --------------------------------------------------------------------------- #

def read_level_dat(path: Path) -> dict:
    data = _load_nbt(path)["Data"]
    info = {
        "level_name": str(data.get("LevelName", "")),
        "data_version": int(data.get("DataVersion", 0)),
        "mc_version": str(data["Version"]["Name"]) if "Version" in data else None,
        "generator": None,
        "flat_layers": None,
    }
    wgs = data.get("WorldGenSettings")
    if wgs is not None:
        gen = wgs["dimensions"]["minecraft:overworld"]["generator"]
        info["generator"] = str(gen["type"]).removeprefix("minecraft:")
        settings = gen.get("settings")
        if info["generator"] == "flat" and isinstance(settings, nbtlib.Compound):
            info["flat_layers"] = [(str(l["block"]), int(l["height"]))
                                   for l in settings.get("layers", [])]
    elif "generatorName" in data:
        info["generator"] = str(data["generatorName"]).lower()
    if info["generator"] == "flat" and info["flat_layers"] is not None:
        if all(b in AIR_LIKE for b, _ in info["flat_layers"]):
            info["generator"] = "void"
    return info


def read_region_chunk(raw: bytes, i: int) -> nbtlib.Compound | None:
    """Chunk number i (= (cz & 31) * 32 + (cx & 31)) of a region file's bytes, or None."""
    if len(raw) < 8192:
        return None
    loc = struct.unpack_from(">I", raw, i * 4)[0]
    offset, sectors = loc >> 8, loc & 0xFF
    if offset == 0 or sectors == 0:
        return None
    start = offset * 4096
    length, ctype = struct.unpack_from(">IB", raw, start)
    payload = raw[start + 5: start + 4 + length]
    if ctype & 0x80:            # stored externally in c.x.z.mcc; skip
        return None
    if ctype == 1:
        payload = gzip.decompress(payload)
    elif ctype == 2:
        payload = zlib.decompress(payload)
    elif ctype != 3:
        return None
    return nbtlib.File.parse(io.BytesIO(payload))


def iter_region_chunks(path: Path) -> Iterator[nbtlib.Compound]:
    raw = path.read_bytes()
    for i in range(1024):
        chunk = read_region_chunk(raw, i)
        if chunk is not None:
            yield chunk


def decode_chunk_sections(chunk: nbtlib.Compound) -> Iterator[tuple[int, np.ndarray, list[str]]]:
    """Yield (section_y, (16,16,16) index array, palette) for each section."""
    data_version = int(chunk.get("DataVersion", 0))
    if "sections" in chunk:                          # 1.18+
        sections = chunk["sections"]
        get_pal = lambda s: s["block_states"].get("palette") if "block_states" in s else None
        get_data = lambda s: s["block_states"].get("data")
    else:                                            # 1.13 - 1.17
        sections = chunk["Level"].get("Sections", [])
        get_pal = lambda s: s.get("Palette")
        get_data = lambda s: s.get("BlockStates")
    spanning = data_version < DATA_VERSION_NO_SPAN
    for s in sections:
        pal = get_pal(s)
        if not pal:
            continue
        palette = [block_state_string(e) for e in pal]
        data = get_data(s)
        if data is None or len(palette) == 1:
            idx = np.zeros(4096, dtype=np.int64)
        else:
            bits = max(4, math.ceil(math.log2(len(palette))))
            idx = unpack_longs(data, 4096, bits, spanning)
        yield int(s["Y"]), idx.reshape(16, 16, 16), palette


def chunk_pos(chunk: nbtlib.Compound) -> tuple[int, int]:
    src = chunk if "xPos" in chunk else chunk["Level"]
    return int(src["xPos"]), int(src["zPos"])
