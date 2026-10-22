"""Tiny isometric voxel renderer (PIL, painter's algorithm). No GPU needed."""
from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COS30, SIN30 = math.cos(math.pi / 6), 0.5
SHADE = {"top": 1.0, "x": 0.78, "z": 0.62}


def render_iso(colors: np.ndarray, occupied: np.ndarray, size: int = 256,
               turns: int = 0, bg=(30, 32, 36)) -> Image.Image:
    """colors: (Y,Z,X,3) uint8, occupied: (Y,Z,X) bool. `turns` rotates 90deg around Y."""
    if turns:
        colors = np.rot90(colors, turns, axes=(1, 2))
        occupied = np.rot90(occupied, turns, axes=(1, 2))
    Y, Z, X = occupied.shape
    img = Image.new("RGB", (size, size), bg)
    if not occupied.any():
        return img
    pad = np.pad(occupied, 1)
    exposed = {
        "top": occupied & ~pad[2:, 1:-1, 1:-1],
        "x": occupied & ~pad[1:-1, 1:-1, 2:],
        "z": occupied & ~pad[1:-1, 2:, 1:-1],
    }
    span = (X + Z) * COS30
    height = Y + (X + Z) * SIN30
    s = 0.92 * size / max(span, height)
    ox = size / 2 - (X - Z) * COS30 * s / 2
    oy = (size - height * s) / 2 + Y * s

    def proj(x, y, z):
        return ox + (x - z) * COS30 * s, oy + ((x + z) * SIN30 - y) * s

    faces = []
    for kind, mask in exposed.items():
        y, z, x = np.nonzero(mask)
        for yi, zi, xi in zip(y.tolist(), z.tolist(), x.tolist()):
            faces.append((xi + yi + zi, kind, xi, yi, zi))
    faces.sort(key=lambda f: f[0])

    draw = ImageDraw.Draw(img)
    for _, kind, x, y, z in faces:
        c = colors[y, z, x].astype(np.float64) * SHADE[kind]
        fill = tuple(int(v) for v in c)
        if kind == "top":
            quad = [(x, y + 1, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
        elif kind == "x":
            quad = [(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)]
        else:
            quad = [(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)]
        draw.polygon([proj(*p) for p in quad], fill=fill)
    return img


_BLOCK_RGB: dict[str, tuple] | None = None
_SHAPE_SUFFIXES = ("_stairs", "_slab", "_wall", "_fence_gate", "_fence", "_trapdoor", "_door",
                   "_pressure_plate", "_button", "_sign", "_wall_sign", "_hanging_sign")
_FALLBACK_RGB = [("leaves", (60, 110, 45)), ("grass", (80, 130, 55)), ("fern", (80, 130, 55)),
                 ("water", (50, 80, 200)), ("lava", (230, 100, 20)), ("glass", (190, 215, 225)),
                 ("path", (150, 120, 70)), ("torch", (250, 200, 80)), ("lantern", (250, 200, 80)),
                 ("lamp", (250, 200, 80)), ("flower", (220, 80, 120)), ("chest", (160, 110, 50)),
                 ("barrel", (130, 95, 55)), ("bookshelf", (150, 110, 70)), ("chain", (60, 60, 70)),
                 ("bars", (110, 110, 115)), ("ladder", (150, 115, 70)), ("carpet", (170, 50, 50))]


@functools.lru_cache(maxsize=None)
def state_rgb(state: str) -> tuple:
    """Average color for any block state, e.g. 'minecraft:oak_stairs[facing=north]' -> oak
    planks color. Uses blocks.json (full blocks) and falls back to the base material."""
    global _BLOCK_RGB
    if _BLOCK_RGB is None:
        import json
        from pathlib import Path
        rows = json.loads((Path(__file__).parent / "blocks.json").read_text())["blocks"]
        _BLOCK_RGB = {b["name"].removeprefix("minecraft:"): tuple(b["rgb"]) for b in rows}
    n = state.split("[")[0].removeprefix("minecraft:")
    base = n
    for suf in _SHAPE_SUFFIXES:
        if n.endswith(suf):
            base = n[: -len(suf)]
            break
    for cand in (n, base, base + "_planks", base + "s", base + "_block", base.replace("_brick", "_bricks"),
                 n.replace("_log", "_planks").replace("_wood", "_planks"), n.removeprefix("stripped_"),
                 n.replace("waxed_", ""), n.replace("smooth_", ""), n.replace("cut_", ""),
                 n.replace("chiseled_", ""), n.replace("polished_", ""), base.replace("polished_", "")):
        if cand in _BLOCK_RGB:
            return _BLOCK_RGB[cand]
    for key, rgb in _FALLBACK_RGB:
        if key in n:
            return rgb
    return (128, 128, 128)


def palette_colors(palette: list[str]) -> np.ndarray:
    """(len(palette), 3) uint8 lookup table for a block-state palette."""
    return np.array([state_rgb(p) for p in palette], dtype=np.uint8)


def contact_sheet(tiles: list[tuple[list[Image.Image], str]], cols: int = 5,
                  tile: int = 256, caption_h: int = 48) -> Image.Image:
    """tiles: [(views, caption)]; each entry's views are laid side by side."""
    nviews = max(len(v) for v, _ in tiles)
    cell_w, cell_h = tile * nviews, tile + caption_h
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * cell_w + (cols - 1) * 6, rows * cell_h + (rows - 1) * 6),
                      (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, (views, caption) in enumerate(tiles):
        cx, cy = (i % cols) * (cell_w + 6), (i // cols) * (cell_h + 6)
        for j, v in enumerate(views):
            sheet.paste(v.resize((tile, tile)), (cx + j * tile, cy))
        draw.multiline_text((cx + 6, cy + tile + 4), _wrap(caption, cell_w // 7, 3),
                            fill=(220, 220, 220), font=font, spacing=1)
    return sheet


def _wrap(text: str, width: int, max_lines: int) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1] + "…"
    return "\n".join(lines)
