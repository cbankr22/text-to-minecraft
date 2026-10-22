"""Textured isometric renders of block grids, for captioning with a vision model.

Textures are read at render time from the user's local Minecraft jar (never copied into
the repo or shipped). Every block is drawn as a box: full cubes, half-height slabs, thin
carpets, and narrow posts for fences, panes, torches and plants. Stairs and doors are drawn
as full cubes, which reads fine at house scale.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MC = Path(os.environ.get("APPDATA", "")) / ".minecraft" / "versions"
SHADE = {"top": 1.0, "x": 0.80, "z": 0.64}
FOLIAGE, GRASS, WATER = (89, 174, 48), (145, 189, 89), (63, 118, 228)
TINTS = {"spruce_leaves": (97, 153, 97), "birch_leaves": (128, 167, 85)}
UNTINTED_LEAVES = {"cherry_leaves", "azalea_leaves", "flowering_azalea_leaves", "pale_oak_leaves"}

# Box shapes in block units: (x0, x1, y0, y1, z0, z1)
FULL = (0, 1, 0, 1, 0, 1)
POST = (0.375, 0.625, 0, 1, 0.375, 0.625)
SMALL = (0.3, 0.7, 0, 0.6, 0.3, 0.7)
CARPET = (0, 1, 0, 1 / 16, 0, 1)
POST_SUFFIXES = ("_fence", "_wall", "_pane", "iron_bars", "chain", "_rod", "end_rod")
SMALL_SUFFIXES = ("torch", "lantern", "_button", "flower_pot", "candle", "_sapling", "_mushroom",
                  "short_grass", "fern", "dandelion", "poppy", "tulip", "orchid", "allium",
                  "bluet", "daisy", "cornflower", "lily_of_the_valley", "bush", "sea_pickle")
CARPET_SUFFIXES = ("_carpet", "_pressure_plate", "rail", "snow", "lily_pad", "redstone_wire",
                   "leaf_litter", "pink_petals", "wildflowers")
SKIP = {"minecraft:air", "minecraft:light", "minecraft:barrier", "minecraft:structure_void"}


def shape_of(state: str) -> tuple | None:
    name = state.split("[")[0].removeprefix("minecraft:")
    if state.split("[")[0] in SKIP:
        return None
    if name.endswith("_slab"):
        if "type=top" in state:
            return (0, 1, 0.5, 1, 0, 1)
        return FULL if "type=double" in state else (0, 1, 0, 0.5, 0, 1)
    if name.endswith("_trapdoor") and "open=false" in state:
        return (0, 1, 0.8125, 1, 0, 1) if "half=top" in state else (0, 1, 0, 0.1875, 0, 1)
    if name.endswith(CARPET_SUFFIXES):
        return CARPET
    if name.endswith(POST_SUFFIXES) or name.endswith("_fence_gate") or "sign" in name:
        return POST
    if name.endswith(SMALL_SUFFIXES):
        return SMALL
    return FULL


def is_occluder(state: str) -> bool:
    s = shape_of(state)
    name = state.split("[")[0]
    return s == FULL and not any(k in name for k in ("glass", "leaves", "door", "stairs", "ice"))


class Textures:
    """Block name -> (top, side) RGBA 16x16 textures, resolved through the jar's models."""

    def __init__(self, jar: Path | None = None):
        jar = jar or MC / "26.2" / "26.2.jar"
        self.zf = zipfile.ZipFile(jar)
        self.names = set(self.zf.namelist())

    def _json(self, path: str) -> dict | None:
        return json.loads(self.zf.read(path)) if path in self.names else None

    def _model_textures(self, model: str) -> dict:
        tex: dict[str, str] = {}
        while model:
            m = self._json(f"assets/minecraft/models/{model.removeprefix('minecraft:')}.json")
            if m is None:
                break
            for k, v in m.get("textures", {}).items():
                if isinstance(v, dict):          # newer jars: {"sprite": "...", ...}
                    v = v.get("sprite", "")
                tex.setdefault(k, v)
            model = m.get("parent", "")
        for _ in range(5):                       # resolve '#side' style references
            tex = {k: (tex.get(v[1:], v) if v.startswith("#") else v) for k, v in tex.items()}
        return tex

    def _png(self, ref: str | None, name: str) -> Image.Image | None:
        if not ref or ref.startswith("#"):
            return None
        path = f"assets/minecraft/textures/{ref.removeprefix('minecraft:')}.png"
        if path not in self.names:
            return None
        img = Image.open(io.BytesIO(self.zf.read(path))).convert("RGBA")
        img = img.crop((0, 0, img.width, img.width)).resize((16, 16), Image.NEAREST)
        tint = self._tint(ref.split("/")[-1], name)
        if tint:
            arr = np.asarray(img).astype(np.float32)
            arr[..., :3] *= np.array(tint, dtype=np.float32) / 255
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")
        return img

    @staticmethod
    def _tint(tex: str, name: str):
        if tex.startswith("water"):
            return WATER
        if tex.endswith("_leaves") and tex not in UNTINTED_LEAVES:
            return TINTS.get(tex, FOLIAGE)
        if tex in ("grass_block_top", "short_grass", "tall_grass_top", "tall_grass_bottom", "fern",
                   "large_fern_top", "large_fern_bottom", "sugar_cane"):
            return GRASS
        if tex in ("vine", "lily_pad"):
            return FOLIAGE
        return None

    @lru_cache(maxsize=None)
    def faces(self, name: str, variant: str = "") -> tuple:
        """(top, side) images for a block name (+ 'axis=x' etc. for logs)."""
        bs = self._json(f"assets/minecraft/blockstates/{name}.json") or {}
        model = None
        if "variants" in bs:
            vs = bs["variants"]
            key = next((k for k in vs if variant and variant in k), next(iter(vs), None))
            v = vs.get(key) if key is not None else None
            model = (v[0] if isinstance(v, list) else v or {}).get("model")
        elif "multipart" in bs:
            app = bs["multipart"][0]["apply"]
            model = (app[0] if isinstance(app, list) else app).get("model")
        tex = self._model_textures(model) if model else {}
        pick = lambda keys: next((tex[k] for k in keys if k in tex), None)  # noqa: E731
        top = pick(("top", "end", "up", "all", "texture", "pane", "particle", "cross", "plant"))
        side = pick(("side", "all", "north", "texture", "front", "pane", "particle", "cross",
                     "plant", "bottom"))
        if name.endswith("_door"):
            top = side = pick(("top",) if "upper" in variant else ("bottom",)) or side
        if name == "water":
            top = side = "minecraft:block/water_still"
        t, s = self._png(top, name), self._png(side, name)
        if t is None and s is None:
            fallback = Image.new("RGBA", (16, 16), (150, 150, 150, 255))
            return fallback, fallback
        return t or s, s or t


def _project(a: float, x: float, y: float, z: float, top_y: float) -> tuple[float, float]:
    """Screen position of a point inside the unit cell; (a, 0) is the cell's top corner."""
    return a + (x - z) * a, (x + z) * a / 2 + (top_y - y) * a


@lru_cache(maxsize=4096)
def _face_sprite(tex_id: int, face: str, box: tuple, a: int, shade: float) -> Image.Image:
    tex = _TEX_REGISTRY[tex_id]
    x0, x1, y0, y1, z0, z1 = box
    if face == "top":
        pts3 = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        uv = lambda p: (16 * p[0], 16 * p[2])  # noqa: E731
    elif face == "x":
        pts3 = [(x1, y1, z1), (x1, y1, z0), (x1, y0, z0), (x1, y0, z1)]
        uv = lambda p: (16 * (1 - p[2]), 16 * (1 - p[1]))  # noqa: E731
    else:
        pts3 = [(x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1)]
        uv = lambda p: (16 * p[0], 16 * (1 - p[1]))  # noqa: E731
    scr = [_project(a, *p, top_y=1) for p in pts3]
    # Affine map screen -> texture from three corners.
    S = np.array([[scr[i][0], scr[i][1], 1] for i in range(3)])
    U = np.array([uv(pts3[i]) for i in range(3)])
    coef = np.linalg.solve(S, U)                 # 3x2
    size = (2 * a, 2 * a)
    data = (coef[0, 0], coef[1, 0], coef[2, 0], coef[0, 1], coef[1, 1], coef[2, 1])
    sprite = tex.transform(size, Image.AFFINE, data, resample=Image.NEAREST)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(scr, fill=255)
    alpha = np.minimum(np.asarray(sprite)[..., 3], np.asarray(mask))
    arr = np.asarray(sprite).astype(np.float32)
    arr[..., :3] *= shade
    arr[..., 3] = alpha
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


_TEX_REGISTRY: list[Image.Image] = []
_TEX_IDS: dict[int, int] = {}


def _tex_id(img: Image.Image) -> int:
    key = id(img)
    if key not in _TEX_IDS:
        _TEX_IDS[key] = len(_TEX_REGISTRY)
        _TEX_REGISTRY.append(img)
    return _TEX_IDS[key]


def render(blocks: np.ndarray, palette: list[str], include: np.ndarray, textures: Textures,
           turns: int = 0, size: int = 512, bg=(232, 232, 228)) -> Image.Image:
    """blocks: (Y, Z, X) palette indices; include: which cells to draw. Viewed from the
    (+x, +z) corner after rotating `turns` x 90 degrees around Y."""
    if turns:
        blocks = np.rot90(blocks, turns, axes=(1, 2))
        include = np.rot90(include, turns, axes=(1, 2))
    ys, zs, xs = np.nonzero(include)
    img = Image.new("RGBA", (size, size), bg + (255,))
    if len(ys) == 0:
        return img.convert("RGB")
    y0, z0, x0 = ys.min(), zs.min(), xs.min()
    Y, Z, X = ys.max() - y0 + 1, zs.max() - z0 + 1, xs.max() - x0 + 1
    span_w, span_h = X + Z, (X + Z) / 2 + Y
    a = max(2, int(2 * size * 0.92 / max(span_w, span_h)))   # render at ~2x, then downscale
    W, H = int(span_w * a) + 2 * a, int(span_h * a) + 2 * a
    canvas = Image.new("RGBA", (W, H), bg + (0,))

    shapes = [shape_of(p) for p in palette]
    occl = np.array([is_occluder(p) for p in palette])
    names = [p.split("[")[0].removeprefix("minecraft:") for p in palette]
    variants = [("axis=x" if "axis=x" in p else "axis=z" if "axis=z" in p else
                 "half=upper" if "half=upper" in p else "") for p in palette]
    sub_inc = include[y0:y0 + Y, z0:z0 + Z, x0:x0 + X]
    sub = blocks[y0:y0 + Y, z0:z0 + Z, x0:x0 + X]
    solid = occl[sub] & sub_inc
    pad = np.pad(solid, 1)
    open_top = ~pad[2:, 1:-1, 1:-1]
    open_x = ~pad[1:-1, 1:-1, 2:]
    open_z = ~pad[1:-1, 2:, 1:-1]
    cells = np.argwhere(sub_inc)
    order = np.argsort(cells[:, 0] + cells[:, 1] + cells[:, 2], kind="stable")
    for y, z, x in cells[order]:
        b = sub[y, z, x]
        box = shapes[b]
        if box is None:
            continue
        top_t, side_t = textures.faces(names[b], variants[b])
        if variants[b] == "axis=x":       # sideways log: the end shows on the x face
            faces = (("top", side_t, open_top), ("x", top_t, open_x), ("z", side_t, open_z))
        elif variants[b] == "axis=z":
            faces = (("top", side_t, open_top), ("x", side_t, open_x), ("z", top_t, open_z))
        else:
            faces = (("top", top_t, open_top), ("x", side_t, open_x), ("z", side_t, open_z))
        sx = int((x - z) * a + Z * a - a) + a
        sy = int((x + z) * a / 2 + (Y - 1 - y) * a)
        for face, tex, open_mask in faces:
            if box == FULL and not open_mask[y, z, x]:
                continue
            spr = _face_sprite(_tex_id(tex), face, box, a, SHADE[face])
            canvas.alpha_composite(spr, (sx, sy))
    canvas.thumbnail((size, size), Image.LANCZOS)
    img.alpha_composite(canvas, ((size - canvas.width) // 2, (size - canvas.height) // 2))
    return img.convert("RGB")
