"""Build the color-matchable block vocabulary from a local Minecraft client jar.

Only average colors are stored (ttmc/blocks.json), never the textures.

    python -m ttmc.block_colors            # auto-picks newest jar in %APPDATA%/.minecraft
    python -m ttmc.block_colors path/to/client.jar
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).with_name("blocks.json")

COLORS16 = ["white", "light_gray", "gray", "black", "brown", "red", "orange", "yellow",
            "lime", "green", "cyan", "light_blue", "blue", "purple", "magenta", "pink"]
WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry",
         "pale_oak", "bamboo", "crimson", "warped"]

# Full, opaque, mostly uniform blocks. Order defines token ids (air is 0).
CANDIDATES = (
    [f"{c}_wool" for c in COLORS16]
    + [f"{c}_concrete" for c in COLORS16]
    + ["terracotta"] + [f"{c}_terracotta" for c in COLORS16]
    + [f"{w}_planks" for w in WOODS]
    + ["oak_log", "spruce_log", "birch_log", "dark_oak_log", "stripped_oak_log"]
    + ["stone", "smooth_stone", "cobblestone", "mossy_cobblestone", "stone_bricks",
       "mossy_stone_bricks", "bricks", "andesite", "polished_andesite", "diorite",
       "polished_diorite", "granite", "polished_granite", "deepslate", "cobbled_deepslate",
       "deepslate_bricks", "deepslate_tiles", "tuff", "calcite", "dripstone_block",
       "blackstone", "polished_blackstone", "gilded_blackstone",
       "sandstone", "smooth_sandstone", "red_sandstone", "sand", "red_sand", "gravel",
       "dirt", "coarse_dirt", "podzol", "mud", "mud_bricks", "packed_mud", "clay",
       "moss_block", "snow_block", "ice", "packed_ice", "blue_ice",
       "quartz_block", "gold_block", "iron_block", "diamond_block", "emerald_block",
       "lapis_block", "redstone_block", "coal_block", "netherite_block", "copper_block",
       "exposed_copper", "weathered_copper", "oxidized_copper", "amethyst_block",
       "netherrack", "nether_bricks", "red_nether_bricks", "nether_wart_block",
       "warped_wart_block", "soul_sand", "end_stone", "end_stone_bricks", "purpur_block",
       "prismarine", "prismarine_bricks", "dark_prismarine", "obsidian", "crying_obsidian",
       "glowstone", "sea_lantern", "shroomlight", "honeycomb_block", "bone_block",
       "hay_block", "pumpkin", "melon", "dried_kelp_block", "sponge", "slime_block",
       "resin_block", "resin_bricks", "pale_moss_block"]
)
SUFFIXES = ["", "_side", "_top", "_front"]
# Blocks whose texture file is named differently.
TEXTURE_ALIASES = {"snow_block": "snow", "smooth_sandstone": "sandstone_top",
                   "dried_kelp_block": "dried_kelp_side"}


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def average_color(png: bytes) -> tuple[list[int], float]:
    """Alpha-weighted mean in linear light; returns sRGB ints and texture std (detail)."""
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float64)
    img = img[: img.shape[1]]                    # animated textures: first frame only
    rgb, a = img[..., :3], img[..., 3:] / 255.0
    lin = srgb_to_linear(rgb)
    mean = (lin * a).sum((0, 1)) / max(a.sum(), 1e-6)
    std = float(rgb[a[..., 0] > 0.5].std(axis=0).mean())
    return [int(round(v)) for v in linear_to_srgb(mean)], std


def find_jar() -> Path:
    versions = Path(os.environ["APPDATA"]) / ".minecraft" / "versions"
    jars = sorted(versions.glob("*/*.jar"), key=lambda p: p.stat().st_mtime, reverse=True)
    jars = [j for j in jars if j.stem == j.parent.name and "forge" not in j.stem]
    if not jars:
        sys.exit("No Minecraft client jar found; pass one explicitly.")
    return jars[0]


def main(argv: list[str]) -> None:
    jar = Path(argv[0]) if argv else find_jar()
    blocks, missing = [], []
    with zipfile.ZipFile(jar) as zf:
        names = set(zf.namelist())
        for block in CANDIDATES:
            base = TEXTURE_ALIASES.get(block, block)
            for suf in SUFFIXES:
                p = f"assets/minecraft/textures/block/{base}{suf}.png"
                if p in names:
                    rgb, std = average_color(zf.read(p))
                    blocks.append({"name": f"minecraft:{block}", "rgb": rgb,
                                   "detail": round(std, 1), "texture": f"{base}{suf}"})
                    break
            else:
                missing.append(block)
    OUT.write_text(json.dumps({"source_jar": jar.name, "blocks": blocks}, indent=1))
    print(f"{len(blocks)} blocks from {jar.name} -> {OUT}")
    if missing:
        print("missing textures:", ", ".join(missing))


if __name__ == "__main__":
    main(sys.argv[1:])
