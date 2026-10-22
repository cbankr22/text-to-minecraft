"""Cut the player builds out of a downloaded Minecraft world.

    .venv/Scripts/python scripts/ingest_world.py data/worlds/celestia_town --license "ARR, use with credit"

1. Scan every chunk once and classify each block: natural ground (stone, dirt, sand...),
   other natural (leaves, upright logs, plants, water...), ground edits (paths, farmland)
   or BUILT (everything a player places: planks, stairs, glass, bricks, sideways logs...).
   Per column keep: lowest/highest built block, number of built blocks, natural ground below.
2. A column belongs to a build when it has >= 2 built blocks reaching the surface (a road
   or a pier is only 1 thick) or a built block >= 3 above the ground (roof overhangs).
   Neighbouring build columns are grouped into footprints (one per build or terrace).
3. Each footprint is cut out with MARGIN blocks of context around it, in full block states,
   with a mask of the build's own blocks. So one file serves both uses:
     - the build alone (blocks where mask, air elsewhere): an Objaverse-style object
     - the build in its terrain: a v2 inpainting example (mask the box, keep the context)

Output: data/worlds/builds/<world>/<uid>.npz (blocks, palette, mask) + one manifest row
per build in data/worlds/builds/manifest.jsonl. Browse them with make_build_gallery.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.formats import AIR_LIKE, decode_chunk_sections, read_level_dat  # noqa: E402
from ttmc.render import palette_colors  # noqa: E402
from ttmc.world import World, cut_box, is_full  # noqa: E402

OUT = ROOT / "data" / "worlds" / "builds"
Y_MIN, HEIGHT = -64, 384
MARGIN = 6                  # context blocks kept around each footprint
MIN_AREA = 12               # columns; smaller groups are lamp posts, signs, statues
MIN_HEIGHT = 3              # blocks from ground to top

AIR, GROUND, NATURE, EDIT, BUILT = 0, 1, 2, 3, 4

GROUND_BLOCKS = {
    "stone", "granite", "diorite", "andesite", "deepslate", "tuff", "calcite", "dirt",
    "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "grass_block", "mud", "clay", "gravel",
    "sand", "red_sand", "sandstone", "red_sandstone", "terracotta", "snow_block", "powder_snow",
    "ice", "packed_ice", "blue_ice", "bedrock", "moss_block", "dripstone_block", "smooth_basalt",
    "magma_block", "amethyst_block", "budding_amethyst", "raw_iron_block", "raw_copper_block",
    "pale_moss_block", "water", "lava",
}
NATURE_BLOCKS = {
    "short_grass", "tall_grass", "grass", "fern", "large_fern", "dead_bush", "bush", "firefly_bush",
    "vine", "glow_lichen", "sugar_cane", "kelp", "kelp_plant", "seagrass", "tall_seagrass",
    "lily_pad", "cactus", "cactus_flower", "sweet_berry_bush", "bamboo", "bamboo_sapling", "snow",
    "moss_carpet", "pale_moss_carpet", "pale_hanging_moss", "pointed_dripstone", "bee_nest",
    "azalea", "flowering_azalea", "cave_vines", "cave_vines_plant", "hanging_roots",
    "spore_blossom", "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "mangrove_roots",
    "muddy_mangrove_roots", "mangrove_propagule", "leaf_litter", "pink_petals", "wildflowers",
    "brown_mushroom", "red_mushroom", "brown_mushroom_block", "red_mushroom_block",
    "mushroom_stem", "sea_pickle", "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet",
    "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower",
    "lily_of_the_valley", "sunflower", "lilac", "rose_bush", "peony", "pitcher_plant",
    "open_eyeblossom", "closed_eyeblossom", "twisting_vines", "weeping_vines", "bubble_column",
    "cobweb", "pumpkin", "melon", "sculk", "sculk_vein", "dried_ghast", "short_dry_grass",
    "tall_dry_grass",
}
EDIT_BLOCKS = {"dirt_path", "farmland"}


def block_class(state: str) -> int:
    if state in AIR_LIKE:
        return AIR
    name = state.split("[")[0].removeprefix("minecraft:")
    if name in GROUND_BLOCKS or name.endswith("_ore") or (
            name.endswith("_terracotta") and "glazed" not in name):
        return GROUND
    if name in EDIT_BLOCKS:
        return EDIT
    if name.endswith("_leaves") and "persistent=true" in state:
        return BUILT        # placed by a player: hedges, custom trees
    if name in NATURE_BLOCKS or name.endswith(("_leaves", "_coral", "_coral_fan", "_coral_block",
                                                "_sapling")):
        return NATURE
    return BUILT        # includes logs; classify() turns the ones that belong to trees to NATURE


def is_tree_log(state: str) -> bool:
    """Blocks a tree can be made of: logs, and in custom (map-maker) trees also 6-sided wood
    blocks and fence-post twigs. classify() decides per cluster whether it really is a tree."""
    name = state.split("[")[0].removeprefix("minecraft:")
    return (name.endswith(("_log", "_stem", "_wood", "_hyphae", "_fence"))
            and not name.startswith("stripped_"))


def is_upright_log(state: str) -> bool:
    name = state.split("[")[0]
    return name.endswith(("_log", "_stem")) and "axis=x" not in state and "axis=z" not in state


def is_leaves(state: str) -> bool:
    """Natural leaves only: player-placed leaves carry persistent=true (they never decay)."""
    return state.split("[")[0].endswith("_leaves") and "persistent=true" not in state


_FLAGS: dict[str, tuple[int, bool, bool, bool]] = {}


def classify(idx: np.ndarray, palette: list[str]) -> np.ndarray:
    """Block classes for a (Y, Z, X) grid of palette indices.

    Logs are only natural when they are part of a tree: a connected cluster of logs and
    leaves with at least half as many leaves as logs. A timber-frame house has many logs and
    at most a few decorative leaves, so its beams and corner posts stay BUILT. A cluster of
    upright logs cut off at the grid's side (a trunk whose crown is in the next chunk) also
    counts as a tree."""
    for p in palette:
        if p not in _FLAGS:
            _FLAGS[p] = (block_class(p), is_tree_log(p), is_leaves(p),
                         is_tree_log(p) and not is_upright_log(p))
    flags = [_FLAGS[p] for p in palette]
    cls = np.array([f[0] for f in flags], dtype=np.uint8)[idx]
    log = np.array([f[1] for f in flags])[idx]
    if not log.any():
        return cls
    leaf = np.array([f[2] for f in flags])[idx]
    sideways = np.array([f[3] for f in flags])[idx]
    lab, n = ndimage.label(log | leaf, np.ones((3, 3, 3)))
    n_log = np.bincount(lab[log], minlength=n + 1)
    n_leaf = np.bincount(lab[leaf], minlength=n + 1)
    n_side = np.bincount(lab[sideways], minlength=n + 1)
    edge = np.zeros(n + 1, dtype=bool)
    for face in (lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
        edge[np.unique(face)] = True
    tree = (n_leaf * 2 >= n_log) | (edge & (n_side == 0))   # n_side: anything but upright logs
    tree[0] = False
    cls[log & tree[lab]] = NATURE
    return cls


WORLDGEN_JUNK = {   # footprints made mostly of these are boulders / ruined portals, not builds
    "boulder": {"mossy_cobblestone"},
    "ruined_portal": {"obsidian", "crying_obsidian", "netherrack", "magma_block"},
}


def worldgen_overlap(mask: np.ndarray, origin: tuple, structures: list) -> str | None:
    """'worldgen:<name>' when >= 25% of the build's blocks lie inside one worldgen
    structure's pieces (villages, monuments, ruins, mineshafts... recorded in the chunks)."""
    x0, y0, z0 = origin
    Y, Z, X = mask.shape
    total = mask.sum()
    inside: dict[str, int] = {}
    for name, (bx0, by0, bz0, bx1, by1, bz1) in structures:
        sl = (slice(max(by0 - y0, 0), max(min(by1 + 1 - y0, Y), 0)),
              slice(max(bz0 - z0, 0), max(min(bz1 + 1 - z0, Z), 0)),
              slice(max(bx0 - x0, 0), max(min(bx1 + 1 - x0, X), 0)))
        n = int(mask[sl].sum())
        if n:
            inside[name] = inside.get(name, 0) + n
    for name, n in sorted(inside.items(), key=lambda kv: -kv[1]):
        if n >= 0.25 * total:
            return f"worldgen:{name}"
    return None


def junk_reason(materials_all: dict[str, int]) -> str | None:
    total = sum(materials_all.values())
    if len(materials_all) <= 2 and max(materials_all.values()) >= 0.85 * total:
        return "single_material_blob"      # decorative rocks/hills (e.g. acacia-wood boulders)
    for reason, names in WORLDGEN_JUNK.items():
        if sum(materials_all.get(n, 0) for n in names) >= 0.5 * total:
            return reason
    return None


# --------------------------------------------------------------------------- #
# Pass 1: per-column summary of the whole world
# --------------------------------------------------------------------------- #

def scan_world(world: World, regions: list[tuple[int, int]], colors: bool = False):
    """2D maps over the world's extent (Z, X): built count, lowest/highest built y,
    highest ground y overall and below the lowest built block. With colors=True also the
    highest non-air block's y ("topany") and its average color ("rgb", Z x X x 3) for maps."""
    rx0, rz0 = min(r[0] for r in regions), min(r[1] for r in regions)
    rx1, rz1 = max(r[0] for r in regions) + 1, max(r[1] for r in regions) + 1
    shape = ((rz1 - rz0) * 512, (rx1 - rx0) * 512)
    maps = {k: np.full(shape, -1, dtype=np.int16) for k in ("count", "low", "top", "surf", "ground")}
    maps["count"][:] = 0
    if colors:
        maps["topany"] = np.full(shape, -1, dtype=np.int16)
        maps["rgb"] = np.zeros(shape + (3,), dtype=np.uint8)
    present = np.zeros((shape[0] // 16, shape[1] // 16), dtype=bool)
    structures = []         # (name, [x0, y0, z0, x1, y1, z1]) of every worldgen structure piece
    ys = np.arange(HEIGHT)[:, None, None]
    t0, n = time.time(), 0
    for cx, cz, ch in world.iter_chunks(regions):
        if not is_full(ch):
            continue
        grid = np.zeros((HEIGHT, 16, 16), dtype=np.int32)    # index into chunk palette
        palette = ["minecraft:air"]
        for sy, idx, pal in decode_chunk_sections(ch):
            y = sy * 16 - Y_MIN
            if 0 <= y < HEIGHT:
                grid[y:y + 16] = idx + len(palette)
                palette += pal
        cls = classify(grid, palette)
        built = cls == BUILT
        ground = (cls == GROUND) | (cls == EDIT)
        any_b, any_g = built.any(0), ground.any(0)
        low = np.where(any_b, np.argmax(built, 0), -1)
        top = np.where(any_b, HEIGHT - 1 - np.argmax(built[::-1], 0), -1)
        surf = np.where(any_g, HEIGHT - 1 - np.argmax(ground[::-1], 0), -1)
        below = ground & (ys < np.where(any_b, low, HEIGHT)[None])
        gnd = np.where(below.any(0), HEIGHT - 1 - np.argmax(below[::-1], 0), surf)
        z, x = (cz - rz0 * 32) * 16, (cx - rx0 * 32) * 16
        for k, v in (("count", built.sum(0)), ("low", low), ("top", top), ("surf", surf),
                     ("ground", gnd)):
            maps[k][z:z + 16, x:x + 16] = v
        present[cz - rz0 * 32, cx - rx0 * 32] = True
        if colors:
            solid = cls != AIR
            tany = np.where(solid.any(0), HEIGHT - 1 - np.argmax(solid[::-1], 0), 0)
            zz, xx = np.mgrid[0:16, 0:16]
            lut = palette_colors(palette)
            maps["topany"][z:z + 16, x:x + 16] = np.where(solid.any(0), tany, -1)
            maps["rgb"][z:z + 16, x:x + 16] = lut[grid[tany, zz, xx]]
        for sname, start in ch.get("structures", {}).get("starts", {}).items():
            if str(start.get("id", "INVALID")) != "INVALID":
                for piece in start.get("Children", []):
                    structures.append((str(sname).removeprefix("minecraft:"),
                                       [int(v) for v in piece["BB"]]))
        n += 1
        if n % 2000 == 0:
            print(f"  scanned {n} chunks ({time.time() - t0:.0f}s)", flush=True)
    print(f"  scanned {n} chunks in {time.time() - t0:.0f}s", flush=True)
    return maps, present, (rx0 * 512, rz0 * 512), structures


# --------------------------------------------------------------------------- #
# Pass 2: footprints -> crops
# --------------------------------------------------------------------------- #

def find_footprints(maps: dict) -> tuple[np.ndarray, int]:
    count, top, surf, ground = maps["count"], maps["top"], maps["surf"], maps["ground"]
    reaches_surface = top >= surf - 2              # not a mineshaft or a buried vault
    build = reaches_surface & ((count >= 2) | ((count >= 1) & (top - ground >= 3)))
    build = ndimage.binary_closing(build, np.ones((3, 3)))   # doorways, gaps between posts
    labels, n = ndimage.label(build, np.ones((3, 3)))
    return labels, n


def describe(blocks: np.ndarray, palette: list[str], mask: np.ndarray, k: int = 6) -> list:
    ids, counts = np.unique(blocks[mask], return_counts=True)
    order = np.argsort(-counts)[:k]
    names = [palette[ids[i]].split("[")[0].removeprefix("minecraft:") for i in order]
    return [[nm, int(counts[i])] for nm, i in zip(names, order)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("world", type=Path)
    ap.add_argument("--name", default=None, help="short id (default: folder name)")
    ap.add_argument("--license", default="unknown", help="the map's terms, recorded per build")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--max-side", type=int, default=96, help="skip footprints wider than this")
    args = ap.parse_args()


    name = args.name or args.world.name
    out_dir = OUT / name
    if out_dir.exists():                    # rerun: replace this world's outputs
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    info = read_level_dat(args.world / "level.dat")
    print(f"{name}: Minecraft {info['mc_version']}, generator {info['generator']}")
    world = World(args.world)
    rows, k = [], 0
    groups = world.region_groups()
    for gi, group in enumerate(groups):
        print(f"  region group {gi + 1}/{len(groups)}: {len(group)} regions")
        maps, present, (ox, oz), structures = scan_world(world, group)
        labels, n = find_footprints(maps)
        print(f"  {n} footprints")

        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            k += 1
            comp = labels[sl] == i
            area = int(comp.sum())
            zs, xs = sl
            sz, sx = zs.stop - zs.start, xs.stop - xs.start
            gnd = maps["ground"][sl][comp]
            surf = maps["surf"][sl][comp]
            # Ground under the lowest built block, unless that lies deep below the surface
            # (a mineshaft or cellar under the build); then the surface.
            gnd = np.where(surf - gnd <= 16, gnd, surf)
            top = maps["top"][sl][comp]
            height = int(top.max() - np.median(gnd))
            if area < MIN_AREA or height < MIN_HEIGHT:
                continue
            uid = f"{name}_{k:05d}"
            # Crop box in world coordinates (exclusive upper bounds).
            x0, x1 = ox + xs.start - MARGIN, ox + xs.stop + MARGIN
            z0, z1 = oz + zs.start - MARGIN, oz + zs.stop + MARGIN
            y0 = Y_MIN + int(np.median(gnd)) - 6     # not min: mineshafts under a build
            y1 = Y_MIN + int(top.max()) + 3
            base = {"uid": uid, "source": f"world:{name}", "license": args.license,
                    "url": args.source_url, "mc_version": info["mc_version"], "origin": [x0, y0, z0],
                    "footprint_xz": [sx, sz], "footprint_area": area}
            if max(sx, sz) > args.max_side:
                rows.append({**base, "status": "ok", "keep": False, "reject": ["too_large"]})
                continue
            blocks, palette, missing = cut_box(world, x0, y0, z0, x1, y1, z1)
            world.clear()
            # The build = BUILT and EDIT blocks over its footprint (dilated by 1 for eaves).
            cols = np.zeros(blocks.shape[1:], dtype=bool)
            cols[MARGIN:MARGIN + sz, MARGIN:MARGIN + sx] = comp
            cols = ndimage.binary_dilation(cols, np.ones((3, 3)))
            cls = classify(blocks, palette)
            mask = ((cls == BUILT) | (cls == EDIT)) & cols[None]
            if mask.sum() < MIN_AREA:
                continue
            ids, counts = np.unique(blocks[mask], return_counts=True)
            all_mats: dict[str, int] = {}
            for b, c in zip(ids, counts):
                nm = palette[b].split("[")[0].removeprefix("minecraft:")
                all_mats[nm] = all_mats.get(nm, 0) + int(c)
            junk = junk_reason(all_mats) or worldgen_overlap(mask, (x0, y0, z0), structures)
            ys = np.nonzero(mask.any((1, 2)))[0]
            bx = np.nonzero(mask.any((0, 1)))[0]
            bz = np.nonzero(mask.any((0, 2)))[0]
            row = {**base, "caption": "", "status": "ok", "keep": junk is None,
                   "reject": [junk] if junk else [], "shape": list(blocks.shape),
                   "build_extent_yzx": [int(np.ptp(ys) + 1), int(np.ptp(bz) + 1), int(np.ptp(bx) + 1)],
                   "n_build_blocks": int(mask.sum()), "missing_chunks": missing,
                   "materials": describe(blocks, palette, mask)}
            rows.append(row)
            if junk is None:
                np.savez_compressed(out_dir / f"{uid}.npz", blocks=blocks, palette=np.array(palette),
                                    mask=np.packbits(mask, axis=None), mask_shape=np.array(mask.shape))
            if len(rows) % 20 == 0:
                print(f"  {len(rows)} footprints cut", flush=True)

    manifest = OUT / "manifest.jsonl"           # rerunning a world replaces its rows
    old = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    old = [l for l in old if l and json.loads(l)["source"] != f"world:{name}"]
    manifest.write_text("".join(l + "\n" for l in old + [json.dumps(r) for r in rows]),
                        encoding="utf-8")
    kept = [r for r in rows if r["keep"]]
    reasons = Counter(x for r in rows for x in r["reject"])
    print(f"  kept {len(kept)} builds; rejected {dict(reasons)}")


if __name__ == "__main__":
    main()
