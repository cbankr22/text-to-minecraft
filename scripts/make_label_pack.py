"""Labeling packs for captioning builds with a chat vision model (e.g. ChatGPT).

    .venv/Scripts/python scripts/make_label_pack.py --pilot 10

For each build: one 1024x1024 image with four textured views (one from each corner, id
stamped top-left) and one line of facts computed from the blocks. prompt.txt holds the
instruction plus the fact lines; paste it together with the images. The model answers with
one JSON object per build, which import_labels.py (next step) merges into the manifest.
Output: data/labeling/<pack>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_build_gallery import load_build  # noqa: E402
from ttmc.textured import Textures, render  # noqa: E402

BUILDS = ROOT / "data" / "worlds" / "builds"
OUT = ROOT / "data" / "labeling"

WORLD_NOTES = {
    "celestia_town": "map 'Celestia Town': a small medieval town of sandstone houses in a valley by the sea",
    "fantasy_medieval_city": "map 'Fantasy Medieval City': a large fantasy-medieval map with castles, "
                             "period houses, magic towers and airships",
    "medieval_port_city": "map 'Medieval Port City': medieval port towns with a lighthouse, a church, "
                          "a sorcerer's tower, farms and a piglin outpost",
}

INSTRUCTION = """You are labeling Minecraft builds for a dataset that teaches a model to build from text prompts.
Each image shows ONE build, cut out of a larger world, from its four corners (the id is in the top-left).
The build may include bits of path or wall where it touched its surroundings; ignore those.
The fact line for each id was computed from the actual blocks, so trust its sizes and materials.

For every id, answer with one JSON object per line (JSON Lines, nothing else), with these keys:
  "id": the id
  "type": what it is, 1-3 words (e.g. "house", "church", "watchtower", "market stall", "windmill",
          "bridge", "wall section", "farm", "ship", "statue"); "unclear" if you can't tell
  "style": e.g. "medieval", "fantasy", "rustic", "Mediterranean", "Japanese", "modern"
  "features": list of 3-6 short visible features (e.g. "red brick roof", "timber frame", "chimney",
              "balcony", "arched windows", "stone foundation")
  "captions": object with four ways a player might ask for this build:
      "short": 2-6 words ("a medieval house")
      "descriptive": one sentence describing shape, materials and colors
      "casual": how a player would type it in chat ("small cozy cottage w/ a dark roof")
      "spoken": how a player would say it out loud ("build me a little stone house with a red roof")
  "confidence": "high", "medium" or "low"

Do not invent materials or sizes that contradict the fact line.

Fact lines:
"""


def facts(row: dict, blocks: np.ndarray, palette: list[str], mask: np.ndarray) -> str:
    base = [p.split("[")[0].removeprefix("minecraft:") for p in palette]
    ys, zs, xs = np.nonzero(mask)
    X, Z, Y = np.ptp(xs) + 1, np.ptp(zs) + 1, np.ptp(ys) + 1
    names = Counter(base[b] for b in blocks[mask])
    # Roof: highest build block per column; base: the lowest layer.
    col_top = np.where(mask.any(0), mask.shape[0] - 1 - np.argmax(mask[::-1], 0), -1)
    zz, xx = np.nonzero(col_top >= 0)
    roof = Counter(base[blocks[col_top[z, x], z, x]] for z, x in zip(zz, xx))
    low = Counter(base[b] for b in blocks[ys.min()][mask[ys.min()]])
    glass = sum(n for k, n in names.items() if "glass" in k)
    doors = sum(n for k, n in names.items() if k.endswith("_door")) // 2
    lights = sum(n for k, n in names.items() if any(s in k for s in ("lantern", "torch", "lamp")))
    top = lambda c, k: ", ".join(f"{m} ({n})" for m, n in c.most_common(k))  # noqa: E731
    world = row["source"].removeprefix("world:")
    return (f"{row['uid']}: {X} wide x {Z} deep x {Y} tall, {int(mask.sum())} blocks; "
            f"most used: {top(names, 5)}; roof/top surface: {top(roof, 3)}; "
            f"bottom layer: {top(low, 2)}; glass blocks: {glass}; doors: {doors}; "
            f"light sources: {lights}; from {WORLD_NOTES.get(world, world)}")


def grid_image(uid: str, blocks, palette, mask, textures: Textures, view: int = 512) -> Image.Image:
    sheet = Image.new("RGB", (2 * view, 2 * view), (232, 232, 228))
    for t in range(4):
        sheet.paste(render(blocks, palette, mask, textures, turns=t, size=view),
                    ((t % 2) * view, (t // 2) * view))
    d = ImageDraw.Draw(sheet)
    d.line([(view, 0), (view, 2 * view)], fill=(190, 190, 186), width=2)
    d.line([(0, view), (2 * view, view)], fill=(190, 190, 186), width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 8), uid, fill=(30, 30, 30), font=font)
    return sheet


def pick_pilot(rows: list[dict], n: int) -> list[dict]:
    """n builds spread over sizes, alternating between worlds, skipping tiny ones."""
    by_world: dict[str, list] = {}
    for r in rows:
        if r["n_build_blocks"] >= 100 and r["build_extent_yzx"][0] >= 5:
            by_world.setdefault(r["source"], []).append(r)
    picks = []
    worlds = sorted(by_world)
    per = -(-n // len(worlds))
    for w in worlds:
        rs = sorted(by_world[w], key=lambda r: r["n_build_blocks"])
        idx = np.linspace(0, len(rs) - 1, per).round().astype(int)
        picks += [rs[i] for i in idx]
    return picks[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=10, help="number of builds")
    ap.add_argument("--all", action="store_true", help="every kept build, not a sample")
    ap.add_argument("--per-prompt", type=int, default=10, help="builds per prompt file")
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--world", default=None, help="only builds from this world")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (BUILDS / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rows = [r for r in rows if r.get("keep")
            and (args.world is None or r["source"] == f"world:{args.world}")]
    picks = sorted(rows, key=lambda r: r["uid"]) if args.all else pick_pilot(rows, args.pilot)
    out = OUT / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    textures = Textures()
    lines = []
    for i, r in enumerate(picks):
        world = r["source"].removeprefix("world:")
        blocks, palette, mask = load_build(BUILDS / world / f"{r['uid']}.npz")
        grid_image(r["uid"], blocks, palette, mask, textures).save(out / f"{r['uid']}.png")
        lines.append(facts(r, blocks, palette, mask))
        print(f"  {i + 1}/{len(picks)} {r['uid']}  {r['build_extent_yzx']}  {r['n_build_blocks']} blocks",
              flush=True)
    k = args.per_prompt
    batches = [lines[i:i + k] for i in range(0, len(lines), k)]
    for b, batch in enumerate(batches, start=1):
        name = "prompt.txt" if len(batches) == 1 else f"prompt_{b:02d}.txt"
        (out / name).write_text(INSTRUCTION + "\n".join(batch) + "\n", encoding="utf-8")
    (out / "ids.json").write_text(json.dumps([r["uid"] for r in picks], indent=1))
    print(f"pack: {out}  ({len(picks)} images, {len(batches)} prompt file(s) of up to {k} builds)")


if __name__ == "__main__":
    main()
