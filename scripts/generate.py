"""Generate Minecraft builds from text.

    .venv/Scripts/python scripts/generate.py "a red mushroom house" "a wooden chair"
    .venv/Scripts/python scripts/generate.py --gallery          # built-in prompt list
    .venv/Scripts/python scripts/generate.py "a coffee mug, 8 blocks tall" "a small house"

For each prompt and seed writes, under data/generated/<run>/:
    <slug>_s<seed>.png     isometric render (two sides)
    <slug>_s<seed>.schem   Sponge schematic (WorldEdit: //schem load <name>, Litematica)
plus gallery.png (contact sheet) and gallery.html (interactive 3D viewer).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from make_viewer import PAGE  # noqa: E402  (scripts/ is on sys.path when run directly)

from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.resize import parse_size, place, resize  # noqa: E402
from ttmc.schem import write_schem  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

GALLERY = [
    "a red mushroom house", "a small wooden cabin", "a medieval stone castle with towers",
    "a green tree with a brown trunk", "a pine tree", "a palm tree",
    "a wooden chair", "a wooden table", "a bed with a red blanket",
    "a blue sports car", "a yellow school bus", "a pirate ship",
    "a white snowman", "a yellow duck", "a pink pig",
    "a gray robot", "a knight in armor", "a red dragon",
    "a sword", "a treasure chest", "a wooden barrel",
    "a lighthouse", "a windmill", "a rocket ship",
    "a pink flower in a pot", "a cactus", "a pumpkin",
    "a hamburger", "a coffee mug", "a birthday cake",
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--gallery", action="store_true")
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "runs" / "t2v_big" / "best.pt"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prompts = args.prompts or (GALLERY if args.gallery else [])
    if not prompts:
        sys.exit("give prompts or --gallery")

    ck = torch.load(args.ckpt, map_location="cuda")
    cfg = ck["config"]
    model = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).cuda().eval()
    model.load_state_dict(ck["model"])
    tok = load_tokenizer(cfg["tokenizer"])
    text = TextEncoder()
    names, vrgb = block_vocab()

    out = Path(args.out) if args.out else ROOT / "data" / "generated" / time.strftime("%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    parsed = [parse_size(p) for p in prompts]
    builds = []                                   # (prompt, seed, cropped and resized blocks)
    for seed in range(args.seeds):
        for i in range(0, len(prompts), 16):
            chunk = parsed[i:i + 16]
            codes = generate(model, text, [c for c, _ in chunk], steps=args.steps, cfg=args.cfg,
                             seed=seed * 1000 + i)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                blocks = tok.decode_indices(codes.view(-1, 8, 8, 8)).argmax(1).cpu().numpy().astype(np.uint8)
            builds += [(p, seed, resize(b, size)) for p, (_, size), b in zip(prompts[i:i + 16], chunk, blocks)]
    res = max([32] + [max(b.shape) for *_, b in builds])
    tiles, models = [], []
    for p, seed, small in builds:
        name = f"{slug(p)}_s{seed}"
        b = place(small, res)
        img = [render_iso(vrgb[b].astype(np.uint8), b > 0, 256),
               render_iso(vrgb[b].astype(np.uint8), b > 0, 256, turns=2)]
        contact_sheet([(img, p)], cols=1).save(out / f"{name}.png")
        if (b > 0).any():
            write_schem(small, names, out / f"{name}.schem")
        tiles.append((img, f"{p}  (seed {seed})"))
        occ = b > 0
        models.append({"caption": f"{p}  (seed {seed})", "license": "generated",
                       "n_voxels": int(occ.sum()),
                       "n_block_types": int(len(np.unique(b[occ]))),
                       "blocks": base64.b64encode(gzip.compress(b.tobytes(), 9)).decode(),
                       "rgb": base64.b64encode(gzip.compress(
                           np.where(occ[..., None], vrgb[b], 0).astype(np.uint8).tobytes(), 9)).decode()})
    contact_sheet(tiles, cols=4).save(out / "gallery.png")
    blocks_js = [{"name": n, "rgb": c.astype(int).tolist()} for n, c in zip(names, vrgb)]
    html = (PAGE.replace("__MODELS__", json.dumps(models)).replace("__BLOCKS__", json.dumps(blocks_js))
            .replace("__RES__", str(res)).replace("<title>Voxel Viewer</title>", "<title>Generated Builds</title>"))
    (out / "gallery.html").write_text(html, encoding="utf-8")
    print(f"{len(tiles)} builds -> {out}  (step {ck['step']})")


if __name__ == "__main__":
    main()
