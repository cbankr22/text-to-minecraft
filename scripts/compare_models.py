"""Same prompts and seeds through several t2v checkpoints, side by side.

    .venv/Scripts/python scripts/compare_models.py data/runs/t2v_big/best.pt data/runs/t2v_big_cool/best.pt \
        --labels "before cooldown" "after cooldown" --out data/runs/compare_cooldown.png

Each checkpoint is decoded with the tokenizer named in its own config, so models built on
different tokenizers (e.g. the old 55M model) can be compared too. Flow-matching checkpoints
(config kind "flow") are sampled from their EMA weights; MaskGIT ones use the settings that
looked best in the 2026-10-08 grid (32 steps, guidance 7, temperature 0.8, choice temp 2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.flow import FlowT2V, latent_to_codes, sample  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

PROMPTS = ["a wooden chair", "a green tree with a brown trunk", "a small house with a red roof",
           "a medieval stone castle with towers", "a red sports car", "a sailing ship",
           "a turtle", "a dragon", "a lamp", "a robot", "a pumpkin", "a coffee mug"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "compare.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    labels = args.labels or [Path(c).parent.name + "/" + Path(c).stem for c in args.ckpts]
    _, vrgb = block_vocab()
    text = TextEncoder()
    builds = []
    for c in args.ckpts:
        ck = torch.load(c, map_location="cuda")
        cfg = ck["config"]
        tok = load_tokenizer(cfg["tokenizer"])
        flow = cfg.get("kind") == "flow"
        if flow:
            model = FlowT2V(latent_dim=cfg["latent_dim"], d=cfg["d"], layers=cfg["layers"],
                            heads=cfg["heads"]).cuda().eval()
            model.load_state_dict(ck["ema"])
        else:
            model = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).cuda().eval()
            model.load_state_dict(ck["model"])
        g = round(model.seq ** (1 / 3))
        out = []
        for i in range(0, len(PROMPTS), 4):             # small batches: fits next to an 8 GB desktop
            if flow:
                z = sample(model, text, PROMPTS[i:i + 4], steps=40, cfg=5.0, seed=args.seed)
                codes = latent_to_codes(z, tok.fsq)
            else:
                codes = generate(model, text, PROMPTS[i:i + 4], steps=32, cfg=7.0, temperature=0.8,
                                 choice_temp=2.0, seed=args.seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(tok.decode_indices(codes.view(-1, g, g, g)).argmax(1).cpu().numpy())
        builds.append(np.concatenate(out))
        print(f"{c}: step {ck['step']}", flush=True)
        del model, tok, ck
        torch.cuda.empty_cache()
    tiles = []
    for i, p in enumerate(PROMPTS):
        for label, b in zip(labels, builds):
            x = b[i]
            tiles.append(([render_iso(vrgb[x].astype(np.uint8), x > 0, 220)], f"{p}  [{label}]"))
    contact_sheet(tiles, cols=len(args.ckpts)).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
