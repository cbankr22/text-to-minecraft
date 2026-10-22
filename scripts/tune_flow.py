"""Sampling settings, speck cleanup and best-of-N for a flow checkpoint -> comparison sheets.

    .venv/Scripts/python scripts/tune_flow.py data/runs/flow_big/best.pt --out data/runs/tune

1. Grid: every (steps, cfg, shift) setting generates the test prompts x 3 seeds; each build is
   rendered and scored by CLIP (cosine similarity of the render's image embedding and the
   prompt's text embedding, the usual "CLIP score"). grid.txt ranks the settings; grid.png
   shows the old default (40 steps, cfg 5, shift 1) next to the top 3 on the preview prompts.
2. cleanup.png: builds with the best setting, before / after ttmc.cleanup.remove_specks.
3. best_of_4.png: 4 seeds per preview prompt with their CLIP scores, and best_vs_random.png:
   seed 0 next to the highest-scoring of the 4 (what a "best of 4" mode would place).
CLIP ViT-B/32 judging a voxel render is rough; the sheets are there to check it by eye.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.cleanup import remove_specks  # noqa: E402
from ttmc.flow import FlowT2V, latent_to_codes, sample  # noqa: E402
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.t2v import CLIP_NAME, TextEncoder  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

PREVIEW = ["a wooden chair", "a green tree with a brown trunk", "a small house with a red roof",
           "a medieval stone castle with towers", "a red sports car", "a sailing ship",
           "a turtle", "a dragon", "a lamp", "a robot", "a pumpkin", "a coffee mug"]
EXTRA = ["a red apple", "a wooden barrel", "a sword", "a blue bicycle", "a cat", "a lighthouse",
         "a palm tree", "a treasure chest", "a guitar", "a snowman", "a windmill", "a rocket"]
TEST = PREVIEW + EXTRA
STEPS, CFGS, SHIFTS = (20, 40, 80), (3.0, 5.0, 7.5, 10.0), (1.0, 3.0)
DEFAULT = (40, 5.0, 1.0)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class Judge:
    """CLIP score of rendered builds against their prompts."""

    def __init__(self):
        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(CLIP_NAME).cuda().eval()
        self.tok = CLIPTokenizer.from_pretrained(CLIP_NAME)

    @staticmethod
    def _emb(out) -> torch.Tensor:
        # transformers 5 returns an output object whose pooler_output is the projected embedding
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def __call__(self, images, prompts) -> np.ndarray:
        px = torch.stack([(torch.from_numpy(np.asarray(im.convert("RGB").resize((224, 224), 3))).permute(2, 0, 1)
                           .float() / 255 - CLIP_MEAN) / CLIP_STD for im in images]).cuda()
        t = self.tok(prompts, padding=True, return_tensors="pt").to("cuda")
        a = F.normalize(self._emb(self.model.get_image_features(pixel_values=px)), dim=-1)
        b = F.normalize(self._emb(self.model.get_text_features(**t)), dim=-1)
        return (a * b).sum(-1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=str(ROOT / "data" / "runs" / "flow_big" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "tune"))
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    ap.add_argument("--cfgs", type=float, nargs="+", default=list(CFGS))
    ap.add_argument("--shifts", type=float, nargs="+", default=list(SHIFTS))
    ap.add_argument("--seeds", type=int, default=3, help="seeds per prompt for each grid setting")
    ap.add_argument("--merge-log", default="", help="add the grid lines of an earlier run's log")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cuda")
    cfg = ck["config"]
    model = FlowT2V(latent_dim=cfg["latent_dim"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"]).cuda().eval()
    model.load_state_dict(ck["ema"])
    del ck
    tok = load_tokenizer(cfg["tokenizer"])
    text, judge = TextEncoder(), Judge()
    _, vrgb = block_vocab()

    def build(prompts, steps, g, shift, seed):
        grids = []
        for i in range(0, len(prompts), args.batch):
            z = sample(model, text, prompts[i:i + args.batch], steps=steps, cfg=g, seed=seed, shift=shift)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                grids.append(tok.decode_indices(latent_to_codes(z, tok.fsq).view(-1, 8, 8, 8)).argmax(1).cpu().numpy())
        return np.concatenate(grids)

    def render(x):
        return render_iso(vrgb[x].astype(np.uint8), x > 0, 220)

    # 1. grid
    results, shown = [], {}
    if args.merge_log:
        for m in re.finditer(r"steps\s+(\d+) cfg\s+([\d.]+) shift (\d+): CLIP ([\d.]+)", Path(args.merge_log).read_text()):
            if int(m[1]) not in args.steps:                   # settings rerun here use the new numbers
                results.append({"steps": int(m[1]), "cfg": float(m[2]), "shift": float(m[3]), "clip": float(m[4]),
                            "from": args.merge_log})
    t0 = time.time()
    for steps, g, shift in itertools.product(args.steps, args.cfgs, args.shifts):
        scores = []
        for seed in range(args.seeds):
            grids = build(TEST, steps, g, shift, seed)
            ims = [render(x) for x in grids]
            scores.append(judge(ims, TEST))
            if seed == 0:
                shown[(steps, g, shift)] = ims[:len(PREVIEW)]
        s = np.stack(scores)                                  # (seeds, prompts)
        results.append({"steps": steps, "cfg": g, "shift": shift, "clip": round(float(s.mean()), 4),
                        "per_prompt": dict(zip(TEST, np.round(s.mean(0), 4).tolist()))})
        print(f"steps {steps:3d} cfg {g:4.1f} shift {shift:.0f}: CLIP {s.mean():.4f}  ({time.time() - t0:.0f} s)", flush=True)
    results.sort(key=lambda r: -r["clip"])
    (out / "grid.json").write_text(json.dumps(results, indent=1))
    lines = [f"{'rank':>4} {'steps':>5} {'cfg':>5} {'shift':>5}  CLIP score (mean over {len(TEST)} prompts x 2-3 seeds)"]
    lines += [f"{i + 1:>4} {r['steps']:>5} {r['cfg']:>5} {r['shift']:>5}  {r['clip']:.4f}"
              + ("   <- old default" if (r["steps"], r["cfg"], r["shift"]) == DEFAULT else "")
              for i, r in enumerate(results)]
    (out / "grid.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]), flush=True)
    top = [(r["steps"], r["cfg"], r["shift"]) for r in results[:3]]
    default = DEFAULT if any((r["steps"], r["cfg"], r["shift"]) == DEFAULT for r in results) else top[0]
    cols = [default] + [c for c in top if c != default][:3]
    for c in cols:
        if c not in shown:                                    # a setting from --merge-log
            shown[c] = [render(x) for x in build(PREVIEW, c[0], c[1], c[2], 0)]
    tiles =[([shown[c][i]], f"{p}  [steps {c[0]}, cfg {c[1]:g}, shift {c[2]:g}" + (", old default]" if c == DEFAULT else "]"))
             for i, p in enumerate(PREVIEW) for c in cols]
    contact_sheet(tiles, cols=len(cols)).save(out / "grid.png")

    # 2. cleanup, 3. best of 4 (best setting)
    steps, g, shift = top[0]
    seeds = [build(PREVIEW, steps, g, shift, s) for s in range(4)]   # (4 seeds) x (prompts, 32, 32, 32)
    tiles = []
    for i, p in enumerate(PREVIEW):
        for s in (0, 1):
            x = seeds[s][i]
            y = remove_specks(x)
            tiles.append(([render(x), render(y)], f"{p} (seed {s}): before | after cleanup, "
                                                   f"{int((x > 0).sum() - (y > 0).sum())} blocks removed"))
    contact_sheet(tiles, cols=2).save(out / "cleanup.png")
    tiles, versus = [], []
    for i, p in enumerate(PREVIEW):
        ims = [render(remove_specks(seeds[s][i])) for s in range(4)]
        sc = judge(ims, [p] * 4)
        b = int(sc.argmax())
        tiles.append((ims, f"{p}: CLIP " + "  ".join(f"{v:.3f}" + ("*" if k == b else "") for k, v in enumerate(sc))))
        versus.append(([ims[0], ims[b]], f"{p}: seed 0 | best of 4 (seed {b})"))
    contact_sheet(tiles, cols=1).save(out / "best_of_4.png")
    contact_sheet(versus, cols=3).save(out / "best_vs_random.png")
    print(f"done in {time.time() - t0:.0f} s; sheets in {out}", flush=True)


if __name__ == "__main__":
    main()
