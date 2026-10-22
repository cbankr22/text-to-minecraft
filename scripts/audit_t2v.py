"""Diagnose a trained t2v checkpoint: does it use the text, where is the loss, does it overfit?

    .venv/Scripts/python scripts/audit_t2v.py data/runs/t2v_big/best.pt --codes data/codes32b

Same objects, same masks for every condition (fixed seeds), orientation 0:
  caption test   loss with the true caption vs a shuffled (wrong) caption vs no caption.
                 A small gap means the model barely listens to the prompt.
  mask ratios    loss and top-1 accuracy when 25 / 50 / 75 / 100 % of tokens are hidden.
  overfit        the same numbers on training objects.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.t2v import T2V, TextEncoder  # noqa: E402


@torch.no_grad()
def losses(model, text, codes, caps, idx, ratio, captions_for, batch=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tot, acc, n = 0.0, 0.0, 0
    for b in range(0, len(idx), batch):
        pick = idx[b:b + batch]
        x = torch.from_numpy(codes[pick, 0].astype(np.int64)).cuda()
        B, L = x.shape
        k = max(1, math.ceil(ratio * L))
        ranks = torch.rand(B, L, device="cuda", generator=g).argsort(1).argsort(1)
        m = ranks < k
        inp = torch.where(m, model.mask_id, x)
        h, pad = text(captions_for(pick))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inp, h, pad, only=m).float()
        tgt = x[m]
        tot += F.cross_entropy(logits, tgt, reduction="sum").item()
        acc += (logits.argmax(-1) == tgt).float().sum().item()
        n += tgt.numel()
    return tot / n, acc / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--codes", default=str(ROOT / "data" / "codes32b"))
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    meta = json.loads((Path(args.codes) / "meta.json").read_text())
    codes = np.load(Path(args.codes) / "codes.npy", mmap_mode="r")
    caps = meta["captions"]
    val = np.array(meta["val"])
    rng = np.random.default_rng(0)
    va = np.sort(rng.choice(np.nonzero(val)[0], args.n, replace=False))
    tr = np.sort(rng.choice(np.nonzero(~val)[0], args.n, replace=False))
    shuffled = {int(i): caps[int(j)] for i, j in zip(va, rng.permutation(va))}

    ck = torch.load(args.ckpt, map_location="cuda")
    cfg = ck["config"]
    model = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).cuda().eval()
    model.load_state_dict(ck["model"])
    text = TextEncoder()
    print(f"{args.ckpt}: step {ck['step']}, codebook {cfg['codebook']} "
          f"(uniform guess = {math.log(cfg['codebook']):.2f})\n")

    true = lambda p: [caps[i] for i in p]  # noqa: E731
    wrong = lambda p: [shuffled[int(i)] for i in p]  # noqa: E731
    empty = lambda p: [""] * len(p)  # noqa: E731
    print("caption test (100% masked = generating from scratch; 50% masked)")
    for ratio in (1.0, 0.5):
        r = {name: losses(model, text, codes, caps, va, ratio, fn)
             for name, fn in (("true", true), ("shuffled", wrong), ("empty", empty))}
        print(f"  {int(ratio * 100):>3}% masked  " + "  ".join(
            f"{k}: loss {v[0]:.3f} acc {v[1]:.1%}" for k, v in r.items()))
    print("\nloss by mask ratio (true captions)      val                  train")
    for ratio in (0.25, 0.5, 0.75, 1.0):
        lv, av = losses(model, text, codes, caps, va, ratio, true)
        lt, at = losses(model, text, codes, caps, tr, ratio, true)
        print(f"  {int(ratio * 100):>3}% masked   loss {lv:.3f} acc {av:.1%}    loss {lt:.3f} acc {at:.1%}")


if __name__ == "__main__":
    main()
