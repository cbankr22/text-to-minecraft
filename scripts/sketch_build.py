"""Sketch to build: a rough gray-wool shape + a prompt -> a detailed build with that shape.

    .venv/Scripts/python scripts/sketch_build.py                       # built-in test sketches
    .venv/Scripts/python scripts/sketch_build.py --strengths 0.6 0.75 0.9

The sketch is encoded with the tokenizer into the flow model's latent, pushed part of the way
to noise (strength: 0 = unchanged, 1 = pure noise, i.e. ordinary text-to-build), then denoised
with the prompt (SDEdit; ttmc/flow.py sample(init=..., strength=...)). No retraining.
Writes data/runs/sketch/sketch.png: per row the sketch, then one build per strength.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.flow import FlowT2V, latent_to_codes, sample  # noqa: E402
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.t2v import TextEncoder  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

WOOL = 4            # minecraft:gray_wool (token = blocks.json index + 1)
N = 32
Y, Z, X = np.ogrid[:N, :N, :N]                     # grids are (Y, Z, X), Y up


def box(g, y0, y1, z0, z1, x0, x1):
    g[y0:y1, z0:z1, x0:x1] = WOOL


def church():
    g = np.zeros((N, N, N), np.uint8)
    box(g, 0, 12, 10, 22, 2, 24)                                   # nave
    for k in range(6):                                             # pitched roof
        box(g, 12 + k, 13 + k, 10 + k, 22 - k, 2, 24)
    box(g, 0, 30, 12, 20, 23, 31)                                  # bell tower
    return g


def blob_animal():
    g = np.zeros((N, N, N), np.uint8)
    g[((Y - 14) / 6.0) ** 2 + ((Z - 16) / 6.0) ** 2 + ((X - 14) / 11.0) ** 2 <= 1] = WOOL   # body
    g[((Y - 21) / 4.5) ** 2 + ((Z - 16) / 4.0) ** 2 + ((X - 28) / 4.0) ** 2 <= 1] = WOOL    # head
    for z, x in ((12, 7), (20, 7), (12, 21), (20, 21)):                                     # legs
        box(g, 0, 11, z - 1, z + 2, x - 1, x + 2)
    return g


def tower():
    g = np.zeros((N, N, N), np.uint8)
    r = np.sqrt((Z - 16.0) ** 2 + (X - 16.0) ** 2)
    g[(Y < 24) & (r <= 6 - Y / 12.0)] = WOOL                       # slightly tapered column
    g[(Y >= 24) & (Y < 31) & (r <= (31 - Y) * 0.9)] = WOOL         # cone on top
    return g


def ship():
    g = np.zeros((N, N, N), np.uint8)
    for y in range(0, 8):                                          # hull widens upward, pointy bow
        half = 3 + y * 0.8
        for x in range(2, 30):
            w = half * min(1.0, (30 - x) / 8.0)
            g[y, int(16 - w):int(16 + w) + 1, x] = WOOL if w >= 1 else 0
    box(g, 8, 30, 15, 17, 15, 17)                                  # mast
    box(g, 12, 26, 15, 16, 9, 23)                                  # sail
    return g


def mushroom():
    g = np.zeros((N, N, N), np.uint8)
    r = np.sqrt((Z - 16.0) ** 2 + (X - 16.0) ** 2)
    g[(Y < 16) & (r <= 5)] = WOOL                                  # stalk
    g[(Y >= 14) & (((Y - 14) / 12.0) ** 2 + (r / 14.0) ** 2 <= 1)] = WOOL   # dome cap
    return g


def arch_bridge():
    g = np.zeros((N, N, N), np.uint8)
    box(g, 10, 14, 12, 20, 0, 32)                                  # deck
    for x0 in (0, 26):
        box(g, 0, 10, 12, 20, x0, x0 + 6)                          # piers
    g[(Y < 10) & (Z >= 12) & (Z < 20) & (((X - 16.0) / 10) ** 2 + (Y / 10.0) ** 2 > 1) & (X > 5) & (X < 26)] = WOOL
    return g


SKETCHES = [
    (church, "a gothic cathedral"), (church, "a japanese temple"),
    (blob_animal, "a cow"), (blob_animal, "a dragon"),
    (tower, "a lighthouse"), (tower, "a wizard tower"),
    (ship, "a pirate ship"), (mushroom, "a mushroom house"), (arch_bridge, "a stone bridge"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=str(ROOT / "data" / "runs" / "flow_big" / "best.pt"))
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.65, 0.8, 0.9])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "sketch" / "sketch.png"))
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cuda")
    cfg = ck["config"]
    model = FlowT2V(latent_dim=cfg["latent_dim"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"]).cuda().eval()
    model.load_state_dict(ck["ema"])
    tok = load_tokenizer(cfg["tokenizer"])
    text = TextEncoder()
    _, vrgb = block_vocab()

    def render(x):
        return render_iso(vrgb[x].astype(np.uint8), x > 0, 220)

    sketches = np.stack([f() for f, _ in SKETCHES])
    prompts = [p for _, p in SKETCHES]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q, _ = tok.encode(torch.from_numpy(sketches).long().cuda())             # (B, dim, 8, 8, 8)
    init = q.float().permute(0, 2, 3, 4, 1).reshape(len(q), -1, q.shape[1])    # -> (B, 512, dim)
    builds = []
    for s in args.strengths:
        z = sample(model, text, prompts, steps=args.steps, cfg=args.cfg, seed=args.seed, shift=args.shift,
                   init=init, strength=s)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            builds.append(tok.decode_indices(latent_to_codes(z, tok.fsq).view(-1, 8, 8, 8)).argmax(1).cpu().numpy())
        print(f"strength {s} done", flush=True)
    tiles = [([render(sketches[i])] + [render(b[i]) for b in builds],
              f"{p}:  sketch | " + " | ".join(f"strength {s:g}" for s in args.strengths))
             for i, p in enumerate(prompts)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(tiles, cols=1).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
