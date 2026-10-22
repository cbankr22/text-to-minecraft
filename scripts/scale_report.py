"""Compare the data-scaling runs (same model and steps, 12k / 24k / 48k training objects).

    .venv/Scripts/python scripts/scale_report.py

Reads data/runs/scale_*k/{log.jsonl,best.pt} and writes data/runs/scale_report.png:
validation loss over training (same held-out objects for every run) and, for the same
prompts and seeds, one build per run side by side. Also data/runs/scale_report.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

RUNS = ["scale_12k", "scale_24k", "scale_48k"]
PROMPTS = ["a wooden chair", "a green tree with a brown trunk", "a red sports car", "a turtle",
           "a small house with a red roof", "a sword", "a dragon", "a lamp", "a sailing ship",
           "a robot"]


def main() -> None:
    runs = ROOT / "data" / "runs"
    lines, curves = [], {}
    for r in RUNS:
        log = [json.loads(l) for l in (runs / r / "log.jsonl").read_text().splitlines() if l]
        curves[r] = ([e["step"] for e in log], [e["val_loss"] for e in log])
        lines.append(f"{r}: best val loss {min(curves[r][1]):.4f}, final {curves[r][1][-1]:.4f}")
    (runs / "scale_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    for r, (x, y) in curves.items():
        ax.plot(x, y, label=r.replace("scale_", "") + " objects")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title("Same model, same steps, more training objects")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(runs / "scale_curves.png")

    _, vrgb = block_vocab()
    text = TextEncoder()
    tiles = []
    builds = {}
    for r in RUNS:
        ck = torch.load(runs / r / "best.pt", map_location="cuda")
        cfg = ck["config"]
        model = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).cuda().eval()
        model.load_state_dict(ck["model"])
        tok = load_tokenizer(cfg["tokenizer"])
        codes = generate(model, text, PROMPTS, seed=0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            builds[r] = tok.decode_indices(codes.view(-1, 8, 8, 8)).argmax(1).cpu().numpy()
        del model, ck
        torch.cuda.empty_cache()
    for i, p in enumerate(PROMPTS):
        for r in RUNS:
            b = builds[r][i]
            tiles.append(([render_iso(vrgb[b].astype(np.uint8), b > 0, 220)],
                          f"{p}  [{r.replace('scale_', '')}]"))
    contact_sheet(tiles, cols=len(RUNS)).save(runs / "scale_report.png")
    print(f"wrote {runs / 'scale_report.png'} and scale_curves.png")


if __name__ == "__main__":
    main()
