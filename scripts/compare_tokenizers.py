"""Score tokenizers on the same held-out objects, to decide which one the transformer uses.

    python scripts/compare_tokenizers.py data/runs/vqvae32/last.pt data/runs/vqvae32b/last.pt \
        --val data/runs/vqvae32b/val_uids.json

Prints occupancy IoU and block accuracy (the share of voxels whose block is reconstructed
exactly) for each, over up to --n validation objects, and writes data/runs/tokenizer_compare.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.dataset import VoxelDataset, load_manifest  # noqa: E402
from ttmc.vqvae import load_tokenizer, recon_metrics  # noqa: E402

DATA = ROOT / "data" / "objaverse"


@torch.no_grad()
def score(path: str, loader) -> dict:
    tok = load_tokenizer(path)
    tot, n = {"occ_iou": 0.0, "block_acc": 0.0}, 0
    for batch in loader:
        x = batch["blocks"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = tok(x)
        for k, v in recon_metrics(logits, x).items():
            if k in tot:
                tot[k] += v
        n += 1
    return {k: round(v / n, 4) for k, v in tot.items()} | {"codebook": tok.fsq.codebook_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+")
    ap.add_argument("--val", required=True, help="val_uids.json to score on")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    val = set(json.loads(Path(args.val).read_text()))
    have = {p.stem for p in (DATA / "vox32").glob("*.npz")}
    rows = [r for r in load_manifest(DATA / "manifest.jsonl") if r["uid"] in val and r["uid"] in have]
    rows = sorted(rows, key=lambda r: r["uid"])[:args.n]
    loader = DataLoader(VoxelDataset(rows, DATA / "vox32"), 16, num_workers=0)
    results = {p: score(p, loader) for p in args.tokenizers}
    for p, r in results.items():
        print(f"{p}: {r}")
    (ROOT / "data" / "runs" / "tokenizer_compare.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
