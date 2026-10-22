"""Train the 3D VQ-VAE voxel tokenizer.

    .venv/Scripts/python scripts/train_vqvae.py --steps 20000
    .venv/Scripts/python scripts/train_vqvae.py --res 32 --levels 7,5,5,5,5 --channels 64,128,256         --batch 16 --out data/runs/vqvae32
    .venv/Scripts/python scripts/train_vqvae.py --resume       # continue from last.pt

Writes to data/runs/vqvae/: last.pt, log.jsonl, recon_<step>.png (val: original | reconstruction).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ttmc.dataset import VoxelDataset, load_manifest, split  # noqa: E402
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import VoxelVQVAE, recon_loss, recon_metrics  # noqa: E402

DATA = ROOT / "data" / "objaverse"


def render_recons(model, val_ds, device, path: Path, n: int = 10) -> None:
    _, vocab_rgb = block_vocab()
    model.eval()
    tiles = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(n, len(val_ds))):
            item = val_ds[i]
            logits, _ = model(item["blocks"][None].to(device))
            pred = logits.argmax(1)[0].cpu().numpy()
            orig = item["blocks"].numpy()
            views = [render_iso(vocab_rgb[g].astype(np.uint8), g > 0) for g in (orig, pred)]
            tiles.append((views, item["caption"]))
    contact_sheet(tiles, cols=2).save(path)
    model.train()


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> dict:
    model.eval()
    tot = {"loss": 0.0, "occ_iou": 0.0, "block_acc": 0.0}
    codes = set()
    n = 0
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        x = batch["blocks"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, idx = model(x)
        tot["loss"] += recon_loss(logits, x).item()
        for k, v in recon_metrics(logits, x).items():
            tot[k] += v
        codes.update(idx.unique().tolist())
        n += 1
    model.train()
    out = {k: round(v / max(n, 1), 4) for k, v in tot.items()}
    out["codes_used"] = len(codes)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "vqvae"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--res", type=int, default=64, choices=(32, 64))
    ap.add_argument("--levels", default="8,5,5,5", help="FSQ levels; product = codebook size")
    ap.add_argument("--channels", default="32,128,256")
    ap.add_argument("--hours", type=float, default=0, help="optional wall-clock limit")
    ap.add_argument("--licenses", default="", help="comma list, e.g. by,cc0 (default: all)")
    ap.add_argument("--fp32", action="store_true",
                    help="no bf16 autocast: on an RTX 4090 (torch 2.9) bf16 3D convs ran 5x slower")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    val_file = out / "val_uids.json"
    pinned = set(json.loads(val_file.read_text())) if val_file.exists() else None
    lic = set(args.licenses.split(",")) if args.licenses else None
    train_rows, val_rows = split(load_manifest(DATA / "manifest.jsonl", licenses=lic), val_uids=pinned)
    if pinned is None:
        val_file.write_text(json.dumps([r["uid"] for r in val_rows]))
    vox_dir = DATA / f"vox{args.res}"
    have = {p.stem for p in vox_dir.glob("*.npz")}
    train_rows = [r for r in train_rows if r["uid"] in have]
    val_rows = [r for r in val_rows if r["uid"] in have]
    train_ds = VoxelDataset(train_rows, vox_dir, augment=True)
    val_ds = VoxelDataset(val_rows, vox_dir)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, persistent_workers=True, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch, num_workers=0)   # keep RAM low
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    vocab = len(block_vocab()[0])
    levels = tuple(int(v) for v in args.levels.split(","))
    channels = tuple(int(v) for v in args.channels.split(","))
    config = {"vocab": vocab, "levels": levels, "channels": channels, "res": args.res}
    model = VoxelVQVAE(vocab, channels=channels, levels=levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.03)
    step = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  "
          f"codebook {model.fsq.codebook_size}  starting at step {step}")

    log = open(out / "log.jsonl", "a")
    t0, running = time.time(), []
    deadline = time.time() + args.hours * 3600 if args.hours else float("inf")
    while step < args.steps:
        for batch in train_dl:
            x = batch["blocks"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                logits, _ = model(x)
            loss = recon_loss(logits, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())

            if step % 50 == 0:
                print(f"step {step}  loss {np.mean(running):.4f}  "
                      f"{50 * args.batch / (time.time() - t0):.1f} samples/s  "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                running, t0 = [], time.time()
            if time.time() > deadline and step < args.steps:
                print(f"time limit reached at step {step}", flush=True)
                args.steps = step
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_dl, device)
                ev["step"] = step
                print("eval", ev, flush=True)
                log.write(json.dumps(ev) + "\n")
                log.flush()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step,
                            "config": config}, out / "last.pt")
                render_recons(model, val_ds, device, out / f"recon_{step:06d}.png")
                t0 = time.time()
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
