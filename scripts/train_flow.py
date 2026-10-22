"""Train the flow-matching text -> voxel-latent model (ttmc/flow.py).

    .venv/Scripts/python scripts/train_flow.py --codes data/codes32b --hours 6 --out data/runs/flow_small
    python scripts/train_flow.py --codes data/codes32b --d 768 --layers 16 --batch 64 --hours 30 \
        --auto-resume --stop-file data/STOP --out data/runs/flow_big          # rented GPU

Uses the codes from scripts/encode_codes.py (turned back into the tokenizer's continuous FSQ
latent), so the tokenizer and data stay the same as for the MaskGIT model. Lessons from the
2026-10-07 run are built in: warmup + cosine decay over --hours (time-based) or --steps, a
divergence guard (val loss jumps -> reload best.pt, halve LR), an EMA copy of the weights
(what gets sampled and exported), last.pt every --save-minutes and atomic writes.

Writes to --out: last.pt, best.pt (lowest val loss of the EMA weights), log.jsonl,
samples_<step>.png (EMA model, preview prompts, two seeds).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.flow import FlowT2V, codes_to_latent, latent_to_codes, sample  # noqa: E402
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.t2v import TextEncoder  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

PREVIEW_PROMPTS = [
    "a wooden chair", "a green tree with a brown trunk", "a small house with a red roof",
    "a medieval stone castle with towers", "a red sports car", "a sailing ship",
    "a turtle", "a dragon", "a lamp", "a robot", "a pumpkin", "a coffee mug",
]


@torch.no_grad()
def preview(model, text, tok, path: Path, steps: int, cfg: float) -> None:
    _, vrgb = block_vocab()
    model.eval()
    tiles = []
    builds = []
    for seed in (0, 1):
        z = sample(model, text, PREVIEW_PROMPTS, steps=steps, cfg=cfg, seed=seed)
        idx = latent_to_codes(z, tok.fsq).view(-1, 8, 8, 8)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            builds.append(tok.decode_indices(idx).argmax(1).cpu().numpy())
    for i, p in enumerate(PREVIEW_PROMPTS):
        imgs = [render_iso(vrgb[b[i]].astype(np.uint8), b[i] > 0, 220) for b in builds]
        tiles.append((imgs, p))
    contact_sheet(tiles, cols=3).save(path)
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=str(ROOT / "data" / "codes32b"))
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "flow"))
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--hours", type=float, default=0, help="time budget; the LR schedule follows it")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--short-prob", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=10000)
    ap.add_argument("--sample-steps", type=int, default=40)
    ap.add_argument("--sample-cfg", type=float, default=5.0)
    ap.add_argument("--save-minutes", type=float, default=15)
    ap.add_argument("--diverge", type=float, default=0.5, help="relative val-loss jump that triggers rollback")
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--best-weights-only", action="store_true",
                    help="best.pt holds only the EMA weights (~1/4 the size): for a nearly full disk")
    ap.add_argument("--stop-file", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.loads((Path(args.codes) / "meta.json").read_text())
    codes = np.load(Path(args.codes) / "codes.npy", mmap_mode="r")
    caps = meta["captions"]
    short = meta.get("short") or [[] for _ in caps]
    val = np.array(meta["val"])
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(f"objects train {len(tr_idx)} val {len(va_idx)}", flush=True)

    tok = load_tokenizer(meta["tokenizer"])
    tok.requires_grad_(False)
    text = TextEncoder(dev)
    dim = len(tok.fsq.levels)
    model = FlowT2V(latent_dim=dim, d=args.d, layers=args.layers, heads=args.heads).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    config = {"kind": "flow", "latent_dim": dim, "d": args.d, "layers": args.layers,
              "heads": args.heads, "tokenizer": meta["tokenizer"]}
    step, best, t_used = 0, float("inf"), 0.0
    if args.auto_resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, best, t_used = ck["step"], ck["best"], ck.get("hours_used", 0.0)
        del ck
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  start step {step}  "
          f"hours used {t_used:.2f}", flush=True)
    guard = [1.0]

    def lr_now(hours_done: float) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup * guard[0]
        frac = hours_done / args.hours if args.hours else step / args.steps
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))) * guard[0]

    def batch_from(idx: np.ndarray, bs: int, rng=None):
        rng = rng or np.random
        pick = rng.choice(idx, bs)
        orient = rng.randint(0, 8, bs)
        x = torch.from_numpy(codes[pick, orient].astype(np.int64)).to(dev, non_blocking=True)

        def caption(i: int) -> str:
            if random.random() < args.cond_drop:
                return ""
            if short[i] and random.random() < args.short_prob:
                return random.choice(short[i])
            return caps[i]
        return codes_to_latent(x, tok.fsq), [caption(i) for i in pick]

    def flow_loss(net, x0, texts, t=None, noise=None):
        if t is None:
            t = torch.sigmoid(torch.randn(len(x0), device=dev))       # logit-normal timesteps
        if noise is None:
            noise = torch.randn_like(x0)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise
        h, pad = text(texts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(xt, t, h, pad)
        return F.mse_loss(v.float(), noise - x0)

    @torch.no_grad()
    def evaluate(net, n: int = 320) -> float:
        """Same objects, timesteps and noise every time, so evals are comparable."""
        net.eval()
        rng = np.random.RandomState(0)
        g = torch.Generator(device=dev).manual_seed(0)
        losses = []
        for _ in range(-(-n // args.batch)):
            pick = rng.choice(va_idx, args.batch)
            x = torch.from_numpy(codes[pick, 0].astype(np.int64)).to(dev)
            x0 = codes_to_latent(x, tok.fsq)
            t = torch.rand(len(x0), device=dev, generator=g) * 0.9 + 0.05
            noise = torch.randn(x0.shape, device=dev, generator=g)
            losses.append(flow_loss(net, x0, [caps[i] for i in pick], t, noise).item())
        net.train()
        return float(np.mean(losses))

    def save(name: str, hours_done: float) -> None:
        tmp = out / (name + ".tmp")
        state = {"ema": ema.state_dict(), "step": step, "best": best, "hours_used": hours_done,
                 "config": config}
        if not (name == "best.pt" and args.best_weights_only):
            state.update(model=model.state_dict(), opt=opt.state_dict())
        torch.save(state, tmp)
        os.replace(tmp, out / name)

    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    stop_file = Path(args.stop_file) if args.stop_file else None
    log = open(out / "log.jsonl", "a")
    t_start = time.time() - t_used * 3600
    hours = lambda: (time.time() - t_start) / 3600  # noqa: E731
    last_save = t0 = time.time()
    running = []
    model.train()
    while step < args.steps:
        for gp in opt.param_groups:
            gp["lr"] = lr_now(hours())
        x0, texts = batch_from(tr_idx, args.batch)
        try:
            loss = flow_loss(model, x0, texts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            del x0, texts
            loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if args.batch <= 4:
                raise
            args.batch //= 2
            print(f"OOM: batch size reduced to {args.batch}", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))                 # warm the EMA up early on
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        running.append(loss.item())
        if step % 100 == 0:
            print(f"step {step}  loss {np.mean(running):.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{100 * args.batch / (time.time() - t0):.0f} samples/s  "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            running, t0 = [], time.time()
        stop = ((args.hours and hours() >= args.hours) or interrupted[0] or step >= args.steps
                or (stop_file is not None and stop_file.exists()))
        if step % args.eval_every == 0 or stop:
            vl = evaluate(ema)
            rec = {"step": step, "val_loss": round(vl, 5), "hours": round(hours(), 2),
                   "lr": opt.param_groups[0]["lr"]}
            print("eval", rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if vl > best * (1 + args.diverge) and (out / "best.pt").exists():
                ck = torch.load(out / "best.pt", map_location=dev)
                model.load_state_dict(ck.get("model", ck["ema"]))   # weights-only best: restart from EMA
                ema.load_state_dict(ck["ema"])
                if "opt" in ck:
                    opt.load_state_dict(ck["opt"])
                del ck
                guard[0] *= 0.5
                msg = {"step": step, "diverged": round(vl, 5), "rolled_back_to": round(best, 5),
                       "lr_factor": guard[0]}
                print("GUARD", msg, flush=True)
                log.write(json.dumps(msg) + "\n")
                log.flush()
            else:
                if vl < best:
                    best = vl
                    save("best.pt", hours())
                save("last.pt", hours())
            last_save = t0 = time.time()
        elif time.time() - last_save > args.save_minutes * 60:
            save("last.pt", hours())
            last_save = time.time()
        if step % args.sample_every == 0 or (stop and not interrupted[0]):
            preview(ema, text, tok, out / f"samples_{step:06d}.png", args.sample_steps, args.sample_cfg)
            t0 = time.time()
        if stop:
            print(f"stopping at step {step}", flush=True)
            break


if __name__ == "__main__":
    main()
