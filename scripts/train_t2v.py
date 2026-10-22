"""Train the text -> voxel-code transformer.

    .venv/Scripts/python scripts/train_t2v.py --hours 3
    .venv/Scripts/python scripts/train_t2v.py --hours 3 --resume     # continue (e.g. on more data)
    python scripts/train_t2v.py --steps 300000 --auto-resume        # interruptible cloud runs

Stops at --steps, after --hours, on SIGTERM/Ctrl-C, or as soon as --stop-file exists. Writes to
data/runs/t2v/: last.pt, best.pt (lowest val loss), log.jsonl, samples_<step>.png.

--resume starts a fresh warmup + LR schedule from last.pt (for a new phase, e.g. more data).
--auto-resume picks up exactly where last.pt left off (same schedule), or starts fresh if
there is no last.pt, so the same command can be rerun after a spot/interruptible instance
dies. last.pt is also saved every --save-minutes, and writes are atomic.
"""
from __future__ import annotations

import argparse
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
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate, mask_tokens  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

PREVIEW_PROMPTS = [
    "a red mushroom house", "a wooden chair", "a green tree with a brown trunk",
    "a medieval stone castle with towers", "a blue sports car", "a white snowman",
    "a small house with a red roof", "a yellow duck", "a wooden barrel",
    "a gray robot", "a pink flower in a pot", "a sword",
    "a turtle", "a lamp", "a ship",                       # casual one-word prompts
]


def decode_preview(t2v, text, tok, prompts, path: Path, seed: int = 0) -> None:
    _, vrgb = block_vocab()
    t2v.eval()
    codes = generate(t2v, text, prompts, seed=seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        blocks = tok.decode_indices(codes.view(-1, 8, 8, 8)).argmax(1).cpu().numpy()
    tiles = [([render_iso(vrgb[b].astype(np.uint8), b > 0, 256),
               render_iso(vrgb[b].astype(np.uint8), b > 0, 256, turns=2)], p)
             for b, p in zip(blocks, prompts)]
    contact_sheet(tiles, cols=3).save(path)
    t2v.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=str(ROOT / "data" / "codes32"))
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "t2v"))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--max-objects", type=int, default=0,
                    help="train on a fixed random subset of this many objects (data-scaling tests); "
                         "subsets are nested, so 12k is inside 24k; val is always the full val set")
    ap.add_argument("--short-prob", type=float, default=0.3,
                    help="chance of training on a short caption variant instead of the full one")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=4000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights and optimizer "
                    "(e.g. another run's best.pt), with a fresh schedule in --out")
    ap.add_argument("--cooldown", action="store_true",
                    help="with --hours: decay the LR from --lr to 5%% over the whole run (finish a run)")
    ap.add_argument("--diverge", type=float, default=0.5,
                    help="if val loss rises this far above the best, reload best.pt and halve the LR")
    ap.add_argument("--save-minutes", type=float, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    meta = json.loads((Path(args.codes) / "meta.json").read_text())
    # memory-mapped: at 700k objects codes.npy is ~5.7 GB, so rows are read per batch
    codes = np.load(Path(args.codes) / "codes.npy", mmap_mode="r")
    caps = meta["captions"]
    short = meta.get("short") or [[] for _ in caps]       # older meta.json has no variants
    val = np.array(meta["val"])
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    if args.max_objects:
        tr_idx = np.sort(np.random.default_rng(0).permutation(tr_idx)[:args.max_objects])
    K = meta["codebook"]
    print(f"objects train {len(tr_idx)} val {len(va_idx)}  codebook {K}", flush=True)

    tok = load_tokenizer(meta["tokenizer"])
    text = TextEncoder(dev)
    model = T2V(K, d=args.d, layers=args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    step, best, sched_start = 0, float("inf"), None
    if (args.resume or args.auto_resume) and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best, sched_start = ck["step"], ck.get("best", best), ck.get("sched_start")
        del ck
    elif args.init:
        ck = torch.load(args.init, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"initialized from {args.init} (step {step}, val {ck.get('best', float('nan')):.4f})", flush=True)
        del ck
    # step at which the current LR schedule (warmup + cosine) began
    start_step = sched_start if args.auto_resume and sched_start is not None else step
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  start step {step}",
          flush=True)

    def lr_at(s: int) -> float:
        s -= start_step                    # --resume: fresh warmup; --auto-resume: same schedule
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        if args.hours:                                    # time-based: hold LR, decay at the end
            return args.lr * lr_scale[0] * guard[0]
        frac = (s - args.warmup) / max(1, args.steps - start_step - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1)))) * guard[0]

    lr_scale = [1.0]
    guard = [1.0]          # halved each time the divergence guard rolls back to best.pt

    def batch_from(idx: np.ndarray, bs: int):
        pick = np.random.choice(idx, bs)
        orient = np.random.randint(0, 8, bs)
        x = torch.from_numpy(codes[pick, orient].astype(np.int64)).to(dev, non_blocking=True)
        def caption(i: int) -> str:
            if random.random() < args.cond_drop:
                return ""
            if short[i] and random.random() < args.short_prob:
                return random.choice(short[i])
            return caps[i]
        return x, [caption(i) for i in pick]

    @torch.no_grad()
    def evaluate(n_samples: int = 320) -> float:
        n_batches = -(-n_samples // args.batch)         # same ~320 objects at any batch size
        model.eval()
        g = torch.Generator().manual_seed(0)
        losses = []
        for b in range(n_batches):
            pick = va_idx[torch.randint(len(va_idx), (args.batch,), generator=g).numpy()]
            x = torch.from_numpy(codes[pick, 0].astype(np.int64)).to(dev)
            inp, m = mask_tokens(x, model.mask_id)
            h, pad = text([caps[i] for i in pick])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp, h, pad, only=m)
            losses.append(F.cross_entropy(logits.float(), x[m]).item())
        model.train()
        return float(np.mean(losses))

    def save(name: str) -> None:
        tmp = out / (name + ".tmp")                   # a kill mid-write must not corrupt last.pt
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "best": best, "sched_start": start_step,
                    "config": {"codebook": K, "d": args.d, "layers": args.layers,
                               "tokenizer": meta["tokenizer"]}}, tmp)
        os.replace(tmp, out / name)

    interrupted = [False]

    def on_signal(signum, _frame):
        print(f"signal {signum}: saving and stopping", flush=True)
        interrupted[0] = True

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)
    last_save = time.time()

    log = open(out / "log.jsonl", "a")
    t_start = t0 = time.time()
    deadline = t_start + args.hours * 3600 if args.hours else float("inf")
    stop_file = Path(args.stop_file) if args.stop_file else None
    running = []
    model.train()
    while step < args.steps:
        # last 15% of a time-limited run: cosine decay to 10%
        if args.hours:
            frac = (time.time() - t_start) / (deadline - t_start)
            if args.cooldown:                             # whole run is the decay
                lr_scale[0] = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1)))
            else:
                lr_scale[0] = 1.0 if frac < 0.85 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (frac - 0.85) / 0.15))
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step)
        x, texts = batch_from(tr_idx, args.batch)
        inp, m = mask_tokens(x, model.mask_id)
        h, pad = text(texts)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp, h, pad, only=m)
            loss = F.cross_entropy(logits.float(), x[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
        except Exception as e:  # noqa: BLE001  (CUDA OOM surfaces as several error types)
            if "out of memory" not in str(e).lower() or args.batch <= 4:
                raise
            del x, inp, m, h, pad
            logits = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            args.batch //= 2
            print(f"OOM: batch size reduced to {args.batch}", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        running.append(loss.item())

        if step % 100 == 0:
            print(f"step {step}  loss {np.mean(running):.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{100 * args.batch / (time.time() - t0):.0f} samples/s  "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            running, t0 = [], time.time()
        stop = (time.time() > deadline or (stop_file is not None and stop_file.exists())
                or interrupted[0] or step >= args.steps)
        if step % args.eval_every == 0 or stop:
            vl = evaluate()
            rec = {"step": step, "val_loss": round(vl, 4), "hours": round((time.time() - t_start) / 3600, 2)}
            print("eval", rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if vl > best + args.diverge and (out / "best.pt").exists():
                # the 2026-10-07 run jumped from ~5.1 to ~7.8 and never came back: roll back
                ck = torch.load(out / "best.pt", map_location=dev)
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                del ck
                guard[0] *= 0.5
                msg = {"step": step, "diverged": round(vl, 4), "rolled_back_to_best": round(best, 4),
                       "lr_factor": guard[0]}
                print("GUARD", msg, flush=True)
                log.write(json.dumps(msg) + "\n")
                log.flush()
            else:
                if vl < best:
                    best = vl
                    save("best.pt")
                save("last.pt")
            last_save = t0 = time.time()
        elif time.time() - last_save > args.save_minutes * 60:
            save("last.pt")
            last_save = time.time()
        if step % args.sample_every == 0 or (stop and not interrupted[0]):
            decode_preview(model, text, tok, PREVIEW_PROMPTS, out / f"samples_{step:06d}.png")
            t0 = time.time()
        if stop:
            print(f"stopping at step {step}", flush=True)
            break


if __name__ == "__main__":
    main()
