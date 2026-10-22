"""Finish the overnight run after a mid-phase restart: continue t2v training until
--wake minus 10 min, then the final gallery and MORNING.md.

    .venv/Scripts/python scripts/finish_night.py --wake 07:45 --batch 24
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight import DATA, T2V_DIR, at, log, run, safe, summary, train_t2v  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", default="07:45")
    ap.add_argument("--batch", default="24")
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--gallery-ckpt", default="best.pt")
    args = ap.parse_args()
    wake = at(args.wake)
    hours = max(0.25, (wake - time.time()) / 3600 - 0.25)
    log(f"finish_night: phase 2 restart at batch {args.batch}, lr {args.lr} for {hours:.2f} h")
    safe(train_t2v, ["--resume", "--hours", f"{hours:.2f}", "--batch", args.batch, "--lr", args.lr],
         lambda: time.time() > wake - 10 * 60, "train_t2v.log")
    safe(run, "gallery-final", ["scripts/generate.py", "--gallery", "--ckpt", str(T2V_DIR / args.gallery_ckpt),
                                "--seeds", "3", "--out", str(DATA / "generated" / "final")],
         "overnight_gen2.log")
    safe(summary, wake)
    log("overnight done")


if __name__ == "__main__":
    main()
