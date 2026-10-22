"""Unattended overnight pipeline. Assumes the 50k build and 32^3 tokenizer training are
already running as separate processes.

    .venv/Scripts/python scripts/overnight.py --wake 07:45

1. wait for the 32^3 tokenizer to finish
2. convert new objects to 32^3, encode codes, train transformer (phase 1) until the
   data build finishes (or --phase1-latest)
3. backup gallery from phase 1
4. re-convert + re-encode everything, continue training (phase 2) until --wake
5. final gallery + MORNING.md summary

Every step is logged to data/overnight.log; a failed step is logged and the pipeline
moves on with whatever exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
DATA = ROOT / "data"
LOG = DATA / "overnight.log"
TOK_LOG = DATA / "train_vqvae32.log"
TOK_STEPS = 25000
BUILD_LOG = DATA / "objaverse_build_50k.log"
T2V_DIR = DATA / "runs" / "t2v"
STOP = T2V_DIR / "STOP"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def at(hhmm: str) -> float:
    """Next occurrence of HH:MM as a timestamp."""
    h, m = map(int, hhmm.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def run(name: str, args: list[str], logfile: str) -> bool:
    log(f"start {name}: {' '.join(args)}")
    with open(DATA / logfile, "w", encoding="utf-8") as out:
        rc = subprocess.call([PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=ENV)
    log(f"done  {name}: exit {rc}")
    return rc == 0


def tokenizer_done() -> bool:
    return TOK_LOG.exists() and f"'step': {TOK_STEPS}" in TOK_LOG.read_text(encoding="utf-8", errors="ignore")


def tokenizer_running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -match 'train_vqvae' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def build_done() -> bool:
    return BUILD_LOG.exists() and "\ntotal " in BUILD_LOG.read_text(encoding="utf-8", errors="ignore")


def train_t2v(extra: list[str], stop_when, logfile: str) -> None:
    STOP.unlink(missing_ok=True)
    args = [PY, "scripts/train_t2v.py", "--stop-file", str(STOP), *extra]
    log(f"start t2v: {' '.join(args[1:])}")
    with open(DATA / logfile, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=ENV)
        while proc.poll() is None:
            if stop_when():
                STOP.touch()
                log("stop condition met; asking t2v to finish its step and save")
                proc.wait()
                break
            time.sleep(60)
    log(f"done  t2v: exit {proc.returncode}")


def safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:  # noqa: BLE001
        log("ERROR\n" + traceback.format_exc())


def summary(wake_ts: float) -> None:
    lines = ["# Overnight results", "", f"Written {dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    tl = DATA / "runs" / "vqvae32" / "log.jsonl"
    if tl.exists():
        last = json.loads(tl.read_text().splitlines()[-1])
        lines += ["## Tokenizer (32³)", f"- step {last['step']}: shape IoU {last['occ_iou']:.2f}, "
                  f"block accuracy {last['block_acc']:.0%}, codes used {last['codes_used']}", ""]
    t2 = T2V_DIR / "log.jsonl"
    if t2.exists():
        recs = [json.loads(l) for l in t2.read_text().splitlines() if l]
        lines += ["## Text → 3D transformer", f"- {len(recs)} evals, last step {recs[-1]['step']}, "
                  f"val loss {recs[0]['val_loss']} → {recs[-1]['val_loss']}", ""]
    meta = DATA / "codes32" / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        lines += [f"- trained on {len(m['uids']):,} objects × 8 orientations", ""]
    gens = sorted((DATA / "generated").glob("*/gallery.png"))
    lines += ["## Galleries"] + [f"- {g.parent}" for g in gens] + [""]
    (ROOT / "MORNING.md").write_text("\n".join(lines), encoding="utf-8")
    log("wrote MORNING.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", default="07:45", help="have everything finished by HH:MM")
    ap.add_argument("--phase1-latest", default="05:30", help="start phase 2 by HH:MM even if build runs")
    args = ap.parse_args()
    wake, p1_latest = at(args.wake), at(args.phase1_latest)
    T2V_DIR.mkdir(parents=True, exist_ok=True)
    log(f"overnight start; wake {args.wake}, phase-1 cutoff {args.phase1_latest}")

    while not tokenizer_done():
        if not tokenizer_running():
            log("tokenizer process not running and not finished; using its last checkpoint")
            break
        time.sleep(60)
    log("tokenizer finished")

    safe(run, "vox32", ["scripts/make_vox32.py"], "overnight_vox32.log")
    safe(run, "encode", ["scripts/encode_codes.py"], "overnight_encode1.log")
    safe(train_t2v, [], lambda: build_done() or time.time() > p1_latest, "train_t2v.log")
    safe(run, "gallery-phase1", ["scripts/generate.py", "--gallery", "--ckpt", str(T2V_DIR / "last.pt"),
                                 "--out", str(DATA / "generated" / "phase1")], "overnight_gen1.log")

    if time.time() < wake - 45 * 60:
        safe(run, "vox32", ["scripts/make_vox32.py"], "overnight_vox32b.log")
        safe(run, "encode", ["scripts/encode_codes.py"], "overnight_encode2.log")
        hours = max(0.25, (wake - time.time()) / 3600 - 0.25)
        safe(train_t2v, ["--resume", "--hours", f"{hours:.2f}"], lambda: time.time() > wake - 10 * 60,
             "train_t2v.log")
        safe(run, "gallery-final", ["scripts/generate.py", "--gallery", "--ckpt", str(T2V_DIR / "last.pt"),
                                    "--seeds", "3", "--out", str(DATA / "generated" / "final")],
             "overnight_gen2.log")
    safe(summary, wake)
    log("overnight done")


if __name__ == "__main__":
    main()
