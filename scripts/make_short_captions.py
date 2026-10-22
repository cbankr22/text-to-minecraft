"""Short caption variants for every built object ("a red and black sword", "a sword").

    .venv/Scripts/python scripts/make_short_captions.py           # objects in manifest.jsonl
    .venv/Scripts/python scripts/make_short_captions.py --all     # every Cap3D caption (~785k)

Writes data/objaverse/short_captions.jsonl, one {"uid", "short": [...]} per line.
Resumable: uids already in the file are skipped. Needs spaCy + en_core_web_sm
(pip install spacy && python -m spacy download en_core_web_sm). ~1k captions/s per process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.captions import make_variants  # noqa: E402

DATA = ROOT / "data" / "objaverse"
OUT = DATA / "short_captions.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all Cap3D captions, not just built objects")
    ap.add_argument("--chunk", type=int, default=20_000)
    args = ap.parse_args()

    done = set()
    if OUT.exists():
        done = {json.loads(l)["uid"] for l in OUT.read_text(encoding="utf-8").splitlines() if l}
    if args.all:
        df = pd.read_csv(DATA / "Cap3D_automated_Objaverse_full.csv", header=None, names=["uid", "caption"])
        todo = [(u, c) for u, c in zip(df.uid, df.caption) if u not in done]
        del df
    else:
        todo = []
        for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("keep") and r["uid"] not in done:
                todo.append((r["uid"], r["caption"]))
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            variants = make_variants([str(c) for _, c in chunk])
            f.write("".join(json.dumps({"uid": u, "short": v}) + "\n" for (u, _), v in zip(chunk, variants)))
            f.flush()
            n = i + len(chunk)
            print(f"{n}/{len(todo)}  {n / (time.time() - t0):.0f}/s", flush=True)


if __name__ == "__main__":
    main()
