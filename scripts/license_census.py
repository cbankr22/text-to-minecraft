"""Count licenses over all Cap3D-captioned Objaverse objects, before the full ingest.

    .venv/Scripts/python scripts/license_census.py

Goes through the 160 Objaverse metadata shards one at a time (~3.6 MB each, downloaded
on demand into hf-objaverse-v1/metadata, where objaverse_build.py reuses them) and writes
one slim row per captioned object to data/objaverse/annotations_slim.jsonl:
    uid, license, faces, name, author (display name), username, url
That file is also the source for the CC-BY attribution list. Resumable by shard.
"""
from __future__ import annotations

import collections
import gzip
import json
import socket
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "objaverse"
META = DATA / "hf-objaverse-v1" / "metadata"
OUT = DATA / "annotations_slim.jsonl"
MAX_FACES = 300_000          # same cut as objaverse_build.py
socket.setdefaulttimeout(60)


def shard(i: int) -> str:
    return f"{i // 1000:03d}-{i % 1000:03d}"


def load_shard(sid: str) -> dict:
    path = META / f"{sid}.json.gz"
    if not path.exists():
        META.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{sid}.json.gz"
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    with gzip.open(path, "rb") as f:
        return json.load(f)


def main() -> None:
    caps = pd.read_csv(DATA / "Cap3D_automated_Objaverse_full.csv", header=None,
                       names=["uid", "caption"], usecols=[0])
    captioned = set(caps.uid)
    del caps
    done_shards = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            done_shards.add(json.loads(line)["shard"])
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(160):
            sid = shard(i)
            if sid in done_shards:
                continue
            data = load_shard(sid)
            rows = []
            for uid, a in data.items():
                if uid not in captioned:
                    continue
                u = a.get("user") or {}
                rows.append({"uid": uid, "shard": sid, "license": (a.get("license") or "").lower(),
                             "faces": a.get("faceCount") or 0, "name": a.get("name"),
                             "author": u.get("displayName"), "username": u.get("username"),
                             "url": a.get("viewerUrl")})
            del data
            # one line per shard marks it done even if it had no captioned objects
            rows = rows or [{"uid": None, "shard": sid}]
            f.write("".join(json.dumps(r) + "\n" for r in rows))
            f.flush()
            print(f"{sid}: {len(rows)} captioned", flush=True)
    report()


def report() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["uid"]]
    total = collections.Counter(r["license"] for r in rows)
    usable = collections.Counter(r["license"] for r in rows if r["faces"] <= MAX_FACES)
    # keep rate per license from the 50k objects already built
    built, kept = collections.Counter(), collections.Counter()
    for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        built[m.get("license")] += 1
        kept[m.get("license")] += bool(m.get("keep"))
    print(f"\n{len(rows)} captioned objects with metadata")
    print(f"{'license':<10}{'captioned':>11}{'<=300k faces':>14}{'keep rate':>11}{'est. kept':>11}")
    for lic, n in total.most_common():
        rate = kept[lic] / built[lic] if built[lic] else float("nan")
        est = usable[lic] * rate if built[lic] else float("nan")
        print(f"{lic:<10}{n:>11,}{usable[lic]:>14,}{rate:>11.1%}{est:>11,.0f}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()
