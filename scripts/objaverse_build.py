"""Build the Objaverse voxel dataset: captioned meshes -> 64^3 Minecraft block grids.

    .venv/Scripts/python scripts/objaverse_build.py --n 5000
    .venv/Scripts/python scripts/objaverse_build.py --n 1000000 --groups 160 --save 32   # everything
    ... --licenses by,cc0          # only these licenses (1.0 release set)

Resumable: uids already in manifest.jsonl are skipped. GLBs are deleted after
voxelizing unless --keep-glb. Set HF_TOKEN (or `huggingface-cli login`) for bulk runs;
anonymous Hugging Face downloads get rate-limited. Outputs under data/objaverse/:
    vox64/<uid>.npz      blocks (Y,Z,X uint8 tokens), rgb (Y,Z,X,3 uint8)   (--save 64)
    vox32/<uid>.npz      blocks downsampled 2x, kept objects only           (--save 32)
    manifest.jsonl       uid, caption, license, status, quality metrics, keep, reject reasons
    sheet_kept.png       random sample of accepted objects
    sheet_rejected.png   random sample of rejected objects with reasons
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import sys
import time
import urllib.request
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import objaverse  # noqa: E402

from ttmc.downsample import down2  # noqa: E402
from ttmc.quality import assess  # noqa: E402
from ttmc.render import contact_sheet, render_iso  # noqa: E402
from ttmc.voxelize import block_vocab, voxelize  # noqa: E402

DATA = ROOT / "data" / "objaverse"
MANIFEST = DATA / "manifest.jsonl"
CAPTIONS_URL = ("https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
                "Cap3D_automated_Objaverse_full.csv")
# Voxelizing is a derivative work, so skip no-derivatives licenses.
EXCLUDED_LICENSES = {"by-nd", "by-nc-nd"}
MAX_FACES = 300_000
RETRYABLE = ("MemoryError", "BrokenProcessPool")
MIN_FREE_GB = 3.0

objaverse._VERSIONED_PATH = str(DATA / "hf-objaverse-v1")
# urlretrieve has no timeout of its own; a dead connection would hang a batch forever.
socket.setdefaulttimeout(60)


def load_captions() -> dict[str, str]:
    path = DATA / "Cap3D_automated_Objaverse_full.csv"
    if not path.exists():
        print("downloading Cap3D captions (~255 MB)…")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CAPTIONS_URL, path)
    df = pd.read_csv(path, header=None, names=["uid", "caption"])
    return dict(zip(df.uid, df.caption))


def candidate_pool(captions: dict[str, str], groups: int, seed: int) -> list[str]:
    """Captioned uids from `groups` random metadata shards (~3.6 MB of annotations each)."""
    paths = object_paths()
    by_group: dict[str, list[str]] = {}
    for uid in captions:
        if uid in paths:
            by_group.setdefault(paths[uid].split("/")[1], []).append(uid)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_group), min(groups, len(by_group)))
    pool = [u for g in chosen for u in by_group[g]]
    rng.shuffle(pool)
    return pool


def slim_annotations(uids: list[str], licenses: set[str] | None = None) -> dict[str, dict]:
    """Annotations for the usable uids, one metadata shard at a time, keeping only the fields
    the manifest needs. Loading all ~800k full records at once would take several GB of RAM."""
    paths = object_paths()
    by_group: dict[str, list[str]] = {}
    for u in uids:
        by_group.setdefault(paths[u].split("/")[1], []).append(u)
    out = {}
    for _, group_uids in sorted(by_group.items()):
        for uid, a in objaverse.load_annotations(group_uids).items():
            lic = (a.get("license") or "").lower()
            if lic not in EXCLUDED_LICENSES and (licenses is None or lic in licenses) \
                    and (a.get("faceCount") or 0) <= MAX_FACES:
                out[uid] = {"name": a.get("name"), "license": a.get("license"),
                            "viewerUrl": a.get("viewerUrl"), "uri": a.get("uri")}
    return out


def process_one(uid: str, glb: str, res: int, save: tuple[int, ...]) -> dict:
    t = time.time()
    vr = voxelize(Path(glb), res=res)
    metrics, reasons = assess(vr.blocks)
    if res in save:
        np.savez_compressed(DATA / f"vox{res}" / f"{uid}.npz", blocks=vr.blocks, rgb=vr.rgb)
    if res // 2 in save and not reasons:
        np.savez_compressed(DATA / f"vox{res // 2}" / f"{uid}.npz", blocks=down2(vr.blocks))
    occ = vr.blocks > 0
    return {"n_faces": vr.n_faces, **metrics,
            "n_block_types": int(len(np.unique(vr.blocks[occ]))),
            "keep": not reasons, "reject": reasons, "seconds": round(time.time() - t, 2)}


def free_gb() -> float:
    return shutil.disk_usage(DATA).free / 1e9


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l]


HF_BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


@lru_cache(maxsize=1)
def hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_one(uid: str, object_path: str, retries: int = 4) -> str | None:
    """Download with retries; returns local path or None. Never raises."""
    dest = Path(objaverse._VERSIONED_PATH) / object_path
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(HF_BASE + object_path, headers=hf_headers())
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, dest)
            return str(dest)
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    return None


@lru_cache(maxsize=1)
def object_paths() -> dict[str, str]:
    return objaverse._load_object_paths()


def download_batch(batch: list[str], threads: int) -> dict[str, str]:
    paths = object_paths()
    with ThreadPoolExecutor(threads) as ex:
        results = ex.map(lambda u: (u, download_one(u, paths[u])), batch)
    return {u: p for u, p in results if p}


def limit_memory(gb: float) -> None:
    """Worker initializer: cap address space so one huge mesh raises MemoryError (logged as an
    error row) instead of exhausting the machine. Linux only; a no-op elsewhere."""
    try:
        import resource
    except ImportError:
        return
    if gb > 0:
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * 2**30), int(gb * 2**30)))


def run_batch(batch: list[str], glbs: dict[str, str], ann: dict, captions: dict, args) -> list[dict]:
    """Voxelize one downloaded batch; `glbs` maps uid -> local file (missing = download failed)."""
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=limit_memory, initargs=(args.mem_gb,)) as ex:
        futs = {u: ex.submit(process_one, u, glbs[u], args.res, args.save) for u in batch if u in glbs}
        for uid in batch:
            a = ann[uid]
            row = {"uid": uid, "caption": captions[uid], "name": a.get("name"),
                   "license": a.get("license"), "source": "objaverse",
                   "url": a.get("viewerUrl") or a.get("uri")}
            if uid not in futs:
                row.update(status="download_failed", keep=False)
            else:
                try:
                    row.update(status="ok", **futs[uid].result(timeout=args.timeout))
                except FutTimeout:
                    row.update(status="timeout", keep=False)
                except Exception as e:  # noqa: BLE001
                    row.update(status="error", keep=False, error=f"{type(e).__name__}: {e}"[:300])
            rows.append(row)
    if not args.keep_glb:
        for p in glbs.values():
            try:
                os.remove(p)
            except OSError:
                pass
    return rows


def make_sheets(rows: list[dict], res: int, seed: int, k: int = 40) -> None:
    _, vocab_rgb = block_vocab()
    rng = random.Random(seed)
    ok = [r for r in rows if r.get("status") == "ok"]
    for name, subset, label in [
        ("sheet_kept.png", [r for r in ok if r["keep"]], lambda r: r["caption"]),
        ("sheet_rejected.png", [r for r in ok if not r["keep"]],
         lambda r: f"[{', '.join(r['reject'])}] {r['caption']}"),
    ]:
        sample = rng.sample(subset, min(k, len(subset)))
        tiles = []
        for r in sample:
            path = DATA / f"vox{res}" / f"{r['uid']}.npz"
            if not path.exists():                 # e.g. --save 32 keeps no vox64 copy
                continue
            d = np.load(path)
            occ = d["blocks"] > 0
            block_rgb = vocab_rgb[d["blocks"]].astype(np.uint8)
            tiles.append(([render_iso(block_rgb, occ), render_iso(block_rgb, occ, turns=2)],
                          label(r)))
        if tiles:
            contact_sheet(tiles, cols=5).save(DATA / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="total objects to have processed")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads")
    ap.add_argument("--save", default="64,32",
                    help="grids to write: 64 (~19 KB/object, every object) and/or 32 (~1 KB, kept only)")
    ap.add_argument("--keep-glb", action="store_true")
    ap.add_argument("--mem-gb", type=float, default=0, help="per-worker memory cap (Linux), 0 = none")
    ap.add_argument("--licenses", default="", help="comma list to keep, e.g. by,cc0 (default: all but ND)")
    args = ap.parse_args()
    args.save = tuple(int(v) for v in args.save.split(","))

    for r in args.save:
        (DATA / f"vox{r}").mkdir(parents=True, exist_ok=True)
    print("Hugging Face token:", "yes" if hf_headers() else "NO (expect rate limits on big runs)")
    # rows that failed for reasons unrelated to the object (a crashed or capped worker) are retried
    done = {r["uid"] for r in read_manifest() if not any(
        x in (r.get("error") or "") for x in RETRYABLE)}
    captions = load_captions()
    pool = [u for u in candidate_pool(captions, args.groups, args.seed) if u not in done]
    print(f"{len(done)} already processed; {len(pool)} candidates")
    ann = slim_annotations(pool, set(args.licenses.split(",")) if args.licenses else None)
    todo = [u for u in pool if u in ann]
    todo = todo[: max(0, args.n - len(done))]
    print(f"processing {len(todo)} objects in batches of {args.batch}")

    t0 = time.time()
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    # download the next batch while this one voxelizes, so the CPUs never wait on the network
    prefetch = ThreadPoolExecutor(1)
    nxt = prefetch.submit(download_batch, batches[0], args.threads) if batches else None
    for b, batch in enumerate(batches):
        i = b * args.batch
        glbs = nxt.result()
        if free_gb() < MIN_FREE_GB:
            print(f"stopping: only {free_gb():.1f} GB free")
            break
        nxt = prefetch.submit(download_batch, batches[b + 1], args.threads) if b + 1 < len(batches) else None
        rows = run_batch(batch, glbs, ann, captions, args)
        with open(MANIFEST, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_done = i + len(rows)
        rate = n_done / (time.time() - t0)
        kept = sum(bool(r.get("keep")) for r in rows)
        print(f"[{n_done}/{len(todo)}] batch kept {kept}/{len(rows)}  "
              f"{rate:.1f} obj/s  eta {(len(todo) - n_done) / rate / 60:.0f} min  "
              f"free {free_gb():.1f} GB", flush=True)

    rows = read_manifest()
    make_sheets(rows, args.res, args.seed)
    ok = [r for r in rows if r.get("status") == "ok"]
    reasons: dict[str, int] = {}
    for r in ok:
        for x in r.get("reject") or []:
            reasons[x] = reasons.get(x, 0) + 1
    print(f"\ntotal {len(rows)}  ok {len(ok)}  kept {sum(r['keep'] for r in ok)}  "
          f"rejects {reasons}  failures {len(rows) - len(ok)}")


if __name__ == "__main__":
    main()
