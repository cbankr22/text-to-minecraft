"""Encode every kept 32^3 object into tokenizer codes, in 8 orientations (4 turns x mirror).

    .venv/Scripts/python scripts/encode_codes.py --tokenizer data/runs/vqvae32/last.pt

Writes data/codes32/codes.npy (N, 8, 512) uint16 and data/codes32/meta.json
(uids, captions, short caption variants, val flags, codebook size). The array is written
through a memmap so RAM stays flat at any dataset size. Objects already in an existing
codes.npy made with the same tokenizer are copied over instead of re-encoded (pass --fresh
to re-encode everything).
--licenses limits which objects go in (e.g. by,cc0 for a release). Short captions come from
data/objaverse/short_captions.jsonl (scripts/make_short_captions.py) when it exists.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.dataset import is_val, load_manifest  # noqa: E402
from ttmc.vqvae import load_tokenizer, project_path  # noqa: E402

DATA = ROOT / "data" / "objaverse"


def orientations(b: np.ndarray) -> np.ndarray:
    out = []
    for k in range(4):
        r = np.rot90(b, k, axes=(1, 2))
        out += [r, r[:, :, ::-1]]
    return np.ascontiguousarray(np.stack(out))


def rel(path) -> str:
    """Project-relative posix path when possible, so meta.json works on other machines."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def previous_codes(out: Path, tokenizer: str) -> tuple[np.ndarray | None, dict[str, int]]:
    """Existing codes (memmapped) and uid -> row, if they came from the same tokenizer file."""
    meta_path, codes_path = out / "meta.json", out / "codes.npy"
    if not (meta_path.exists() and codes_path.exists()):
        return None, {}
    meta = json.loads(meta_path.read_text())
    same = project_path(meta["tokenizer"]).resolve() == Path(tokenizer).resolve() \
        and meta.get("tokenizer_mtime") == os.path.getmtime(tokenizer)
    if not same:
        return None, {}
    return np.load(codes_path, mmap_mode="r"), {u: i for i, u in enumerate(meta["uids"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "runs" / "vqvae32" / "last.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "codes32"))
    ap.add_argument("--batch", type=int, default=16, help="objects per batch (x8 orientations)")
    ap.add_argument("--fresh", action="store_true", help="re-encode everything")
    ap.add_argument("--licenses", default="", help="comma list, e.g. by,cc0 (default: all)")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast (faster 3D convs on some GPUs)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vox = DATA / "vox32"
    have = {e.name[:-4] for e in os.scandir(vox) if e.name.endswith(".npz")}
    rows = [r for r in load_manifest(DATA / "manifest.jsonl") if r["uid"] in have]
    if args.licenses:
        allowed = set(args.licenses.split(","))
        rows = [r for r in rows if r.get("license") in allowed]
    old, old_row = (None, {}) if args.fresh else previous_codes(out, args.tokenizer)

    tmp = out / "codes.tmp.npy"
    codes = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(len(rows), 8, 512))
    todo = []
    for i, r in enumerate(rows):
        j = old_row.get(r["uid"])
        if j is None:
            todo.append(i)
        else:
            codes[i] = old[j]
    print(f"{len(rows)} objects: {len(rows) - len(todo)} reused, {len(todo)} to encode", flush=True)

    tok = load_tokenizer(args.tokenizer)
    def load(i: int) -> np.ndarray:
        return orientations(np.load(vox / f"{rows[i]['uid']}.npz")["blocks"])

    # reading ~500k small files is the bottleneck (GPU idle at ~5%): load batches on threads,
    # one batch ahead of the GPU
    chunks = [todo[b:b + args.batch] for b in range(0, len(todo), args.batch)]
    pool = ThreadPoolExecutor(8)
    ahead = pool.submit(lambda c: list(pool.map(load, c)), chunks[0]) if chunks else None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
        for n, chunk in enumerate(chunks):
            b = n * args.batch
            grids = np.concatenate(ahead.result())
            if n + 1 < len(chunks):
                ahead = pool.submit(lambda c: list(pool.map(load, c)), chunks[n + 1])
            _, idx = tok.encode(torch.from_numpy(grids.astype(np.int64)).cuda())
            codes[chunk] = idx.view(len(chunk), 8, -1).cpu().numpy().astype(np.uint16)
            if (b // args.batch) % 100 == 0:
                print(f"{b + len(chunk)}/{len(todo)}", flush=True)
    codes.flush()
    del codes, old                                   # release the files before replacing them
    os.replace(tmp, out / "codes.npy")
    short = {}
    if (DATA / "short_captions.jsonl").exists():
        for line in (DATA / "short_captions.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            short[d["uid"]] = d["short"]
    meta = {"uids": [r["uid"] for r in rows], "captions": [r["caption"] for r in rows],
            "short": [short.get(r["uid"], []) for r in rows], "licenses": args.licenses or "all",
            "val": [is_val(r["uid"]) for r in rows], "codebook": tok.fsq.codebook_size,
            "tokenizer": rel(args.tokenizer), "tokenizer_mtime": os.path.getmtime(args.tokenizer)}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"encoded {len(rows)} objects -> {out / 'codes.npy'}  "
          f"({sum(meta['val'])} val)  codebook {meta['codebook']}  "
          f"short captions for {sum(bool(v) for v in meta['short'])}")


if __name__ == "__main__":
    main()
