"""Hand slicing tool: draw a box around each build on a top-down map of a world.

    .venv/Scripts/python scripts/slicer.py celestia_town          # then open http://127.0.0.1:8771
    .venv/Scripts/python scripts/slicer.py celestia_town --cut    # cut every box into builds

The first run scans the world once (about a minute) and caches a top-down map in
data/worlds/<world>/slicer/. The page shows the map with the automatic footprints as
suggestions; you accept, fix or draw boxes and tag them. Every change is saved to
data/worlds/<world>/slicer/boxes.json, which (with the world itself) is the real result:
--cut turns it into npz builds + manifest rows (method "hand"), replacing the automatic
ones for that world. Run the server in your own terminal; Ctrl+C stops it.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ingest_world import (BUILT, EDIT, MARGIN, OUT as BUILDS, Y_MIN, classify,  # noqa: E402
                          describe, scan_world)
from make_label_pack import grid_image  # noqa: E402
from ttmc.formats import read_level_dat  # noqa: E402
from ttmc.textured import Textures  # noqa: E402
from ttmc.world import World, cut_box  # noqa: E402

WORLDS = ROOT / "data" / "worlds"
PAGE = Path(__file__).with_name("slicer.html")
TAGS = ["house", "tower", "church", "shop", "farm", "wall", "bridge", "tree", "other"]


# --------------------------------------------------------------------------- #
# Prepare: one scan -> map image + height maps + suggestions
# --------------------------------------------------------------------------- #

def prepare(world_dir: Path, cache: Path) -> None:
    world = World(world_dir)
    group = world.region_groups()[0]                 # the main area of the world
    print(f"scanning {world_dir.name} ({len(group)} regions, once)...", flush=True)
    maps, present, (ox, oz), _ = scan_world(world, group, colors=True)
    # Crop to the explored chunks.
    zs, xs = np.nonzero(present)
    z0, z1, x0, x1 = zs.min() * 16, (zs.max() + 1) * 16, xs.min() * 16, (xs.max() + 1) * 16
    crop = {k: v[z0:z1, x0:x1] for k, v in maps.items()}
    ox, oz = ox + x0, oz + z0
    # Hillshade from the height of the highest block.
    h = crop["topany"].astype(np.float32)
    shade = np.clip(1 + 0.07 * (h - np.roll(np.roll(h, 1, 0), 1, 1)), 0.6, 1.35)
    rgb = np.clip(crop["rgb"] * shade[..., None], 0, 255).astype(np.uint8)
    rgb[crop["topany"] < 0] = (20, 20, 24)
    cache.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(cache / "map.png")
    np.savez_compressed(cache / "maps.npz", **{k: crop[k] for k in ("count", "top", "surf", "ground")})
    # Suggestions: the automatic footprints from ingest_world (kept and too-large ones).
    sugg = []
    manifest = BUILDS / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["source"] != f"world:{world_dir.name}" or r.get("method") == "hand":
                continue
            if not (r.get("keep") or "too_large" in r.get("reject", [])):
                continue
            sx, sz = r["footprint_xz"]
            bx, bz = r["origin"][0] + MARGIN, r["origin"][2] + MARGIN
            sugg.append({"x0": bx, "z0": bz, "x1": bx + sx, "z1": bz + sz})
    meta = {"world": world_dir.name, "ox": int(ox), "oz": int(oz), "width": int(x1 - x0),
            "height": int(z1 - z0), "suggestions": sugg, "tags": TAGS}
    (cache / "meta.json").write_text(json.dumps(meta))
    print(f"map {meta['width']}x{meta['height']} blocks, {len(sugg)} suggestions", flush=True)


class Slicer:
    def __init__(self, name: str):
        self.name = name
        self.world_dir = WORLDS / name
        self.cache = self.world_dir / "slicer"
        if not (self.cache / "meta.json").exists():
            prepare(self.world_dir, self.cache)
        self.meta = json.loads((self.cache / "meta.json").read_text())
        m = np.load(self.cache / "maps.npz")
        self.maps = {k: m[k] for k in m.files}
        self.world = World(self.world_dir)
        self.textures = Textures()
        self.lock = threading.Lock()
        self.boxes_path = self.cache / "boxes.json"

    def state(self) -> dict:
        """{"boxes": [...], "rejected": [suggestion indices]} as saved by the page."""
        if not self.boxes_path.exists():
            return {"boxes": [], "rejected": []}
        data = json.loads(self.boxes_path.read_text())
        return data if isinstance(data, dict) else {"boxes": data, "rejected": []}

    def boxes(self) -> list:
        return self.state()["boxes"]

    def save(self, state: dict) -> None:
        tmp = self.boxes_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self.boxes_path)

    def auto_y(self, b: dict) -> tuple[int, int]:
        """Ground (median, minus 2) to the highest built block inside the box."""
        ox, oz = self.meta["ox"], self.meta["oz"]
        sl = (slice(max(b["z0"] - oz, 0), max(b["z1"] - oz, 0)),
              slice(max(b["x0"] - ox, 0), max(b["x1"] - ox, 0)))
        count, top = self.maps["count"][sl], self.maps["top"][sl]
        ground, surf = self.maps["ground"][sl], self.maps["surf"][sl]
        built = count > 0
        if not built.any():
            g = int(np.median(surf)) if surf.size else 64 - Y_MIN
            return g + Y_MIN - 2, g + Y_MIN + 12
        g = np.where(surf - ground <= 16, ground, surf)[built]
        return int(np.median(g)) + Y_MIN - 2, int(top[built].max()) + Y_MIN + 1

    def cut(self, b: dict, margin: int = 0):
        """Blocks of the box (+ margin of context), palette, and the build mask."""
        y0, y1 = (b["y0"], b["y1"]) if b.get("y0") is not None else self.auto_y(b)
        with self.lock:
            blocks, palette, _ = cut_box(self.world, b["x0"] - margin, y0 - margin // 2,
                                         b["z0"] - margin, b["x1"] + margin,
                                         y1 + margin // 2 + 1, b["z1"] + margin)
            self.world.clear()
        cls = classify(blocks, palette)
        mask = (cls == BUILT) | (cls == EDIT)
        inside = np.zeros_like(mask)
        m, mh = margin, margin // 2
        inside[mh:inside.shape[0] - mh, m:inside.shape[1] - m, m:inside.shape[2] - m] = True
        return blocks, palette, mask & inside, (b["x0"] - margin, y0 - mh, b["z0"] - margin)

    def preview(self, b: dict) -> bytes:
        blocks, palette, mask, _ = self.cut(b)
        img = grid_image("preview", blocks, palette, mask, self.textures, view=300)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


def make_handler(s: Slicer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/map.png":
                    self.send(200, (s.cache / "map.png").read_bytes(), "image/png")
                elif u.path == "/meta.json":
                    self.send(200, json.dumps({**s.meta, "boxes": s.state()}).encode(),
                              "application/json")
                elif u.path in ("/auto_y", "/preview"):
                    b = {k: int(q[k]) for k in ("x0", "z0", "x1", "z1")}
                    if "y0" in q and "y1" in q:
                        b["y0"], b["y1"] = int(q["y0"]), int(q["y1"])
                    if u.path == "/auto_y":
                        y0, y1 = s.auto_y(b)
                        self.send(200, json.dumps({"y0": y0, "y1": y1}).encode(), "application/json")
                    else:
                        self.send(200, s.preview(b), "image/png")
                else:
                    self.send(404, b"not found", "text/plain")
            except Exception as e:                    # keep the server alive
                self.send(500, repr(e).encode(), "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                return self.send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            s.save(json.loads(self.rfile.read(n)))
            self.send(200, b"ok", "text/plain")

    return Handler


# --------------------------------------------------------------------------- #
# Cut: boxes.json -> builds
# --------------------------------------------------------------------------- #

def cut_all(s: Slicer, license_: str, url: str) -> None:
    boxes = s.boxes()
    info = read_level_dat(s.world_dir / "level.dat")
    out_dir = BUILDS / s.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = []
    for i, b in enumerate(boxes, start=1):
        blocks, palette, mask, origin = s.cut(b, margin=MARGIN)
        if mask.sum() == 0:
            continue
        uid = f"{s.name}_h{i:04d}"
        np.savez_compressed(out_dir / f"{uid}.npz", blocks=blocks, palette=np.array(palette),
                            mask=np.packbits(mask, axis=None), mask_shape=np.array(mask.shape))
        ys, zs, xs = np.nonzero(mask)
        rows.append({"uid": uid, "source": f"world:{s.name}", "method": "hand",
                     "tag": b.get("tag"), "license": license_, "url": url,
                     "mc_version": info["mc_version"], "caption": "", "status": "ok",
                     "keep": True, "reject": [], "origin": list(origin),
                     "box": {k: b.get(k) for k in ("x0", "z0", "x1", "z1", "y0", "y1")},
                     "shape": list(blocks.shape),
                     "footprint_xz": [b["x1"] - b["x0"], b["z1"] - b["z0"]],
                     "build_extent_yzx": [int(np.ptp(ys) + 1), int(np.ptp(zs) + 1), int(np.ptp(xs) + 1)],
                     "n_build_blocks": int(mask.sum()), "materials": describe(blocks, palette, mask)})
        print(f"  {uid} {b.get('tag')} {rows[-1]['build_extent_yzx']}", flush=True)
    manifest = BUILDS / "manifest.jsonl"           # replaces this world's automatic rows
    old = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    old = [l for l in old if l and json.loads(l)["source"] != f"world:{s.name}"]
    manifest.write_text("".join(l + "\n" for l in old + [json.dumps(r) for r in rows]),
                        encoding="utf-8")
    print(f"cut {len(rows)} builds from {len(boxes)} boxes -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("world", help="folder name under data/worlds")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--rescan", action="store_true", help="rebuild the cached map")
    ap.add_argument("--cut", action="store_true", help="cut boxes.json into builds and exit")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--source-url", default="")
    args = ap.parse_args()
    if args.rescan:
        (WORLDS / args.world / "slicer" / "meta.json").unlink(missing_ok=True)
    s = Slicer(args.world)
    if args.cut:
        return cut_all(s, args.license, args.source_url)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"slicer for {args.world}: {url}  (Ctrl+C to stop; boxes autosave to {s.boxes_path})")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(s))
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
