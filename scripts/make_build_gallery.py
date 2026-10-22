"""Browsable gallery of the builds cut by ingest_world.py.

    .venv/Scripts/python scripts/make_build_gallery.py celestia_town

Renders two small views per build (the build alone, and the build in its terrain without
plants) into data/worlds/builds/<world>/thumbs/ and writes data/worlds/builds/<world>.html:
a grid you can filter by material or uid and sort by size. Open it in any browser.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.render import palette_colors, render_iso  # noqa: E402

BUILDS = ROOT / "data" / "worlds" / "builds"


def load_build(path: Path):
    d = np.load(path)
    shape = tuple(d["mask_shape"])
    mask = np.unpackbits(d["mask"])[: np.prod(shape)].reshape(shape).astype(bool)
    return d["blocks"], [str(p) for p in d["palette"]], mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--size", type=int, default=220)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from ingest_world import NATURE, classify

    rows = [json.loads(l) for l in (BUILDS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l and f'"world:{args.world}"' in l]
    rows = [r for r in rows if r.get("keep")]
    thumbs = BUILDS / args.world / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        a, b = thumbs / f"{r['uid']}_a.png", thumbs / f"{r['uid']}_b.png"
        if not (a.exists() and b.exists()):
            blocks, palette, mask = load_build(BUILDS / args.world / f"{r['uid']}.npz")
            lut = palette_colors(palette)
            nature = classify(blocks, palette) == NATURE
            render_iso(lut[blocks], mask, size=args.size, turns=1).save(a)
            render_iso(lut[blocks], ((blocks != 0) & ~nature) | mask, size=args.size, turns=1).save(b)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} rendered", flush=True)

    cards = []
    for r in rows:
        mats = ", ".join(f"{m} {n}" for m, n in r["materials"][:4])
        y, z, x = r["build_extent_yzx"]
        cards.append(
            f'<div class="card" data-main="{html.escape(r["materials"][0][0])}" '
            f'data-n="{r["n_build_blocks"]}" data-h="{y}" data-uid="{r["uid"]}" '
            f'data-text="{html.escape(r["uid"] + " " + mats)}">'
            f'<img loading="lazy" src="{args.world}/thumbs/{r["uid"]}_a.png">'
            f'<img loading="lazy" src="{args.world}/thumbs/{r["uid"]}_b.png">'
            f'<div class="cap"><b>{r["uid"].rsplit("_", 1)[1]}</b> {x}&times;{z}&times;{y} '
            f'({r["n_build_blocks"]} blocks)<br>{html.escape(mats)}</div></div>')
    mains = sorted({r["materials"][0][0] for r in rows},
                   key=lambda m: -sum(r["materials"][0][0] == m for r in rows))
    options = "".join(
        f'<label><input type="checkbox" class="hide" value="{html.escape(m)}"> '
        f'{html.escape(m)} ({sum(r["materials"][0][0] == m for r in rows)})</label>' for m in mains[:12])
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{args.world} builds</title><style>
:root {{ --bg:#16171a; --card:#222429; --fg:#ddd; --mut:#999; }}
body {{ background:var(--bg); color:var(--fg); font:13px system-ui,sans-serif; margin:0; padding:16px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0 12px; z-index:1; }}
.bar input[type=text] {{ padding:6px 8px; width:240px; background:var(--card); color:var(--fg); border:1px solid #444; }}
.bar select {{ padding:5px; background:var(--card); color:var(--fg); border:1px solid #444; }}
.hides {{ margin-top:8px; color:var(--mut); }} .hides label {{ margin-right:12px; white-space:nowrap; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:10px; }}
.card {{ background:var(--card); border-radius:6px; padding:6px; }}
.card img {{ width:50%; image-rendering:pixelated; }} .cap {{ color:var(--mut); padding:4px 2px 0; }}
.cap b {{ color:var(--fg); }}
</style></head><body>
<div class="bar"><b>{args.world}</b>: <span id="count"></span> builds shown &nbsp;
<input type="text" id="q" placeholder="filter: material or id"> &nbsp;
sort <select id="sort"><option value="uid">id</option><option value="n">size (blocks)</option>
<option value="h">height</option></select>
<div class="hides">hide main material: {options}</div>
<div style="color:var(--mut);margin-top:4px">left: build alone &middot; right: build in its terrain (plants hidden)</div></div>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
const grid = document.getElementById('grid'), cards = [...grid.children];
function update() {{
  const q = document.getElementById('q').value.toLowerCase();
  const hide = new Set([...document.querySelectorAll('.hide:checked')].map(c => c.value));
  const key = document.getElementById('sort').value;
  const sorted = [...cards].sort((a, b) => key === 'uid' ? a.dataset.uid.localeCompare(b.dataset.uid)
                                                          : b.dataset[key] - a.dataset[key]);
  let n = 0;
  for (const c of sorted) {{
    const show = !hide.has(c.dataset.main) && c.dataset.text.toLowerCase().includes(q);
    c.style.display = show ? '' : 'none'; n += show; grid.appendChild(c);
  }}
  document.getElementById('count').textContent = n;
}}
document.querySelectorAll('input,select').forEach(e => e.addEventListener('input', update));
update();
</script></body></html>"""
    out = BUILDS / f"{args.world}.html"
    out.write_text(page, encoding="utf-8")
    print(f"gallery: {out}")


if __name__ == "__main__":
    main()
