"""Side-by-side 3D viewer: original vs current VQ-VAE reconstruction on held-out objects.

    .venv/Scripts/python scripts/make_recon_viewer.py [--n 60]   # -> data/objaverse/recon_viewer.html

Safe to run while training: loads data/runs/vqvae/last.pt read-only, inference only.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.dataset import VoxelDataset, load_manifest, split  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import VoxelVQVAE  # noqa: E402

DATA = ROOT / "data" / "objaverse"
CKPT = ROOT / "data" / "runs" / "vqvae" / "last.pt"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Reconstruction Viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#141518;--panel:#1d1f24;--line:#2a2d33;--fg:#e6e6e6;--muted:#9aa0a6;--accent:#6aa9ff;
  --good:#5ec27a;--bad:#e5675e;--warn:#e7b549;--extra:#5aa0f0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;height:100vh}
#list{width:300px;overflow:auto;background:var(--panel);border-right:1px solid var(--line);flex-shrink:0}
#list .hdr{padding:12px;color:var(--muted);font-size:12px;border-bottom:1px solid var(--line)}
#list .item{padding:8px 12px;cursor:pointer;border-bottom:1px solid #26292e;color:var(--muted);display:flex;gap:8px}
#list .item:hover{color:var(--fg)} #list .item.on{color:var(--fg);background:#262a31;box-shadow:inset 3px 0 var(--accent)}
#list .score{font-variant-numeric:tabular-nums;min-width:38px;text-align:right}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#top{padding:12px 16px;border-bottom:1px solid var(--line)}
#top b{font-size:16px} #stats{color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums}
#views{flex:1;display:flex;min-height:0}
.view{flex:1;position:relative;border-right:1px solid var(--line);min-width:0}
.view:last-child{border-right:none}
.label{position:absolute;left:12px;top:10px;color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase;pointer-events:none}
canvas{display:block}
#bar{padding:10px 16px;border-top:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:var(--panel);color:var(--fg);border:1px solid #3a3d44;padding:6px 12px;border-radius:6px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent)}
.legend{color:var(--muted);font-size:12px;display:flex;gap:12px;margin-left:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
@media (max-width:800px){body{flex-direction:column}#list{width:auto;height:25vh}#views{flex-direction:column}.view{border-right:none;border-bottom:1px solid var(--line)}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="list"><div class="hdr">__HEADER__</div></div>
<div id="main">
  <div id="top"><b id="cap"></b><div id="stats"></div></div>
  <div id="views">
    <div class="view" id="vL"><div class="label">Original</div></div>
    <div class="view" id="vR"><div class="label" id="rlabel">Reconstruction</div></div>
  </div>
  <div id="bar">
    <button data-m="recon" class="on">Reconstruction</button>
    <button data-m="errors">Errors</button>
    <button data-m="codes">Codes (16³)</button>
    <div class="legend" id="legend"></div>
  </div>
</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const ITEMS = __ITEMS__, BLOCKS = __BLOCKS__, R = 64;
const ERR = {ok:[0.37,0.76,0.48], missing:[0.9,0.4,0.37], extra:[0.35,0.63,0.94], wrong:[0.9,0.71,0.29]};

function makeView(el){
  const r = new THREE.WebGLRenderer({antialias:true}); r.setPixelRatio(devicePixelRatio); el.prepend(r.domElement);
  const s = new THREE.Scene(); s.background = new THREE.Color(0x141518);
  s.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1,2,1.4); s.add(sun);
  s.add(new THREE.GridHelper(R, 8, 0x333740, 0x25282d));
  return {el, r, s, mesh:null};
}
const L = makeView(document.getElementById('vL')), Rv = makeView(document.getElementById('vR'));
const cam = new THREE.PerspectiveCamera(40, 1, 0.1, 1000); cam.position.set(R*1.3, R*1.0, R*1.3);
const ctlL = new OrbitControls(cam, L.r.domElement), ctlR = new OrbitControls(cam, Rv.r.domElement);
for (const c of [ctlL, ctlR]) { c.target.set(0, R*0.35, 0); c.update(); }
function resize(){ for (const v of [L, Rv]) { v.r.setSize(v.el.clientWidth, v.el.clientHeight); }
  cam.aspect = L.el.clientWidth / L.el.clientHeight; cam.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

async function inflate(b64){
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
const idx = (y,z,x) => (y*R+z)*R+x;
function surface(grid){
  const solid = (y,z,x) => y>=0&&z>=0&&x>=0&&y<R&&z<R&&x<R&&grid[idx(y,z,x)]>0;
  const out = [];
  for (let y=0;y<R;y++) for (let z=0;z<R;z++) for (let x=0;x<R;x++){
    if (!grid[idx(y,z,x)]) continue;
    if (solid(y+1,z,x)&&solid(y-1,z,x)&&solid(y,z+1,x)&&solid(y,z-1,x)&&solid(y,z,x+1)&&solid(y,z,x-1)) continue;
    out.push([y,z,x]);
  }
  return out;
}
function setMesh(v, cells, colorOf){
  if (v.mesh){ v.s.remove(v.mesh); v.mesh.geometry.dispose(); v.mesh.material.dispose(); }
  const m = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshLambertMaterial(), Math.max(cells.length,1));
  m.count = cells.length;
  const mat = new THREE.Matrix4(), col = new THREE.Color();
  cells.forEach(([y,z,x],k) => {
    mat.makeTranslation(x-R/2+0.5, y+0.5, z-R/2+0.5); m.setMatrixAt(k, mat);
    const c = colorOf(y,z,x); col.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace); m.setColorAt(k, col);
  });
  v.s.add(m); v.mesh = m;
}
const blockRGB = id => BLOCKS[id].map(c => c/255);
function codeRGB(code){ // stable pseudo-random color per code id
  const h = (code * 0.6180339887) % 1; return new THREE.Color().setHSL(h, 0.55, 0.55).toArray(); }

let cur = null, mode = 'recon';
const LEGENDS = {
  recon: '',
  errors: `<span><i class="sw" style="background:var(--good)"></i>correct</span><span><i class="sw" style="background:var(--warn)"></i>wrong block</span><span><i class="sw" style="background:var(--bad)"></i>missing</span><span><i class="sw" style="background:var(--extra)"></i>extra</span>`,
  codes: '<span>each color = one of 1,000 learned codes; each code covers a 4×4×4 chunk</span>',
};
function drawRight(){
  const {orig, pred, codes} = cur;
  document.getElementById('legend').innerHTML = LEGENDS[mode];
  if (mode === 'recon') {
    document.getElementById('rlabel').textContent = 'Reconstruction';
    setMesh(Rv, surface(pred), (y,z,x) => blockRGB(pred[idx(y,z,x)]));
  } else if (mode === 'errors') {
    document.getElementById('rlabel').textContent = 'Errors (vs original)';
    const union = new Uint8Array(R*R*R);
    for (let i=0;i<union.length;i++) union[i] = (orig[i]||pred[i]) ? 1 : 0;
    setMesh(Rv, surface(union), (y,z,x) => { const i = idx(y,z,x), o = orig[i], p = pred[i];
      return !p ? ERR.missing : !o ? ERR.extra : o === p ? ERR.ok : ERR.wrong; });
  } else {
    document.getElementById('rlabel').textContent = 'What the model stores: 16×16×16 codes';
    // show the code of every 4x4x4 chunk that decodes to something non-empty
    setMesh(Rv, surface(pred), (y,z,x) => codeRGB(codes[((y>>2)*16+(z>>2))*16+(x>>2)] | (codes2[((y>>2)*16+(z>>2))*16+(x>>2)] << 8)));
  }
}
let codes2 = null;
async function show(i){
  document.querySelectorAll('#list .item').forEach((d,j) => d.classList.toggle('on', j===i));
  const it = ITEMS[i];
  document.getElementById('cap').textContent = it.caption;
  document.getElementById('stats').textContent =
    `shape overlap (IoU) ${(it.iou*100).toFixed(0)}%  ·  correct block where shape matches ${(it.acc*100).toFixed(0)}%  ·  ${it.n_orig.toLocaleString()} → ${it.n_pred.toLocaleString()} blocks  ·  262,144 voxels stored as 4,096 codes`;
  const [orig, pred, lo, hi] = await Promise.all([inflate(it.orig), inflate(it.pred), inflate(it.codes_lo), inflate(it.codes_hi)]);
  cur = {orig, pred, codes: lo}; codes2 = hi;
  setMesh(L, surface(orig), (y,z,x) => blockRGB(orig[idx(y,z,x)]));
  drawRight();
}
const list = document.getElementById('list');
ITEMS.forEach((it,i) => { const d = document.createElement('div'); d.className = 'item';
  d.innerHTML = `<span class="score">${(it.iou*100).toFixed(0)}%</span><span>${it.caption}</span>`;
  d.onclick = () => show(i); list.append(d); });
document.querySelectorAll('#bar button').forEach(b => b.onclick = () => {
  mode = b.dataset.m; document.querySelectorAll('#bar button').forEach(x => x.classList.toggle('on', x===b));
  if (cur) drawRight(); });
function loop(){ L.r.render(L.s, cam); Rv.r.render(Rv.s, cam); requestAnimationFrame(loop); }
loop(); if (ITEMS.length) show(0);
</script></body></html>
"""


def b64gz(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    ck = torch.load(CKPT, map_location="cuda")
    model = VoxelVQVAE(ck["config"]["vocab"]).cuda().eval()
    model.load_state_dict(ck["model"])
    val_file = CKPT.parent / "val_uids.json"
    pinned = set(json.loads(val_file.read_text())) if val_file.exists() else None
    _, val_rows = split(load_manifest(DATA / "manifest.jsonl"), val_uids=pinned)
    ds = VoxelDataset(val_rows[: args.n], DATA / "vox64")

    items = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(ds)):
            it = ds[i]
            x = it["blocks"][None].cuda()
            logits, codes = model(x)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            orig = it["blocks"].numpy().astype(np.uint8)
            codes = codes[0].cpu().numpy().astype(np.uint16)
            po, oo = pred > 0, orig > 0
            both = po & oo
            items.append({
                "caption": it["caption"],
                "iou": float((po & oo).sum() / max((po | oo).sum(), 1)),
                "acc": float((pred[both] == orig[both]).mean()) if both.any() else 0.0,
                "n_orig": int(oo.sum()), "n_pred": int(po.sum()),
                "orig": b64gz(orig), "pred": b64gz(pred),
                "codes_lo": b64gz((codes & 0xFF).astype(np.uint8)),
                "codes_hi": b64gz((codes >> 8).astype(np.uint8)),
            })
    items.sort(key=lambda r: -r["iou"])
    names, rgb = block_vocab()
    mean_iou = np.mean([r["iou"] for r in items])
    header = (f"Step {ck['step']:,} · {len(items)} held-out objects · mean IoU {mean_iou:.0%}. "
              f"Sorted best → worst.")
    html = (PAGE.replace("__ITEMS__", json.dumps(items))
            .replace("__BLOCKS__", json.dumps(rgb.astype(int).tolist()))
            .replace("__HEADER__", header))
    out = DATA / "recon_viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"step {ck['step']}: {len(items)} objects, mean IoU {mean_iou:.3f} -> {out}")


if __name__ == "__main__":
    main()
