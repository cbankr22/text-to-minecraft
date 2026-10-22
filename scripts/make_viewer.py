"""Build a self-contained 3D viewer (three.js) for voxelized objects.

    .venv/Scripts/python scripts/make_viewer.py [--n 150] [--rejected]   # -> data/objaverse/viewer.html
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.voxelize import block_vocab  # noqa: E402

DATA = ROOT / "data" / "objaverse"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Voxel Viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#141518;--panel:#1d1f24;--fg:#e6e6e6;--muted:#9aa0a6;--accent:#6aa9ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;height:100vh}
#list{width:300px;overflow:auto;background:var(--panel);border-right:1px solid #2a2d33}
#list div{padding:8px 12px;cursor:pointer;border-bottom:1px solid #26292e;color:var(--muted)}
#list div:hover{color:var(--fg)} #list div.on{color:var(--fg);background:#262a31;border-left:3px solid var(--accent)}
#main{flex:1;position:relative} canvas{display:block}
#info{position:absolute;left:12px;top:12px;right:12px;pointer-events:none}
#info b{font-size:16px} #info span{color:var(--muted)}
#mode{position:absolute;right:12px;bottom:12px}
button{background:var(--panel);color:var(--fg);border:1px solid #333;padding:6px 10px;border-radius:6px;cursor:pointer}
@media (max-width:700px){body{flex-direction:column}#list{width:auto;height:30vh}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="list"></div>
<div id="main"><div id="info"></div><div id="mode"><button id="toggle">Show: blocks</button></div></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const MODELS = __MODELS__;
const BLOCKS = __BLOCKS__;
const R = __RES__;
const main = document.getElementById('main');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio);
main.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x141518);
const cam = new THREE.PerspectiveCamera(40, 1, 0.1, 1000);
cam.position.set(R*1.3, R*1.0, R*1.3);
const ctl = new OrbitControls(cam, renderer.domElement); ctl.target.set(0, R*0.35, 0); ctl.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1,2,1.4); scene.add(sun);
const grid = new THREE.GridHelper(R, 8, 0x333740, 0x25282d); scene.add(grid);
function resize(){const w=main.clientWidth,h=main.clientHeight;renderer.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();}
addEventListener('resize', resize); resize();

async function inflate(b64){
  const bytes = Uint8Array.from(atob(b64), c=>c.charCodeAt(0));
  const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(ds).arrayBuffer());
}
let mesh = null, showBlocks = true, current = 0;
async function show(i){
  current = i;
  document.querySelectorAll('#list div').forEach((d,j)=>d.className = j===i?'on':'');
  const m = MODELS[i];
  document.getElementById('info').innerHTML = `<b>${m.caption}</b><br><span>${m.n_voxels} voxels · ${m.n_block_types} block types · ${m.license||''}</span>`;
  const blocks = await inflate(m.blocks), rgb = await inflate(m.rgb);
  const idx = (y,z,x)=> (y*R+z)*R+x;
  const solid = (y,z,x)=> y>=0&&z>=0&&x>=0&&y<R&&z<R&&x<R&&blocks[idx(y,z,x)]>0;
  const cells = [];
  for(let y=0;y<R;y++)for(let z=0;z<R;z++)for(let x=0;x<R;x++){
    if(!blocks[idx(y,z,x)]) continue;
    if(solid(y+1,z,x)&&solid(y-1,z,x)&&solid(y,z+1,x)&&solid(y,z-1,x)&&solid(y,z,x+1)&&solid(y,z,x-1)) continue;
    cells.push([y,z,x]);
  }
  if(mesh){scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();}
  mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshLambertMaterial(), cells.length);
  const mat = new THREE.Matrix4(), col = new THREE.Color();
  cells.forEach(([y,z,x],k)=>{
    mat.makeTranslation(x-R/2+0.5, y+0.5, z-R/2+0.5); mesh.setMatrixAt(k, mat);
    const id = idx(y,z,x);
    if(showBlocks){const c=BLOCKS[blocks[id]].rgb; col.setRGB(c[0]/255,c[1]/255,c[2]/255, THREE.SRGBColorSpace);}
    else col.setRGB(rgb[id*3]/255, rgb[id*3+1]/255, rgb[id*3+2]/255, THREE.SRGBColorSpace);
    mesh.setColorAt(k, col);
  });
  scene.add(mesh);
}
const list = document.getElementById('list');
MODELS.forEach((m,i)=>{const d=document.createElement('div'); d.textContent=m.caption; d.onclick=()=>show(i); list.append(d);});
document.getElementById('toggle').onclick = e => {showBlocks=!showBlocks; e.target.textContent='Show: '+(showBlocks?'blocks':'true color'); show(current);};
renderer.setAnimationLoop(()=>renderer.render(scene, cam));
if(MODELS.length) show(0);
</script></body></html>
"""


def b64gz(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="random sample size (keeps the page small)")
    ap.add_argument("--rejected", action="store_true", help="show rejected objects instead")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("status") == "ok" and bool(r.get("keep", True)) != args.rejected]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    names, rgb = block_vocab()
    models, res = [], 64
    for r in rows:
        if args.rejected:
            r["caption"] = f"[{', '.join(r['reject'])}] {r['caption']}"
        d = np.load(DATA / "vox64" / f"{r['uid']}.npz")
        res = d["blocks"].shape[0]
        models.append({"caption": r["caption"], "license": r.get("license"),
                       "n_voxels": r["n_voxels"], "n_block_types": r["n_block_types"],
                       "blocks": b64gz(d["blocks"]), "rgb": b64gz(d["rgb"])})
    blocks = [{"name": n, "rgb": c.astype(int).tolist()} for n, c in zip(names, rgb)]
    html = (PAGE.replace("__MODELS__", json.dumps(models))
            .replace("__BLOCKS__", json.dumps(blocks)).replace("__RES__", str(res)))
    out = DATA / "viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(models)} models -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
