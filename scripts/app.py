"""Local playground: type a prompt, see the build in 3D, download the .schem.

    .venv/Scripts/python scripts/app.py            # then open http://127.0.0.1:8770

Models load once at startup (~20 s); each generation then takes about a second.
Sizes in the prompt are understood: "a coffee mug, 8 blocks tall", "a small house".
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.resize import parse_size, place, resize  # noqa: E402
from ttmc.schem import write_schem  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

CKPT = ROOT / "data" / "runs" / "t2v_big" / "best.pt"   # 175M, step 208k (old 55M: runs/t2v)
PORT = 8770

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Block Builder</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#141518;--panel:#1d1f24;--line:#2a2d33;--fg:#e6e6e6;--muted:#9aa0a6;--accent:#6aa9ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input[type=text]{flex:1;min-width:220px;background:var(--panel);color:var(--fg);border:1px solid #3a3d44;border-radius:8px;padding:10px 12px;font-size:15px}
button,select{background:var(--panel);color:var(--fg);border:1px solid #3a3d44;padding:9px 14px;border-radius:8px;cursor:pointer;font-size:14px}
button.primary{background:var(--accent);color:#0b1220;border-color:var(--accent);font-weight:600}
button:disabled{opacity:.5;cursor:default}
label{color:var(--muted);font-size:12px;display:flex;align-items:center;gap:6px}
#grid{flex:1;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1px;background:var(--line);min-height:0}
.cell{position:relative;background:var(--bg);min-height:260px}
.cell canvas{display:block;width:100%;height:100%}
.cap{position:absolute;left:10px;top:8px;color:var(--muted);font-size:12px;pointer-events:none}
.dl{position:absolute;right:8px;bottom:8px;font-size:12px;padding:6px 10px}
#status{color:var(--muted);font-size:12px;padding:6px 16px;border-top:1px solid var(--line)}
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:0 16px 10px}
.chip{font-size:12px;padding:5px 10px;border-radius:999px}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<header>
  <input id="prompt" type="text" placeholder="Describe a build… e.g. a wooden chair" value="a green tree with a brown trunk">
  <label>variations <select id="n"><option>1</option><option>2</option><option selected>4</option></select></label>
  <label>guidance <select id="cfg"><option>2</option><option selected>3</option><option>5</option></select></label>
  <button class="primary" id="go">Generate</button>
</header>
<div class="chips" id="chips"></div>
<div id="grid"></div>
<div id="status">Ready.</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const BLOCKS = __BLOCKS__; let R = 32;
const IDEAS = ["a wooden chair","a wooden table","a green tree with a brown trunk","a pumpkin","a coffee mug",
  "a wooden barrel","a windmill","a red dragon","a blue sports car","a sword","a pink flower in a pot","a white snowman"];
const chips = document.getElementById('chips');
IDEAS.forEach(t => { const b = document.createElement('button'); b.className='chip'; b.textContent=t;
  b.onclick = () => { document.getElementById('prompt').value = t; run(); }; chips.append(b); });

const views = [];
const cam = new THREE.PerspectiveCamera(40, 1, 0.1, 500); cam.position.set(R*1.4, R*1.1, R*1.4);
function makeView(el){
  const r = new THREE.WebGLRenderer({antialias:true}); r.setPixelRatio(devicePixelRatio); el.prepend(r.domElement);
  const s = new THREE.Scene(); s.background = new THREE.Color(0x141518);
  s.add(new THREE.HemisphereLight(0xffffff,0x445566,1.6));
  const sun = new THREE.DirectionalLight(0xffffff,1.6); sun.position.set(1,2,1.4); s.add(sun);
  s.add(new THREE.GridHelper(R, 8, 0x333740, 0x25282d));
  const c = new OrbitControls(cam, r.domElement); c.target.set(0, R*0.4, 0); c.update();
  return {el, r, s};
}
function resize(){ for (const v of views){ v.r.setSize(v.el.clientWidth, v.el.clientHeight); }
  if (views[0]) { cam.aspect = views[0].el.clientWidth / views[0].el.clientHeight; cam.updateProjectionMatrix(); } }
addEventListener('resize', resize);
async function inflate(b64){
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}
function addMesh(v, g){
  const idx = (y,z,x) => (y*R+z)*R+x;
  const solid = (y,z,x) => y>=0&&z>=0&&x>=0&&y<R&&z<R&&x<R&&g[idx(y,z,x)]>0;
  const cells = [];
  for (let y=0;y<R;y++) for (let z=0;z<R;z++) for (let x=0;x<R;x++){
    if (!g[idx(y,z,x)]) continue;
    if (solid(y+1,z,x)&&solid(y-1,z,x)&&solid(y,z+1,x)&&solid(y,z-1,x)&&solid(y,z,x+1)&&solid(y,z,x-1)) continue;
    cells.push([y,z,x]); }
  const m = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshLambertMaterial(), Math.max(1,cells.length));
  m.count = cells.length; const mat = new THREE.Matrix4(), col = new THREE.Color();
  cells.forEach(([y,z,x],k) => { mat.makeTranslation(x-R/2+0.5, y+0.5, z-R/2+0.5); m.setMatrixAt(k, mat);
    const c = BLOCKS[g[idx(y,z,x)]]; col.setRGB(c[0]/255,c[1]/255,c[2]/255, THREE.SRGBColorSpace); m.setColorAt(k, col); });
  v.s.add(m);
}
async function run(){
  const prompt = document.getElementById('prompt').value.trim(); if (!prompt) return;
  const go = document.getElementById('go'); go.disabled = true;
  const status = document.getElementById('status'); status.textContent = 'Generating…';
  const t0 = performance.now();
  try {
    const res = await fetch('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt, n:+document.getElementById('n').value, cfg:+document.getElementById('cfg').value})});
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    R = data.res; cam.position.set(R*1.4, R*1.1, R*1.4);
    const grid = document.getElementById('grid'); grid.innerHTML = ''; views.forEach(v => v.r.dispose()); views.length = 0;
    for (const [i, item] of data.results.entries()){
      const cell = document.createElement('div'); cell.className = 'cell';
      cell.innerHTML = `<div class="cap">${prompt} · seed ${item.seed} · ${item.size[2]}×${item.size[0]}×${item.size[1]} (w×h×d) · ${item.n_voxels.toLocaleString()} blocks</div>`;
      const dl = document.createElement('button'); dl.className = 'dl'; dl.textContent = 'Download .schem';
      dl.onclick = () => { const a = document.createElement('a'); a.href = 'data:application/octet-stream;base64,' + item.schem;
        a.download = prompt.replace(/[^a-z0-9]+/gi,'_').toLowerCase() + `_s${item.seed}.schem`; a.click(); };
      cell.append(dl); grid.append(cell);
      const v = makeView(cell); views.push(v); addMesh(v, await inflate(item.blocks));
    }
    resize();
    status.textContent = `Done in ${((performance.now()-t0)/1000).toFixed(1)} s. Drag to rotate (all views move together), scroll to zoom.`;
  } catch (e) { status.textContent = 'Error: ' + e.message; }
  go.disabled = false;
}
document.getElementById('go').onclick = run;
document.getElementById('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
(function loop(){ for (const v of views) v.r.render(v.s, cam); requestAnimationFrame(loop); })();
</script></body></html>
"""


class Engine:
    def __init__(self):
        ck = torch.load(CKPT, map_location="cuda")
        cfg = ck["config"]
        self.model = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).cuda().eval()
        self.model.load_state_dict(ck["model"])
        self.tok = load_tokenizer(cfg["tokenizer"])
        self.text = TextEncoder()
        self.names, self.rgb = block_vocab()
        self.lock = threading.Lock()
        self.step = ck["step"]

    def run(self, prompt: str, n: int, cfg: float) -> tuple[list[dict], int]:
        """Builds (blocks placed in a res^3 canvas for the viewer) and res."""
        prompt, size = parse_size(prompt)
        seeds = list(np.random.randint(0, 1_000_000, n))
        builds = []
        with self.lock:
            for seed in seeds:
                codes = generate(self.model, self.text, [prompt], cfg=cfg, seed=int(seed))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    b = self.tok.decode_indices(codes.view(1, 8, 8, 8)).argmax(1)[0].cpu().numpy().astype(np.uint8)
                builds.append((int(seed), resize(b, size)))
        res = max([32] + [max(b.shape) for _, b in builds])
        out = []
        for seed, b in builds:
            with tempfile.NamedTemporaryFile(suffix=".schem", delete=False) as f:
                path = Path(f.name)
            write_schem(b, self.names, path)
            schem = base64.b64encode(path.read_bytes()).decode()
            path.unlink(missing_ok=True)
            out.append({"seed": seed, "n_voxels": int((b > 0).sum()), "size": list(b.shape),
                        "blocks": base64.b64encode(gzip.compress(place(b, res).tobytes(), 6)).decode(),
                        "schem": schem})
        return out, res


ENGINE: Engine | None = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            blocks = json.dumps(ENGINE.rgb.astype(int).tolist())
            self._send(200, PAGE.replace("__BLOCKS__", blocks).encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = str(req.get("prompt", "")).strip()[:300]
            n = max(1, min(int(req.get("n", 4)), 8))
            cfg = float(req.get("cfg", 3.0))
            results, res = ENGINE.run(prompt, n, cfg)
            self._send(200, json.dumps({"results": results, "res": res}).encode(), "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    global ENGINE
    print("loading models…", flush=True)
    ENGINE = Engine()
    print(f"ready (model step {ENGINE.step}) -> http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
