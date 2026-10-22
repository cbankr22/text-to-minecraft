"""Worldgen harvester: villages cut out of freshly generated worlds, with piece labels.

    .venv/Scripts/python scripts/harvest_worldgen.py --seeds 40 --accept-eula
    (cloud: scripts/cloud_harvest.sh runs several workers side by side)

For each seed:
1. start a headless vanilla server (default 26.2) with that seed and RCON on 127.0.0.1
2. find villages with `/execute positioned X 0 Z run locate structure #minecraft:village`
   on a grid of probe points (one per village placement cell, 34 chunks apart)
3. `/forceload` a square around each village so its chunks generate fully, wait until
   `/execute if loaded` passes for every chunk, then release them
4. stop the server, read the region files, and save each village as
   data/worldgen/villages/<seed>_<n>.npz (blocks (Y,Z,X) uint16 + full block-state palette)
   plus one line in data/worldgen/manifest.jsonl with every structure piece's template
   name, bounding box and rotation (from the chunk's structures.starts NBT)
5. delete the world (unless --keep-world)

Resumable: seeds already in manifest_seeds.txt are skipped. A STOP file in
data/worldgen ends the run cleanly after the current seed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.world import World, cut_box  # noqa: E402

OUT = ROOT / "data" / "worldgen"
JAR_DIR = OUT / "server"           # shared server jar + eula.txt
VILLAGE_DIR = OUT / "villages"
STOP = OUT / "STOP"
# Per worker (set by configure_worker): several servers can run side by side.
SERVER_DIR = OUT / "server"
MANIFEST = OUT / "manifest.jsonl"
SEEDS_DONE = OUT / "manifest_seeds.txt"
LOG = OUT / "harvest.log"
RCON_PORT = 25575
MC = Path(os.environ.get("APPDATA", "")) / ".minecraft"
LAUNCHER_RUNTIME = Path(r"C:\Program Files (x86)\Minecraft Launcher\runtime")

CELL = 34 * 16           # village placement spacing in blocks
RADIUS = 112             # blocks forceloaded around each village (15x15 = 225 chunks <= 256)
MARGIN_XZ = 32           # context kept around the union of the village pieces
MARGIN_DOWN, MARGIN_UP = 16, 16


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Server setup
# --------------------------------------------------------------------------- #

def version_info(version: str) -> dict:
    return json.loads((MC / "versions" / version / f"{version}.json").read_text())


def find_java(version: str) -> str:
    if os.name != "nt":                      # Linux (cloud): Java from the PATH
        return shutil.which("java") or sys.exit("no java on PATH; pass --java")
    component = version_info(version)["javaVersion"]["component"]
    java = LAUNCHER_RUNTIME / component / "windows-x64" / component / "bin" / "java.exe"
    if not java.exists():
        sys.exit(f"no Java runtime '{component}' at {java}; pass --java")
    return str(java)


def configure_worker(worker: int) -> None:
    """Worker w gets its own server folder, manifest, log and ports."""
    global SERVER_DIR, MANIFEST, SEEDS_DONE, LOG, RCON_PORT
    suffix = f"_w{worker}" if worker else ""
    SERVER_DIR = OUT / f"server{suffix}"
    MANIFEST = OUT / f"manifest{suffix}.jsonl"
    SEEDS_DONE = OUT / f"manifest_seeds{suffix}.txt"
    LOG = OUT / f"harvest{suffix}.log"
    RCON_PORT = 25575 + 10 * worker
    if SERVER_DIR != JAR_DIR:
        SERVER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(JAR_DIR / "eula.txt", SERVER_DIR / "eula.txt")


def ensure_server_jar(version: str) -> Path:
    """Download the official server jar listed in the local version json (sha1-checked)."""
    jar = JAR_DIR / f"server-{version}.jar"
    if jar.exists():
        return jar
    dl = version_info(version)["downloads"]["server"]
    log(f"downloading server {version} ({dl['size'] / 1e6:.0f} MB) from {dl['url']}")
    JAR_DIR.mkdir(parents=True, exist_ok=True)
    tmp = jar.with_suffix(".part")
    with urllib.request.urlopen(dl["url"], timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if hashlib.sha1(tmp.read_bytes()).hexdigest() != dl["sha1"]:
        tmp.unlink()
        sys.exit("server jar sha1 mismatch")
    tmp.rename(jar)
    return jar


def write_properties(seed: int, level: str, rcon_port: int, password: str) -> None:
    props = {
        "level-seed": seed, "level-name": level, "generate-structures": "true",
        "server-ip": "127.0.0.1", "server-port": rcon_port + 1, "online-mode": "false",
        "enable-rcon": "true", "rcon.port": rcon_port, "rcon.password": password,
        "difficulty": "peaceful", "spawn-protection": 0, "max-players": 1,
        "view-distance": 3, "simulation-distance": 3, "sync-chunk-writes": "false",
        "max-tick-time": -1,    # heavy generation must not trip the watchdog
        "pause-when-empty-seconds": -1,   # 1.21.2+: an empty server pauses and stops generating
        "enable-status": "false", "spawn-monsters": "false",
    }
    text = "".join(f"{k}={v}\n" for k, v in props.items())
    (SERVER_DIR / "server.properties").write_text(text)


class Rcon:
    """Minimal Source RCON client (the protocol the vanilla server speaks)."""

    def __init__(self, port: int, password: str, timeout: float = 300):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.next_id = 1
        if self._send(3, password)[0] == -1:
            raise RuntimeError("RCON login failed")

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = self.sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("RCON closed")
            buf += part
        return buf

    def _send(self, kind: int, body: str) -> tuple[int, str]:
        rid, self.next_id = self.next_id, self.next_id + 1
        payload = struct.pack("<ii", rid, kind) + body.encode("utf-8") + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(payload)) + payload)
        length = struct.unpack("<i", self._recv_exact(4))[0]
        data = self._recv_exact(length)
        got_id = struct.unpack("<i", data[:4])[0]
        return got_id, data[8:-2].decode("utf-8", "replace")

    def cmd(self, command: str) -> str:
        return self._send(2, command)[1]

    def close(self) -> None:
        self.sock.close()


def start_server(java: str, jar: Path, xmx: str, threads: int, port: int, password: str,
                 timeout: float = 600) -> tuple[subprocess.Popen, Rcon]:
    out = open(SERVER_DIR / "console.log", "a", encoding="utf-8")
    cmd = [java, f"-Xmx{xmx}", f"-Xms{xmx}"]
    if threads:                              # worldgen thread pool size
        cmd.append(f"-Dmax.bg.threads={threads}")
    proc = subprocess.Popen(cmd + ["-jar", str(jar.resolve()), "nogui"],
                            cwd=SERVER_DIR, stdin=subprocess.PIPE, stdout=out,
                            stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see console.log")
        try:
            return proc, Rcon(port, password)
        except (ConnectionError, OSError):
            time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not open RCON in time")


def stop_server(proc: subprocess.Popen, rcon: Rcon) -> None:
    try:
        rcon.cmd("save-all flush")
        rcon.cmd("stop")
        rcon.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------- #
# Finding and generating villages
# --------------------------------------------------------------------------- #

LOCATE_RE = re.compile(r"\[(-?\d+), ~, (-?\d+)\]")


def locate_villages(rcon: Rcon, grid: int, max_villages: int) -> list[tuple[int, int]]:
    """Nearest village to each probe point on a grid x grid lattice of placement cells."""
    half = grid // 2
    found: list[tuple[int, int]] = []
    for i in range(-half, grid - half):
        for j in range(-half, grid - half):
            px, pz = i * CELL + CELL // 2, j * CELL + CELL // 2
            reply = rcon.cmd(f"execute positioned {px} 0 {pz} run "
                             f"locate structure #minecraft:village")
            m = LOCATE_RE.search(reply)
            if not m:
                continue
            pos = (int(m.group(1)), int(m.group(2)))
            if all(abs(pos[0] - q[0]) > 64 or abs(pos[1] - q[1]) > 64 for q in found):
                found.append(pos)
            if len(found) >= max_villages:
                return found
    return found


def generate_area(rcon: Rcon, cx: int, cz: int, timeout: float) -> bool:
    """Forceload the square around (cx, cz) until every chunk is fully loaded."""
    x0, z0, x1, z1 = cx - RADIUS, cz - RADIUS, cx + RADIUS, cz + RADIUS
    rcon.cmd(f"forceload add {x0} {z0} {x1} {z1}")
    pending = {(x, z) for x in range(x0 >> 4, (x1 >> 4) + 1)
               for z in range(z0 >> 4, (z1 >> 4) + 1)}
    t0 = time.time()
    while pending and time.time() - t0 < timeout:
        for c in sorted(pending):
            reply = rcon.cmd(f"execute if loaded {c[0] * 16} 0 {c[1] * 16}")
            if "passed" not in reply.lower():
                break                    # still generating; check again shortly
            pending.discard(c)
        if pending:
            time.sleep(0.5)
    rcon.cmd(f"forceload remove {x0} {z0} {x1} {z1}")
    return not pending


# --------------------------------------------------------------------------- #
# Reading villages back out of the region files
# --------------------------------------------------------------------------- #

def piece_info(piece) -> dict:
    bb = [int(v) for v in piece["BB"]]
    el = piece.get("pool_element")
    if el is not None:
        template = str(el.get("location") or el.get("feature") or el.get("element_type"))
        projection = str(el.get("projection", ""))
    else:
        template, projection = str(piece.get("Template", piece.get("id", ""))), ""
    rot = piece.get("rotation", piece.get("Rot"))
    return {"template": template.removeprefix("minecraft:"), "bb": bb,
            "rotation": str(rot) if rot is not None else None, "projection": projection,
            "ground_delta": int(piece.get("ground_level_delta", 0))}


def structure_starts(world: World, cx: int, cz: int) -> list[dict]:
    starts = []
    for x in range((cx - RADIUS) >> 4, ((cx + RADIUS) >> 4) + 1):
        for z in range((cz - RADIUS) >> 4, ((cz + RADIUS) >> 4) + 1):
            ch = world.chunk(x, z)
            if ch is None or "structures" not in ch:
                continue
            for name, start in ch["structures"].get("starts", {}).items():
                if str(start.get("id", "INVALID")) == "INVALID":
                    continue
                starts.append({"structure": str(name).removeprefix("minecraft:"),
                               "chunk": [x, z],
                               "pieces": [piece_info(p) for p in start.get("Children", [])]})
    return starts


def extract_village(world: World, seed: int, n: int, cx: int, cz: int) -> dict | None:
    starts = structure_starts(world, cx, cz)
    villages = [s for s in starts if s["structure"].startswith("village")]
    if not villages:
        return None
    # The village whose start chunk is closest to the located position.
    v = min(villages, key=lambda s: abs(s["chunk"][0] * 16 - cx) + abs(s["chunk"][1] * 16 - cz))
    bbs = np.array([p["bb"] for p in v["pieces"]])
    x0 = max(bbs[:, 0].min() - MARGIN_XZ, cx - RADIUS)
    z0 = max(bbs[:, 2].min() - MARGIN_XZ, cz - RADIUS)
    x1 = min(bbs[:, 3].max() + 1 + MARGIN_XZ, cx + RADIUS + 1)
    z1 = min(bbs[:, 5].max() + 1 + MARGIN_XZ, cz + RADIUS + 1)
    y0, y1 = bbs[:, 1].min() - MARGIN_DOWN, bbs[:, 4].max() + 1 + MARGIN_UP
    blocks, palette, missing = cut_box(world, int(x0), int(y0), int(z0),
                                       int(x1), int(y1), int(z1))
    uid = f"{seed}_{n}"
    np.savez_compressed(VILLAGE_DIR / f"{uid}.npz", blocks=blocks, palette=np.array(palette))
    others = [s for s in starts if s is not v
              and any(p["bb"][3] >= x0 and p["bb"][0] < x1 and p["bb"][5] >= z0 and p["bb"][2] < z1
                      for p in s["pieces"])]
    return {"uid": uid, "seed": seed, "structure": v["structure"],
            "origin": [int(x0), int(y0), int(z0)], "shape": list(blocks.shape),
            "missing_chunks": missing, "palette_size": len(palette),
            "pieces": v["pieces"], "other_structures": others}


# --------------------------------------------------------------------------- #

def harvest_seed(seed: int, args, java: str, jar: Path) -> int:
    level = f"world_{seed}"
    world_path = SERVER_DIR / level
    if world_path.exists():
        shutil.rmtree(world_path)        # leftover from an interrupted run
    port = RCON_PORT
    password = secrets.token_hex(12)
    write_properties(seed, level, port, password)
    t0 = time.time()
    proc, rcon = start_server(java, jar, args.xmx, args.threads, port, password)
    log(f"seed {seed}: server up in {time.time() - t0:.0f}s")
    try:
        t0 = time.time()
        villages = locate_villages(rcon, args.probe_grid, args.villages_per_seed)
        log(f"seed {seed}: {len(villages)} villages located in {time.time() - t0:.0f}s")
        done = []
        for cx, cz in villages:
            t0 = time.time()
            ok = generate_area(rcon, cx, cz, args.gen_timeout)
            log(f"  village at {cx},{cz}: {'ok' if ok else 'TIMEOUT'} {time.time() - t0:.0f}s")
            done.append((cx, cz))
            if STOP.exists():
                break
    finally:
        stop_server(proc, rcon)

    world = World(world_path)
    count = 0
    with open(MANIFEST, "a", encoding="utf-8") as mf:
        for n, (cx, cz) in enumerate(done):
            try:
                rec = extract_village(world, seed, n, cx, cz)
            except Exception as e:                  # one bad village must not end the run
                log(f"  extract {seed}_{n} failed: {e!r}")
                continue
            finally:
                world.clear()
            if rec is None:
                log(f"  no village start found near {cx},{cz}")
                continue
            rec["located"] = [cx, cz]
            rec["mc_version"] = args.version
            mf.write(json.dumps(rec) + "\n")
            count += 1
            houses = sum("/houses/" in p["template"] for p in rec["pieces"])
            log(f"  {rec['uid']} {rec['structure']} shape {rec['shape']} "
                f"{len(rec['pieces'])} pieces ({houses} houses), missing {rec['missing_chunks']}")
    if not args.keep_world:
        shutil.rmtree(world_path, ignore_errors=True)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds to harvest")
    ap.add_argument("--seed-start", type=int, default=1, help="seeds are seed-start, +1, ...")
    ap.add_argument("--villages-per-seed", type=int, default=30)
    ap.add_argument("--probe-grid", type=int, default=10, help="probe points per side")
    ap.add_argument("--version", default="26.2")
    ap.add_argument("--java", default=None)
    ap.add_argument("--xmx", default="3G")
    ap.add_argument("--gen-timeout", type=float, default=180)
    ap.add_argument("--threads", type=int, default=0, help="worldgen threads (0 = Java default)")
    ap.add_argument("--workers", type=int, default=1, help="servers running side by side")
    ap.add_argument("--worker", type=int, default=0,
                    help="this process's index; it takes the seeds with seed %% workers == worker")
    ap.add_argument("--keep-world", action="store_true")
    ap.add_argument("--accept-eula", action="store_true",
                    help="you agree to the Minecraft EULA (https://aka.ms/MinecraftEULA)")
    args = ap.parse_args()

    VILLAGE_DIR.mkdir(parents=True, exist_ok=True)
    eula = JAR_DIR / "eula.txt"
    if not (eula.exists() and "eula=true" in eula.read_text()):
        if not args.accept_eula:
            sys.exit("pass --accept-eula to agree to the Minecraft EULA")
        JAR_DIR.mkdir(parents=True, exist_ok=True)
        eula.write_text("eula=true\n")
    configure_worker(args.worker)
    java = args.java or find_java(args.version)
    jar = ensure_server_jar(args.version)
    done = set(SEEDS_DONE.read_text().split()) if SEEDS_DONE.exists() else set()

    total = 0
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if str(seed) in done or seed % args.workers != args.worker:
            continue
        if STOP.exists():
            log("STOP file found; ending")
            break
        t0 = time.time()
        n = harvest_seed(seed, args, java, jar)
        total += n
        with open(SEEDS_DONE, "a") as f:
            f.write(f"{seed}\n")
        log(f"seed {seed}: {n} villages saved in {(time.time() - t0) / 60:.1f} min (run total {total})")


if __name__ == "__main__":
    main()
