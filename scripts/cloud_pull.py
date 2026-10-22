"""Copy ingest results home from the cloud machine while it runs (see scripts/cloud_ingest.sh).

    .venv/Scripts/python scripts/cloud_pull.py ec2-user@<ip>

Every 10 minutes: new pack_*.tar files and the latest manifest.jsonl go to
data/objaverse/cloud_packs/. Light (network only), so it can run while the PC is in use.
Stops when the machine reports DONE (after one last pull) or has been unreachable for 30 min.
Unpack at the end with --unpack: extracts every pack into data/objaverse/vox32 and swaps in
the cloud manifest (a superset of the local one; the old one is kept as manifest.local.jsonl).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "objaverse"
PACKS = DATA / "cloud_packs"
REMOTE = "TTMC/data/packs"
KEY = Path.home() / ".ssh" / "ttmc.pem"
GIT = Path(r"C:\Program Files\Git\usr\bin")
SSH = str(GIT / "ssh.exe") if (GIT / "ssh.exe").exists() else "ssh"
SCP = str(GIT / "scp.exe") if (GIT / "scp.exe").exists() else "scp"
# own known_hosts file: spot machines come and go, and a reused IP must not block the next one
KNOWN_HOSTS = ROOT / "data" / "ingest_known_hosts"
OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30", "-o", f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pull_once(host: str) -> bool | None:
    """True if the remote build is done, False if not, None if unreachable."""
    r = subprocess.run([SSH, *OPTS, host, f"ls {REMOTE}"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    names = r.stdout.split()
    new = [n for n in names if n.startswith("pack_") and n.endswith(".tar") and not (PACKS / n).exists()]
    for n in new + ["manifest.jsonl"]:
        tmp = PACKS / f".{n}.part"
        subprocess.run([SCP, *OPTS, f"{host}:{REMOTE}/{n}", str(tmp)], check=True, timeout=3600)
        os.replace(tmp, PACKS / n)
    lines = sum(1 for _ in open(PACKS / "manifest.jsonl", encoding="utf-8"))
    log(f"pulled {len(new)} packs, manifest {lines} rows")
    return "DONE" in names


def unpack() -> None:
    packs = sorted(PACKS.glob("pack_*.tar"))
    for p in packs:                      # later packs overwrite earlier partial writes
        with tarfile.open(p) as t:
            t.extractall(DATA, filter="data")
    cloud = PACKS / "manifest.jsonl"
    local = DATA / "manifest.jsonl"
    n_cloud = sum(1 for _ in open(cloud, encoding="utf-8"))
    n_local = sum(1 for _ in open(local, encoding="utf-8"))
    if n_cloud < n_local:
        sys.exit(f"cloud manifest has fewer rows ({n_cloud}) than local ({n_local}); not swapping")
    shutil.copy2(local, DATA / "manifest.local.jsonl")
    shutil.copy2(cloud, local)
    print(f"extracted {len(packs)} packs; manifest {n_local} -> {n_cloud} rows")


def main() -> None:
    if sys.argv[1:] == ["--unpack"]:
        return unpack()
    host = sys.argv[1]
    PACKS.mkdir(parents=True, exist_ok=True)
    last_ok = time.time()
    while True:
        try:
            done = pull_once(host)
        except Exception as e:  # noqa: BLE001  (a failed copy is retried next round)
            log(f"pull failed: {type(e).__name__}: {e}")
            done = None
        if done is None:
            log("machine unreachable")
            if time.time() - last_ok > 1800:
                log("unreachable for 30 min; stopping (spot reclaim or terminated?)")
                break
        else:
            last_ok = time.time()
            if done:
                log("remote build DONE; final pull complete")
                break
        time.sleep(600)


if __name__ == "__main__":
    main()
