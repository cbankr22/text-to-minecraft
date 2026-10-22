"""Copy training results home from a RunPod pod while it trains, then delete the pod.

    .venv/Scripts/python scripts/runpod_watch.py <pod id> --out data/runs/t2v_cloud

Every 10 min: log.jsonl and new samples_*.png from /workspace/TTMC/<out> on the pod to the
same folder here; last.pt and best.pt (~2 GB each for the big model) when they changed and at
most once per --ckpt-every (1 h), so the pod never holds the only recent copy of anything.
When the pod reports data/FINISHED (scripts/cloud_train.sh), it does a final copy and deletes
the pod: no GPU or storage billing afterwards. Light (network only); runs while the PC is used.
Needs RUNPOD_API_KEY in the environment (setx) and ~/.ssh/runpod_ed25519 on the account.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNPODCTL = str(ROOT / "tools" / "runpodctl.exe")
KEY = Path.home() / ".ssh" / "runpod_ed25519"
GIT = Path(r"C:\Program Files\Git\usr\bin")
SSH = str(GIT / "ssh.exe") if (GIT / "ssh.exe").exists() else "ssh"
SCP = str(GIT / "scp.exe") if (GIT / "scp.exe").exists() else "scp"
KNOWN_HOSTS = ROOT / "data" / "runpod_known_hosts"   # per pod via use_pod(): pods reuse IPs


def use_pod(pod: str) -> None:
    global KNOWN_HOSTS
    KNOWN_HOSTS = ROOT / "data" / f"runpod_known_hosts_{pod}"
REMOTE_ROOT = "/workspace/TTMC"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def env() -> dict:
    e = dict(os.environ)
    if not e.get("RUNPOD_API_KEY"):          # set with setx after this shell started
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "[Environment]::GetEnvironmentVariable('RUNPOD_API_KEY','User')"],
                           capture_output=True, text=True)
        e["RUNPOD_API_KEY"] = r.stdout.strip()
    return e


def runpodctl(*args: str) -> str:
    r = subprocess.run([RUNPODCTL, *args], capture_output=True, text=True, env=env(), timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:] or r.stdout.strip()[-300:])
    return r.stdout


def find(obj, *keys):
    """First value under any of `keys`, searching nested JSON (the CLI's field names vary)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (str, int)):
                return v
        for v in obj.values():
            hit = find(v, *keys)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find(v, *keys)
            if hit is not None:
                return hit
    return None


def ssh_target(pod: str) -> tuple[str, int]:
    info = json.loads(runpodctl("ssh", "info", pod))
    ip, port = find(info, "ip", "publicIp", "host"), find(info, "port", "publicPort")
    if not ip or not port:
        raise RuntimeError(f"no public ssh address in: {json.dumps(info)[:300]}")
    return str(ip), int(port)


def opts(port: int, scp: bool = False) -> list[str]:
    return ["-i", str(KEY), "-P" if scp else "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}", "-o", "ConnectTimeout=20"]


LAST_COPY: dict[str, tuple[float, str]] = {}      # checkpoint -> (when copied, remote mtime)


def pull(ip: str, port: int, out: str, ckpt_every: float, final: bool = False, margin: float = 2e9) -> bool:
    """Copy new results; True if the pod says training finished."""
    local = ROOT / out
    local.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_ROOT}/{out}"
    r = subprocess.run([SSH, *opts(port), f"root@{ip}",
                        f"cd {remote} && stat -c '%n %Y %s' *; test -f {REMOTE_ROOT}/data/FINISHED && echo __FINISHED__"],
                       capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    if r.returncode not in (0, 1):
        raise RuntimeError(r.stderr.strip()[-300:])
    rows = [line.split() for line in r.stdout.splitlines() if len(line.split()) == 3]
    mtimes = {n: m for n, m, _ in rows}
    sizes = {n: int(s) for n, _, s in rows}
    finished = "__FINISHED__" in r.stdout
    want = [n for n in mtimes if n == "log.jsonl" or n == "tokenizer.pt"
            or (n.startswith("samples_") and not (local / n).exists())]
    for n in ("best.pt", "last.pt"):
        if n not in mtimes:
            continue
        when, seen = LAST_COPY.get(n, (0.0, ""))
        if mtimes[n] != seen and (final or finished or time.time() - when >= ckpt_every):
            want.append(n)
    copied = []
    for n in want:
        tmp = local / f".{n}.part"
        need = sizes.get(n, 0) + margin              # keep this much free on the (nearly full) home disk
        if shutil.disk_usage(local).free < need and (local / n).exists() and n.endswith(".pt"):
            log(f"disk low: replacing the local {n} in place (the pod still has its copy)")
            (local / n).unlink()
        if shutil.disk_usage(local).free < need:
            log(f"disk low: skipping {n} ({sizes.get(n, 0) / 1e9:.1f} GB) this time")
            continue
        subprocess.run([SCP, *opts(port, scp=True), f"root@{ip}:{remote}/{n}", str(tmp)],
                       check=True, timeout=3600)
        os.replace(tmp, local / n)                 # never leaves a half-copied checkpoint
        if n in mtimes and n.endswith(".pt"):
            LAST_COPY[n] = (time.time(), mtimes[n])
        copied.append(n)
    log(f"copied {', '.join(copied) or 'nothing new'}")
    return finished


def behind(out: str, reference: str, until: int, margin: float = 0.03) -> str:
    """Go/no-go: the reason to stop if this run's val loss is more than `margin` above the
    reference run's at the same step on two evals in a row (from step 10k up to `until`)."""
    def curve(path: Path) -> dict[int, float]:
        if not path.exists():
            return {}
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {r["step"]: r["val_loss"] for r in recs if "val_loss" in r}
    ref, run = curve(ROOT / reference / "log.jsonl"), curve(ROOT / out / "log.jsonl")
    worse = [s for s in sorted(run) if 10000 <= s <= until and s in ref and run[s] > ref[s] * (1 + margin)]
    for a, b in zip(worse, worse[1:]):
        if b - a <= 2000 * 1.5:                    # consecutive evals
            return f"step {b}: val {run[b]:.4f} vs reference {ref[b]:.4f}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod")
    ap.add_argument("--out", default="data/runs/t2v_cloud")
    ap.add_argument("--every", type=int, default=600, help="seconds between copies")
    ap.add_argument("--ckpt-every", type=float, default=3600, help="seconds between checkpoint copies")
    ap.add_argument("--keep-pod", action="store_true", help="don't delete the pod at the end")
    ap.add_argument("--compare-to", default="", help="reference run (e.g. data/runs/flow_small_b12): "
                    "stop training if this run's val loss falls behind it at the same step")
    ap.add_argument("--compare-until", type=int, default=40000)
    args = ap.parse_args()
    use_pod(args.pod)
    failures = 0
    stopped = False
    while True:
        try:
            ip, port = ssh_target(args.pod)
            finished = pull(ip, port, args.out, args.ckpt_every)
            failures = 0
            reason = args.compare_to and not stopped and behind(args.out, args.compare_to, args.compare_until)
            if reason:
                log(f"AUTO-STOP: behind the reference run ({reason}); stopping training")
                subprocess.run([SSH, *opts(port), f"root@{ip}", f"touch {REMOTE_ROOT}/data/STOP"],
                               timeout=120, stdin=subprocess.DEVNULL, check=True)
                stopped = True
        except Exception as e:  # noqa: BLE001
            failures += 1
            log(f"copy failed ({failures}): {e}")
            finished = False
            if failures >= 12:                     # ~2 h unreachable: stopped, deleted or out of credit
                log("giving up; check the pod in the RunPod console")
                return
        if finished:
            pull(ip, port, args.out, args.ckpt_every, final=True)   # the checkpoints as of the end
            log("training FINISHED; final copy done")
            if not args.keep_pod:
                runpodctl("pod", "delete", args.pod)
                log(f"deleted pod {args.pod}")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
