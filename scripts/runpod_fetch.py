"""Get a finished run off a stopped pod whose host GPU is busy, then delete the pod.

    .venv/Scripts/python scripts/runpod_fetch.py <pod id> --out data/runs/flow_big

Tries `runpodctl pod start` every --every seconds (a stopped pod can only restart on its own
host, and only when that host has a free GPU). Once it runs: copies log, samples, best.pt and
last.pt (replacing the local copies in place: the home disk is nearly full), checks the sizes
against the pod's, and deletes the pod. Log: data/runpod_fetch.log
"""
from __future__ import annotations

import argparse
import subprocess
import time

from runpod_watch import ROOT, REMOTE_ROOT, SSH, log, opts, pull, runpodctl, ssh_target, use_pod


def remote_sizes(ip: str, port: int, out: str) -> dict[str, int]:
    r = subprocess.run([SSH, *opts(port), f"root@{ip}", f"cd {REMOTE_ROOT}/{out} && stat -c '%n %s' *.pt"],
                       capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL, check=True)
    return {n: int(s) for n, s in (line.split() for line in r.stdout.splitlines() if len(line.split()) == 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod")
    ap.add_argument("--out", default="data/runs/flow_big")
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--hours", type=float, default=72, help="give up after this long")
    args = ap.parse_args()
    use_pod(args.pod)
    t0 = time.time()
    while True:
        try:
            runpodctl("pod", "start", args.pod)
            log("pod started")
            break
        except RuntimeError as e:
            if "running" in str(e).lower():
                break
            log(f"not startable yet: {str(e)[:120]}")
        if time.time() - t0 > args.hours * 3600:
            log("giving up")
            return
        time.sleep(args.every)
    for attempt in range(40):                       # ssh comes up a minute or two after start
        try:
            ip, port = ssh_target(args.pod)
            pull(ip, port, args.out, 0, final=True, margin=3e8)
            want = remote_sizes(ip, port, args.out)
            have = {n: (ROOT / args.out / n).stat().st_size for n in want if (ROOT / args.out / n).exists()}
            if have == want:
                log(f"verified {want}; deleting pod")
                runpodctl("pod", "delete", args.pod)
                log(f"deleted pod {args.pod}")
                return
            log(f"incomplete copy: have {have}, pod has {want}; retrying")
        except Exception as e:  # noqa: BLE001
            log(f"attempt {attempt + 1}: {str(e)[:200]}")
        time.sleep(30)
    runpodctl("pod", "stop", args.pod)               # don't leave the GPU billing; data stays on the pod
    log("could not copy everything; pod stopped again (data still on it)")


if __name__ == "__main__":
    main()
