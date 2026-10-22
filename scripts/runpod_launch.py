"""One command from a funded RunPod account to a training run that copies itself home.

    .venv/Scripts/python scripts/runpod_launch.py --script train_flow --d 768 --layers 16 \
        --heads 12 --batch 32 --lr 1.5e-4 --out data/runs/flow_big --reserve 3
    .venv/Scripts/python scripts/runpod_launch.py --pod <id> ...      # continue after a failed step

1. Creates a GPU pod, trying --gpus in order, and keeps the first whose GPU really computes
   (a community L40S once came with a broken GPU: "CUDA unknown error").
2. Uploads the code, the encoded data (--codes) and the tokenizer, each file as 4 parallel byte
   ranges (a single ssh stream from home ran at 0.07-2 MB/s), and checks every file's md5.
3. Installs dependencies (scripts/pod_run.sh setup).
4. Starts scripts/cloud_train.sh with a time budget of (balance - --reserve) / price, so the run
   always ends with money left. When it finishes, the pod stops itself 20 min later.
5. Starts scripts/runpod_watch.py at home (copies samples and log every 10 min, checkpoints
   hourly, deletes the pod after the final copy).
Needs RUNPOD_API_KEY (setx) and ~/.ssh/runpod_ed25519 registered on the account.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from runpod_watch import SSH, env, log, opts, runpodctl, ssh_target, use_pod  # noqa: E402

IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPUS = ["COMMUNITY:NVIDIA GeForce RTX 4090", "COMMUNITY:NVIDIA L40S", "SECURE:NVIDIA GeForce RTX 4090",
        "SECURE:NVIDIA L40S", "COMMUNITY:NVIDIA A100-SXM4-80GB", "SECURE:NVIDIA A100-SXM4-80GB"]
REMOTE = "/workspace/TTMC"


def ssh(ip: str, port: int, cmd: str, timeout: int = 1800, stdin: bytes | None = None) -> str:
    # stdin must be closed (DEVNULL) or ssh can keep waiting after the remote command has ended
    r = subprocess.run([SSH, *opts(port), f"root@{ip}", cmd], input=stdin, capture_output=True, timeout=timeout,
                       **({} if stdin is not None else {"stdin": subprocess.DEVNULL}))
    if r.returncode != 0:
        raise RuntimeError(f"`{cmd[:80]}` failed: {r.stderr.decode(errors='replace')[-400:]}")
    return r.stdout.decode(errors="replace")


def our_pods() -> list[dict]:
    return [p for p in json.loads(runpodctl("pod", "list")) if p.get("name") == "ttmc-train"]


def create_pod(gpus: list[str], disk: int) -> str:
    for spec in gpus:
        cloud, gpu = spec.split(":", 1)
        args = ["pod", "create", "--name", "ttmc-train", "--cloud-type", cloud, "--gpu-id", gpu,
                "--image", IMAGE, "--container-disk-in-gb", "30", "--volume-in-gb", str(disk),
                "--ports", "22/tcp", "--min-cuda-version", "12.8"]
        if cloud == "COMMUNITY":
            args.append("--public-ip")
        try:
            runpodctl(*args)
        except RuntimeError as e:
            log(f"{spec}: {str(e)[:120]}")
        time.sleep(5)
        pods = our_pods()
        if not pods:
            continue
        pod = pods[0]["id"]
        use_pod(pod)
        log(f"created {spec} pod {pod} at ${pods[0].get('costPerHr')}/h")
        if gpu_works(pod):
            return pod
        log(f"pod {pod}: GPU does not work; deleting it and trying the next option")
        runpodctl("pod", "delete", pod)
        time.sleep(5)
    raise SystemExit("no working GPU pod available right now; try again later")


def gpu_works(pod: str) -> bool:
    for _ in range(40):                                   # ssh comes up a minute or two after create
        try:
            ip, port = ssh_target(pod)
            out = ssh(ip, port, "python -c \"import torch; x = torch.randn(2048, 2048, device='cuda'); "
                                "print(torch.cuda.get_device_name(0), float((x @ x).std()))\"", timeout=180)
            log(f"GPU check: {out.strip().splitlines()[-1]}")
            return True
        except Exception as e:  # noqa: BLE001
            last = str(e)
            if "CUDA" in last:
                log(f"GPU check failed: {last[-200:]}")
                return False
            time.sleep(15)
    return False


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def send(ip: str, port: int, local: Path, remote: str, parts: int = 4) -> None:
    """Upload as `parts` parallel byte ranges, join on the pod, verify md5."""
    size = local.stat().st_size
    chunk = -(-size // parts)
    t = time.time()

    def part(i: int) -> None:
        off, n = i * chunk, max(0, min(chunk, size - i * chunk))
        for attempt in range(3):
            try:
                p = subprocess.Popen([SSH, *opts(port), f"root@{ip}", f"cat > {remote}.part{i}"],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                with open(local, "rb") as f:
                    f.seek(off)
                    left = n
                    while left:
                        b = f.read(min(left, 1 << 22))
                        p.stdin.write(b)
                        left -= len(b)
                p.stdin.close()
                if p.wait() == 0:
                    return
            except Exception as e:  # noqa: BLE001
                log(f"part {i} attempt {attempt + 1}: {e}")
            time.sleep(5)
        raise RuntimeError(f"upload of {local.name} part {i} failed")

    threads = [threading.Thread(target=part, args=(i,)) for i in range(parts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    names = " ".join(f"{remote}.part{i}" for i in range(parts))
    remote_md5 = ssh(ip, port, f"mkdir -p $(dirname {remote}) && cat {names} > {remote} && rm {names} "
                               f"&& md5sum {remote} | cut -c1-32").strip()
    if remote_md5 != md5(local):
        raise RuntimeError(f"{local.name}: md5 mismatch after upload")
    log(f"uploaded {local.name} ({size / 1e9:.2f} GB) in {time.time() - t:.0f} s, md5 ok")


def code_tar() -> Path:
    path = ROOT / "data" / "pod_code.tar"
    with tarfile.open(path, "w") as t:
        for p in sorted((ROOT / "ttmc").glob("*.py")) + [ROOT / "ttmc" / "blocks.json", ROOT / "requirements.txt"] \
                + sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "scripts").glob("*.sh")):
            t.add(p, p.relative_to(ROOT).as_posix())
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="", help="use this existing pod (skip creation)")
    ap.add_argument("--gpus", nargs="*", default=GPUS, help="CLOUD:gpu id, in order of preference")
    ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--codes", default="data/codes32b")
    ap.add_argument("--tokenizer", default="data/runs/vqvae32b/last.pt")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--script", default="train_flow", choices=("train_flow", "train_t2v"))
    ap.add_argument("--out", default="data/runs/flow_big")
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=10000)
    ap.add_argument("--reserve", type=float, default=3.0, help="dollars to leave on the account")
    ap.add_argument("--hours", type=float, default=0, help="override the budget-derived hours")
    ap.add_argument("--compare-to", default="data/runs/flow_small_b12",
                    help="the watcher stops training if the run falls behind this run's val curve ('' = off)")
    args = ap.parse_args()

    pod = args.pod or create_pod(args.gpus, args.disk)
    use_pod(pod)
    ip, port = ssh_target(pod)
    log(f"pod {pod}: ssh root@{ip} -p {port}")
    if not args.skip_upload:
        ssh(ip, port, f"mkdir -p {REMOTE}/data/runs/vqvae32b {REMOTE}/{args.codes}")
        send(ip, port, code_tar(), f"{REMOTE}/pod_code.tar")
        ssh(ip, port, f"cd {REMOTE} && tar xf pod_code.tar --no-same-owner && rm pod_code.tar "
                      f"&& sed -i 's/\\r$//' scripts/*.sh")
        send(ip, port, ROOT / args.tokenizer, f"{REMOTE}/{args.tokenizer}")
        send(ip, port, ROOT / args.codes / "meta.json", f"{REMOTE}/{args.codes}/meta.json")
        send(ip, port, ROOT / args.codes / "codes.npy", f"{REMOTE}/{args.codes}/codes.npy")
        log("setup (dependencies, CLIP)...")
        print(ssh(ip, port, f"cd {REMOTE} && bash scripts/pod_run.sh setup 2>&1 | tail -3", timeout=3600))

    me = json.loads(runpodctl("user"))
    price = float(next(p for p in json.loads(runpodctl("pod", "list")) if p["id"] == pod)["costPerHr"])
    hours = args.hours or round((float(me["clientBalance"]) - args.reserve) / (price + 0.02) - 0.5, 2)
    if hours <= 0.2:
        raise SystemExit(f"balance ${me['clientBalance']:.2f} leaves no training time after the ${args.reserve} reserve")
    log(f"balance ${float(me['clientBalance']):.2f}, pod ${price}/h -> training for {hours} h")
    # best.pt = EMA weights only (what gets exported): ~0.8 GB instead of ~3.4 GB to copy home
    extra = f"--ema {args.ema}" + (f" --heads {args.heads} --best-weights-only" if args.script == "train_flow" else "")
    env_vars = (f"SCRIPT=scripts/{args.script}.py HOURS={hours} LAYERS={args.layers} D={args.d} "
                f"BATCH={args.batch} LR={args.lr} WARMUP={args.warmup} STEPS=100000000 OUT={args.out} "
                f"EVAL_EVERY={args.eval_every} SAMPLE_EVERY={args.sample_every} EXTRA='{extra}'")
    ssh(ip, port, f"cd {REMOTE} && {env_vars} bash scripts/cloud_train.sh < /dev/null > /dev/null 2>&1")
    time.sleep(90)
    print(ssh(ip, port, f"grep -E '^(objects|params|step|eval)|Error|Traceback' {REMOTE}/data/train_t2v_cloud.log | tail -5"))

    watch_log = open(ROOT / "data" / "runpod_watch.log", "a")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    watch = [sys.executable, str(ROOT / "scripts" / "runpod_watch.py"), pod, "--out", args.out]
    if args.compare_to:
        watch += ["--compare-to", args.compare_to]
    subprocess.Popen(watch,
                     cwd=ROOT, stdout=watch_log, stderr=subprocess.STDOUT, creationflags=flags, env=env())
    log(f"home watcher started for pod {pod}; results arrive in {args.out}/")


if __name__ == "__main__":
    main()
