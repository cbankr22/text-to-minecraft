"""Keep the AWS ingest alive overnight: launch a spot machine, start the build, pull results
home every 10 min, and launch a new machine when AWS reclaims the old one.

    .venv/Scripts/python scripts/cloud_ingest_watch.py            # detached, log in data/

Each new machine starts from the newest pulled manifest, so finished objects are never redone.
When the build reports DONE it runs one retry pass (objects whose worker crashed or hit the
memory cap), then terminates the machine. Also stops after --max-launches machines or --hours.
Every machine also terminates itself after 10 h (shutdown -h, terminate-on-shutdown).
Needs: AWS CLI configured, key pair `ttmc` (~/.ssh/ttmc.pem), security group id in TTMC_AWS_SG,
huggingfaceaccesstoken.txt in the project root (copied to the machine, never bundled).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from cloud_pull import KNOWN_HOSTS, OPTS, PACKS, SCP, SSH, log, pull_once  # noqa: E402

REGION = os.environ.get("TTMC_AWS_REGION", "us-east-1")
SG = os.environ.get("TTMC_AWS_SG", "")          # security group id that allows ssh from your IP
# all 32 vCPU; the type AWS last reclaimed moves to the back (its spot pool is under pressure)
TYPES = ["m7a.8xlarge", "c6a.8xlarge", "c7i.8xlarge", "c6i.8xlarge", "c7a.8xlarge"]
AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
NAME = "ttmc-ingest"
BUNDLE = ROOT / "data" / "ingest_bundle.tar"


def aws(*args: str) -> str:
    r = subprocess.run(["aws", "--region", REGION, *args, "--output", "json"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-400:])
    return r.stdout


def live_instance() -> tuple[str, str] | None:
    """(id, public ip) of a pending/running ttmc-ingest machine, if any."""
    out = json.loads(aws("ec2", "describe-instances", "--filters", f"Name=tag:Name,Values={NAME}",
                         "Name=instance-state-name,Values=pending,running"))
    for res in out["Reservations"]:
        for inst in res["Instances"]:
            return inst["InstanceId"], inst.get("PublicIpAddress", "")
    return None


def launch() -> str:
    ami = json.loads(aws("ssm", "get-parameter", "--name", AMI_PARAM))["Parameter"]["Value"]
    for itype in TYPES:
        try:
            out = json.loads(aws(
                "ec2", "run-instances", "--image-id", ami, "--instance-type", itype,
                "--key-name", "ttmc", "--security-group-ids", SG,
                "--instance-market-options",
                "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}",
                "--instance-initiated-shutdown-behavior", "terminate",
                "--block-device-mappings",
                "DeviceName=/dev/xvda,Ebs={VolumeSize=80,VolumeType=gp3,DeleteOnTermination=true}",
                "--tag-specifications", f"ResourceType=instance,Tags=[{{Key=Name,Value={NAME}}}]"))
            iid = out["Instances"][0]["InstanceId"]
            log(f"launched {itype} {iid}")
            TYPES.remove(itype)
            TYPES.append(itype)                 # if this one is reclaimed, try the others first
            return iid
        except RuntimeError as e:
            log(f"{itype}: {e}")
    raise RuntimeError("no spot capacity for any instance type")


def ssh(host: str, cmd: str, timeout: int = 900, stdin: str | None = None) -> str:
    r = subprocess.run([SSH, *OPTS, host, cmd], input=stdin, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def make_bundle() -> None:
    """Code + newest pulled manifest (as data/objaverse/manifest.jsonl). Never the token."""
    manifest = PACKS / "manifest.jsonl"
    if not manifest.exists():
        manifest = ROOT / "data" / "objaverse" / "manifest.jsonl"
    with tarfile.open(BUNDLE, "w") as t:
        for p in sorted((ROOT / "ttmc").glob("*.py")) + [ROOT / "ttmc" / "blocks.json",
                                                          ROOT / "scripts" / "objaverse_build.py",
                                                          ROOT / "scripts" / "cloud_ingest.sh"]:
            t.add(p, p.relative_to(ROOT).as_posix())
        t.add(manifest, "data/objaverse/manifest.jsonl")


def start(iid: str) -> str:
    """Wait for the machine, upload, set up, start the build. Returns ec2-user@ip."""
    subprocess.run(["aws", "--region", REGION, "ec2", "wait", "instance-running", "--instance-ids", iid],
                   check=True, timeout=600)
    ip = live_instance()[1]
    host = f"ec2-user@{ip}"
    KNOWN_HOSTS.unlink(missing_ok=True)                  # a brand-new machine, new host key
    for _ in range(30):
        try:
            ssh(host, "true", timeout=30)
            break
        except Exception:  # noqa: BLE001
            time.sleep(10)
    ssh(host, "sudo shutdown -h +600; mkdir -p TTMC")
    make_bundle()
    subprocess.run([SCP, *OPTS, str(BUNDLE), f"{host}:TTMC/"], check=True, timeout=600)
    token = (ROOT / "huggingfaceaccesstoken.txt").read_text().strip()
    ssh(host, "umask 077; cat > ~/.hf_token", stdin=token)
    ssh(host, "cd TTMC && tar xf ingest_bundle.tar && bash scripts/cloud_ingest.sh setup", timeout=1200)
    ssh(host, "cd TTMC && bash scripts/cloud_ingest.sh run")
    log(f"build running on {host}")
    return host


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=14)
    ap.add_argument("--max-launches", type=int, default=5)
    args = ap.parse_args()
    PACKS.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    launches = 0
    retried = False
    host = None
    cur = live_instance()
    if cur:
        host = f"ec2-user@{cur[1]}"
        log(f"watching existing machine {cur[0]} at {host}")
    while time.time() < deadline:
        if host is None:
            if launches >= args.max_launches:
                log("launch limit reached; stopping")
                return
            try:
                host = start(launch())
                launches += 1
            except Exception as e:  # noqa: BLE001
                log(f"launch failed: {e}; retrying in 10 min")
                host = None
                time.sleep(600)
                continue
        try:
            done = pull_once(host)
        except Exception as e:  # noqa: BLE001
            log(f"pull failed: {e}")
            done = None
        if done and not retried:
            # objects that failed from a crashed or memory-capped worker get one more try
            log("build DONE; starting the retry pass")
            try:
                ssh(host, "cd TTMC && rm -f data/packs/DONE && bash scripts/cloud_ingest.sh run")
                retried = True
            except Exception as e:  # noqa: BLE001
                log(f"retry start failed: {e}")
            time.sleep(600)
            continue
        if done:
            log("retry pass DONE; final pull complete; terminating the machine")
            cur = live_instance()
            if cur:
                aws("ec2", "terminate-instances", "--instance-ids", cur[0])
            return
        if done is None and live_instance() is None:
            log("machine is gone (spot reclaim?); launching a new one")
            host = None
            continue
        time.sleep(600)
    log("time limit reached; stopping (the machine terminates itself)")


if __name__ == "__main__":
    main()
