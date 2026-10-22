#!/bin/bash
# Worldgen harvest on a rented CPU machine (Amazon Linux 2023), from the unpacked bundle:
#
#   tar xf harvest_bundle.tar && bash scripts/cloud_harvest.sh setup
#   bash scripts/cloud_harvest.sh test                    # 1 seed, 3 villages, ~5 min
#   SEEDS=80 WORKERS=4 bash scripts/cloud_harvest.sh run  # detached; logs in data/worldgen
#   bash scripts/cloud_harvest.sh status
#
# The bundle carries the server jar and eula.txt (accepted by the user) in data/worldgen/server.
set -e
cd "$(dirname "$0")/.."
PY=python3.11

case "$1" in
setup)
    sudo dnf install -y -q java-25-amazon-corretto-headless $PY $PY-pip
    $PY -m pip install -q --user numpy nbtlib
    java -version
    ;;
test)
    $PY scripts/harvest_worldgen.py --seeds 1 --seed-start 9999 --villages-per-seed 3 \
        --probe-grid 4 --keep-world --xmx 8G
    ;;
run)
    WORKERS=${WORKERS:-4}
    SEEDS=${SEEDS:-80}
    THREADS=$(( $(nproc) / WORKERS ))
    MEM_GB=$(( $(free -g | awk '/Mem:/ {print $2}') * 3 / 4 / WORKERS ))
    for w in $(seq 0 $((WORKERS - 1))); do
        nohup $PY scripts/harvest_worldgen.py --seeds "$SEEDS" --workers "$WORKERS" --worker "$w" \
            --threads "$THREADS" --xmx "${MEM_GB}G" > "data/worldgen/worker$w.out" 2>&1 &
    done
    echo "started $WORKERS workers, $THREADS threads and ${MEM_GB}G each"
    ;;
status)
    tail -n 3 data/worldgen/harvest*.log
    echo "villages: $(ls data/worldgen/villages | wc -l), seeds done: $(cat data/worldgen/manifest_seeds*.txt 2>/dev/null | wc -l)"
    pgrep -fc harvest_worldgen.py || true
    ;;
*)
    echo "usage: $0 setup|test|run|status"; exit 1
    ;;
esac
