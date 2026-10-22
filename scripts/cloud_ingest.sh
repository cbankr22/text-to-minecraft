#!/bin/bash
# Full Objaverse ingest on a rented CPU machine (Amazon Linux 2023), from the unpacked bundle:
#
#   tar xf ingest_bundle.tar && bash scripts/cloud_ingest.sh setup
#   bash scripts/cloud_ingest.sh test     # ~300 objects, checks downloads + voxelizing
#   bash scripts/cloud_ingest.sh run      # detached build + packer; logs in data/
#   bash scripts/cloud_ingest.sh status
#
# The HF token is read from ~/.hf_token (copied over separately, never in the bundle).
# Results leave the machine through data/packs/: every 10 min the packer tars the new
# vox32 files and snapshots manifest.jsonl there, and scripts/cloud_pull.py copies them home.
# The build is resumable by manifest, so after a spot reclaim: new machine, newest manifest, run.
set -e
cd "$(dirname "$0")/.."
PY=python3.11
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null)
LICENSES=${LICENSES:-by,cc0}
# 30 workers froze a 32-core/61 GB machine (huge meshes at once): 16 workers, each capped at
# 24 GB of address space (3.5 GB failed almost every object, 10 GB about 7%: numpy and trimesh
# reserve far more address space than they use; actual use is ~1 GB per worker)
WORKERS=${WORKERS:-16}
MEM_GB=${MEM_GB:-24}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$1" in
setup)
    sudo dnf install -y -q $PY $PY-pip tar
    $PY -m pip install -q --user numpy trimesh pillow scipy pandas objaverse networkx
    $PY -c "import trimesh, objaverse; print('deps ok')"
    ;;
test)
    n=$(( $(wc -l < data/objaverse/manifest.jsonl) + 300 ))
    $PY scripts/objaverse_build.py --n $n --groups 160 --save 32 --licenses "$LICENSES" \
        --workers "$WORKERS" --mem-gb "$MEM_GB" --threads 32 --batch 150
    ;;
run)
    mkdir -p data/packs
    nohup $PY scripts/objaverse_build.py --n 1000000 --groups 160 --save 32 --licenses "$LICENSES" \
        --workers "$WORKERS" --mem-gb "$MEM_GB" --threads 48 --batch 1000 > data/ingest.log 2>&1 &
    nohup bash scripts/cloud_ingest.sh pack-loop > data/packer.log 2>&1 &
    echo "started build ($WORKERS workers) and packer"
    ;;
pack-loop)
    [ -f data/packs/.marker ] || touch -d @0 data/packs/.marker   # a rerun packs only new files
    while true; do
        running=$(pgrep -fc objaverse_build.py || true)
        bash scripts/cloud_ingest.sh pack || echo "pack failed; retrying next round"
        if [ "$running" = "0" ]; then
            touch data/packs/DONE; echo "build finished; final pack written"; break
        fi
        sleep 600
    done
    ;;
pack)
    mkdir -p data/objaverse/vox32          # the build may not have created it yet
    touch data/packs/.next
    ts=$(date +%Y%m%d_%H%M%S)
    (cd data/objaverse && find vox32 -name '*.npz' -newer ../packs/.marker ! -newer ../packs/.next) \
        > data/packs/.list
    if [ -s data/packs/.list ]; then
        tar cf "data/packs/.pack_$ts.tar" -C data/objaverse -T data/packs/.list
        mv "data/packs/.pack_$ts.tar" "data/packs/pack_$ts.tar"
    fi
    cp data/objaverse/manifest.jsonl data/packs/.manifest.jsonl
    mv data/packs/.manifest.jsonl data/packs/manifest.jsonl
    mv data/packs/.next data/packs/.marker
    echo "$ts packed $(wc -l < data/packs/.list) files, manifest $(wc -l < data/packs/manifest.jsonl)"
    ;;
status)
    tail -n 2 data/ingest.log; tail -n 1 data/packer.log
    echo "manifest $(wc -l < data/objaverse/manifest.jsonl)  vox32 $(ls data/objaverse/vox32 | wc -l)  disk free $(df -h . | awk 'NR==2 {print $4}')"
    ;;
*)
    echo "usage: $0 setup|test|run|status"; exit 1
    ;;
esac
