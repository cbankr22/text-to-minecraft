#!/usr/bin/env bash
# TTMC 1.0 on one rented GPU pod, run from /workspace/TTMC (unpacked pod bundle + block grids):
#
#   bash scripts/pod_run.sh setup              # venv on the image's torch, deps, CLIP cache
#   bash scripts/pod_run.sh tokenizer STEPS    # tokenizer v2 (detached) -> data/runs/vqvae32b
#   bash scripts/pod_run.sh compare            # old vs v2 tokenizer on the same held-out objects
#   TOK=data/runs/vqvae32b/last.pt bash scripts/pod_run.sh encode   # codes, CC0 + CC-BY only
#   HOURS=40 LAYERS=16 D=768 BATCH=64 OUT=data/runs/t2v_big bash scripts/cloud_train.sh
#
# Everything lives on /workspace (survives a pod stop). Checkpoints are copied home by
# scripts/runpod_watch.py on the home PC.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
setup)
  python -m venv --system-site-packages .venv
  source .venv/bin/activate
  python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -q numpy trimesh pillow scipy pandas matplotlib transformers huggingface_hub
  python - <<'PY'
from transformers import CLIPTextModel, CLIPTokenizer
from ttmc.t2v import CLIP_NAME
CLIPTokenizer.from_pretrained(CLIP_NAME); CLIPTextModel.from_pretrained(CLIP_NAME)
import torch
print("CLIP cached | torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
PY
  echo "block grids: $(ls data/objaverse/vox32 | wc -l)"; df -h /workspace | tail -1
  ;;
tokenizer)
  source .venv/bin/activate
  nohup python scripts/train_vqvae.py --res 32 --levels 8,8,8,6,5 --channels 64,128,256 \
    --batch 32 --workers 8 --lr 3e-4 --steps "${2:-40000}" --eval-every 2000 \
    --licenses by,cc0 --fp32 --out data/runs/vqvae32b > data/train_vqvae32b.log 2>&1 &
  echo "tokenizer training in background: tail -f data/train_vqvae32b.log"
  ;;
compare)
  source .venv/bin/activate
  python scripts/compare_tokenizers.py data/runs/vqvae32/last.pt data/runs/vqvae32b/last.pt \
    --val data/runs/vqvae32b/val_uids.json
  ;;
encode)
  source .venv/bin/activate
  python scripts/encode_codes.py --tokenizer "${TOK:?set TOK to the chosen tokenizer}" \
    --licenses by,cc0 --batch 64 --fresh --fp32
  ;;
*)
  echo "usage: $0 setup|tokenizer [steps]|compare|encode"; exit 1
  ;;
esac
