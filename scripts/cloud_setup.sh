#!/usr/bin/env bash
# Set up a rented Linux GPU box (RunPod / Vast "PyTorch" image) to train the t2v model.
#
#   bash scripts/cloud_setup.sh              # run inside the unpacked cloud_bundle.tar
#   bash scripts/cloud_setup.sh hf:NAME/ttmc-data   # pull the bundle from a private HF repo
#                                                   # (needs HF_TOKEN in the environment)
#
# Put the project on the persistent volume (/workspace on RunPod) so checkpoints survive
# an interrupted instance. Then start training with scripts/cloud_train.sh.
set -euo pipefail

SRC="${1:-}"
if [[ "$SRC" == hf:* ]]; then
  pip install -q "huggingface_hub>=0.34"
  python - "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1][3:], repo_type="dataset", local_dir=".")
PY
fi

for f in data/codes32/codes.npy data/codes32/meta.json ttmc/t2v.py scripts/train_t2v.py; do
  [[ -f "$f" ]] || { echo "missing $f (run from the unpacked bundle)"; exit 1; }
done

# Reuse the image's CUDA build of torch; install it only if absent.
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt

# Fetch CLIP now so a training restart never waits on the network.
python - <<'PY'
from transformers import CLIPTextModel, CLIPTokenizer
from ttmc.t2v import CLIP_NAME
CLIPTokenizer.from_pretrained(CLIP_NAME); CLIPTextModel.from_pretrained(CLIP_NAME)
print("CLIP cached")
PY

python - <<'PY'
import json, numpy as np, torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB",
      "| bf16:", torch.cuda.is_bf16_supported())
c = np.load("data/codes32/codes.npy", mmap_mode="r")
m = json.load(open("data/codes32/meta.json"))
print(f"codes {c.shape} {c.dtype}, {sum(m['val'])} val, tokenizer {m['tokenizer']}")
PY
df -h . | tail -1
echo
echo "ready. benchmark first (~10 min):  BENCH=1 bash scripts/cloud_train.sh"
echo "then the real run:                 bash scripts/cloud_train.sh"
