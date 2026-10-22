#!/usr/bin/env bash
# Train the t2v transformer on a rented GPU, surviving disconnects, crashes and restarts.
#
#   BENCH=1 bash scripts/cloud_train.sh      # ~600 steps into data/runs/t2v_bench: measure samples/s
#   bash scripts/cloud_train.sh              # the real run, detached; log in data/train_t2v_cloud.log
#   LAYERS=16 D=768 OUT=data/runs/t2v_big bash scripts/cloud_train.sh   # ~3x bigger model (~160M)
#   HOURS=3 bash scripts/cloud_train.sh      # hard wall-clock limit (the budget cap)
#
# Re-running the same command after an interruption continues from last.pt on the same
# LR schedule (--auto-resume). Stop cleanly with:  touch data/STOP
# Defaults: batch 128 x 235k steps = ~30M samples (CLOUD_PLAN.md). LR 2e-4: 3e-4 diverged
# on the local run at batch 24, 1e-4 was stable; a 4x larger batch tolerates a bit more.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

STEPS="${STEPS:-235000}"
BATCH="${BATCH:-128}"
LR="${LR:-2e-4}"
WARMUP="${WARMUP:-2000}"
LAYERS="${LAYERS:-12}"
D="${D:-512}"
OUT="${OUT:-data/runs/t2v_cloud}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10000}"   # a samples_<step>.png sheet of the preview prompts
HOURS="${HOURS:-0}"          # >0: stop after this many hours of training, whatever STEPS says
SCRIPT="${SCRIPT:-scripts/train_t2v.py}"   # or scripts/train_flow.py (same flags)
EXTRA="${EXTRA:-}"                          # extra trainer flags, e.g. "--ema 0.9999 --heads 12"

if [[ "${BENCH:-0}" == 1 ]]; then
  exec python scripts/train_t2v.py --steps 600 --batch "$BATCH" --lr "$LR" --warmup 200 \
    --layers "$LAYERS" --d "$D" --out data/runs/t2v_bench --eval-every 300 --sample-every 100000
fi

if [[ "${_TTMC_DETACHED:-0}" != 1 ]]; then
  mkdir -p data
  _TTMC_DETACHED=1 nohup bash "$0" > data/train_t2v_cloud.log 2>&1 &
  echo "training in background (pid $!). follow with:  tail -f data/train_t2v_cloud.log"
  exit 0
fi

rm -f data/STOP data/FINISHED
START=$(date +%s)
for attempt in $(seq 1 20); do
  LEFT=0
  if [[ "$HOURS" != 0 ]]; then
    # hours left in the budget; restarts get only what remains
    LEFT=$(python -c "print(max(0.0, $HOURS - ($(date +%s) - $START) / 3600))")
    python -c "import sys; sys.exit(0 if $LEFT > 0.02 else 1)" || break
  fi
  echo "=== attempt $attempt $(date) ==="
  # train_flow.py stores the hours it has used in its checkpoint and wants the total budget;
  # train_t2v.py restarts its clock, so it gets what is left
  H="$LEFT"
  [[ "$SCRIPT" == *train_flow.py ]] && H="$HOURS"
  python "$SCRIPT" --auto-resume --steps "$STEPS" --batch "$BATCH" --lr "$LR" $EXTRA \
    --warmup "$WARMUP" --layers "$LAYERS" --d "$D" --out "$OUT" --stop-file data/STOP \
    --eval-every "$EVAL_EVERY" --sample-every "$SAMPLE_EVERY" --hours "$H" && break
  echo "exited with $?; retrying in 30 s"
  sleep 30
done
echo "=== finished $(date) ==="
# tells scripts/runpod_watch.py (on the home PC) to do its final copy, then it deletes the pod
touch data/FINISHED
# stop GPU billing right away even if the home PC is off; /workspace survives a stop.
# RunPod sets the pod id and a pod-scoped key only for the container's main process, not for
# ssh logins (which is how this script gets started), so read them from there.
eval "$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl > /dev/null; then
  sleep 1200                 # 20 min for the watcher (copies every 10 min) to take the final checkpoint
  runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
fi
