#!/usr/bin/env bash
# Attach VSI evaluation to an already-running Stage B process.
set -euo pipefail
source "$(dirname "$0")/../train/v3_common.sh"
PID="${1:?usage: wait_stage_b_then_vsi.sh <train-pid> [checkpoint-dir]}"
CKPT="${2:-$ROOT/output/v4/stage_b_a1_full_1ep}"
FRAMES="${FRAMES:-32}"
LOG_DIR="$ROOT/output/v4/logs"
LOG="$LOG_DIR/stage_b_a1_full_chain.log"
mkdir -p "$LOG_DIR"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
if [ ! -f "$CKPT/model.safetensors" ]; then
  printf '[%s] training PID %s ended without a final model; VSI evaluation skipped\n' "$(date '+%F %T')" "$PID" >> "$LOG"
  exit 1
fi
sleep 30
printf '[%s] Stage B complete; starting %s-frame VSI evaluation\n' "$(date '+%F %T')" "$FRAMES" >> "$LOG"
for attempt in 1 2 3; do
  NGPU="${NUM_GPUS:-8}" FRAMES="$FRAMES" bash "$ROOT/scripts/evaluation/run_vsi_geomemory.sh" "$CKPT" >> "$LOG" 2>&1 && exit 0
  printf '[%s] VSI evaluation attempt %s failed; resuming\n' "$(date '+%F %T')" "$attempt" >> "$LOG"
  sleep 30
done
exit 1
