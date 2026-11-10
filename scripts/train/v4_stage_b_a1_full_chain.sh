#!/usr/bin/env bash
# Full-data Stage B followed by VSI-Bench evaluation (32 frames by default).
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/stage_b_a1_full_1ep}"
FRAMES="${FRAMES:-32}"
LOG_DIR="$ROOT/output/v4/logs"
LOG="$LOG_DIR/stage_b_a1_full_chain.log"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/stage_b_a1_full_1ep.log}"
mkdir -p "$LOG_DIR"
printf '[%s] Stage B train start\n' "$(date '+%F %T')" >> "$LOG"
bash "$ROOT/scripts/train/v4_stage_b_a1_full_1ep.sh" > "$TRAIN_LOG" 2>&1
[ -f "$OUT/model.safetensors" ] || { printf '[%s] Stage B did not produce model.safetensors\n' "$(date '+%F %T')" >> "$LOG"; exit 1; }
sleep 30
printf '[%s] Stage B train complete; %s-frame VSI evaluation start\n' "$(date '+%F %T')" "$FRAMES" >> "$LOG"
for attempt in 1 2 3; do
  NGPU="${NUM_GPUS:-8}" FRAMES="$FRAMES" bash "$ROOT/scripts/evaluation/run_vsi_geomemory.sh" "$OUT" >> "$LOG" 2>&1 && exit 0
  printf '[%s] VSI evaluation attempt %s failed; resuming\n' "$(date '+%F %T')" "$attempt" >> "$LOG"
  sleep 30
done
exit 1
