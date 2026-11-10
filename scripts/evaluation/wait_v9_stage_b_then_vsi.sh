#!/usr/bin/env bash
# Watch running v9 Stage B (PID) and auto-run VSI once it finishes.
set -euo pipefail
source "$(dirname "$0")/../train/v7_common.sh"
PID="${1:?usage: wait_v9_stage_b_then_vsi.sh <train-pid>}"
CKPT="${2:-$ROOT/output/v9/stage_b}"
VSI_LOG="$CKPT/vsi.log"
CHAIN_LOG="$CKPT/chain.log"
mkdir -p "$CKPT"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
if [ ! -f "$CKPT/model.safetensors" ]; then
  printf '[%s] training PID %s ended without a final model; VSI skipped\n' "$(date '+%F %T')" "$PID" >> "$CHAIN_LOG"
  exit 1
fi
sleep 30
printf '[%s] Stage B complete; starting VSI evaluation (m32-v32, m64-v32)\n' "$(date '+%F %T')" >> "$CHAIN_LOG"
for attempt in 1 2 3; do
  NGPU="${NUM_GPUS:-8}" QWEN_FRAMES=32 bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$CKPT" > "$VSI_LOG" 2>&1 && exit 0
  printf '[%s] VSI attempt %s failed; retrying\n' "$(date '+%F %T')" "$attempt" >> "$CHAIN_LOG"
  sleep 30
done
exit 1
