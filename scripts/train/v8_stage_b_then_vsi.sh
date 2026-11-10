#!/usr/bin/env bash
# Final-only Stage B, followed by VSI with 32+32 and 64+32 frame budgets.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v8/stage_b}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/output/v8/logs/v8_stage_b.log}"
VSI_LOG="${VSI_LOG:-$ROOT/output/v8/logs/v8_stage_b_vsi.log}"
mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$VSI_LOG")"
OUT="$OUT" bash "$ROOT/scripts/train/v8_stage_b.sh" "$@" > "$TRAIN_LOG" 2>&1
test -f "$OUT/model.safetensors"
test -f "$OUT/trainer_state.json"
NGPU="${NUM_GPUS:-8}" QWEN_FRAMES=32 bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$OUT" > "$VSI_LOG" 2>&1
