#!/usr/bin/env bash
# Final-only Stage B, followed by VSI with 32+32 and 64+32 frame budgets.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v71/stage_b2}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/output/v7/logs/v71_stage_b2.log}"
VSI_LOG="${VSI_LOG:-$ROOT/output/v7/logs/v71_stage_b2_vsi.log}"
mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$VSI_LOG")"
OUT="$OUT" bash "$ROOT/scripts/train/v71_stage_b.sh" "$@" > "$TRAIN_LOG" 2>&1
test -f "$OUT/model.safetensors"
test -f "$OUT/trainer_state.json"
NGPU="${NUM_GPUS:-8}" QWEN_FRAMES=32 bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$OUT" > "$VSI_LOG" 2>&1
