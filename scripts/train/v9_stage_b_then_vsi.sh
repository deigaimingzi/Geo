#!/usr/bin/env bash
# GEO-Memory v9 Stage B -> VSI chain: train Stage B, then run VSI (32+32, 64+32).
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v9/stage_b}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/output/v9/stage_b/train.log}"
VSI_LOG="${VSI_LOG:-$ROOT/output/v9/stage_b/vsi.log}"
mkdir -p "$(dirname "$TRAIN_LOG")" "$(dirname "$VSI_LOG")"
OUT="$OUT" bash "$ROOT/scripts/train/v9_stage_b.sh" "$@" > "$TRAIN_LOG" 2>&1
test -f "$OUT/model.safetensors"
test -f "$OUT/trainer_state.json"
NGPU="${NUM_GPUS:-8}" QWEN_FRAMES=32 bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$OUT" > "$VSI_LOG" 2>&1
