#!/usr/bin/env bash
set -euo pipefail
ROOT="$(realpath "$(dirname "$0")/../..")"
source "$ROOT/scripts/train/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
CKPT="${CKPT:-$ROOT/output/v8/ost_stage_c/train}"
OUT="${OUT:-$ROOT/output/v8/ost_stage_c/eval}"
MODEL="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/output/v8/ost_stage_c/data/eval_features}"
NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$OUT/logs"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if (( ${#DEVICES[@]} < NUM_GPUS )); then
  printf 'Not enough CUDA_VISIBLE_DEVICES for %s workers\n' "$NUM_GPUS" >&2
  exit 1
fi
PIDS=()
for ((rank=0; rank<NUM_GPUS; rank++)); do
  CUDA_VISIBLE_DEVICES="${DEVICES[$rank]}" python "$ROOT/scripts/evaluation/eval_ost_geomemory.py" \
    --ckpt "$CKPT" --qwen_path "$MODEL" --out_dir "$OUT" --feature_root "$FEATURE_ROOT" \
    --rank "$rank" --world "$NUM_GPUS" "$@" > "$OUT/logs/rank${rank}.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
if (( FAILED )); then
  printf 'OST evaluation worker failed; see %s/logs\n' "$OUT" >&2
  exit 1
fi
python "$ROOT/scripts/evaluation/eval_ost_geomemory.py" --ckpt "$CKPT" --qwen_path "$MODEL" \
  --out_dir "$OUT" --world "$NUM_GPUS" --merge "$@"
