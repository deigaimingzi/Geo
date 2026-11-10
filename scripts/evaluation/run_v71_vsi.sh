#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../train/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
CKPT="${1:?usage: run_v71_vsi.sh <final-checkpoint-dir> [evaluation args]}"
shift
PYTHON="${PYTHON:-/root/anaconda3/envs/geo/bin/python}"
NGPU="${NGPU:-${NUM_GPUS:-8}}"
QWEN_FRAMES="${QWEN_FRAMES:-32}"
if [[ "${VSI_SINGLE_RUN:-0}" != 1 ]]; then
  EVAL_BASE="${EVAL_BASE:-$CKPT}"
  failed=0
  for memory_frames in 32 64; do
    VSI_SINGLE_RUN=1 FRAMES="$memory_frames" QWEN_FRAMES=32 \
      EVAL_OUT="$EVAL_BASE/vsi-eval-m${memory_frames}-v32" \
      bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$CKPT" "$@" || failed=1
  done
  exit "$failed"
fi
FRAMES="${FRAMES:-32}"
QWEN="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
VSI_BASE=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/benchmark/VSI-Bench
VSI_JSONL="${VSI_JSONL:-$VSI_BASE/raw/test.jsonl}"
VSI_ROOT="${VSI_ROOT:-$VSI_BASE/videos}"
EVAL_OUT="${EVAL_OUT:-$CKPT/vsi-eval-m${FRAMES}-v${QWEN_FRAMES}}"
mkdir -p "$EVAL_OUT"
pids=()
for ((rank=0; rank<NGPU; rank++)); do
  CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" "$ROOT/scripts/evaluation/eval_vsi_geomemory.py" \
    --ckpt "$CKPT" --qwen_path "$QWEN" --vsi_jsonl "$VSI_JSONL" --vsi_root "$VSI_ROOT" \
    --out_dir "$EVAL_OUT" --rank "$rank" --world "$NGPU" --max_num_frames "$FRAMES" --qwen_frames "$QWEN_FRAMES" \
    "$@" > "$EVAL_OUT/eval_rank${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then exit 1; fi
"$PYTHON" /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation/merge_vsi_mra.py \
  --out_dir "$EVAL_OUT" --world "$NGPU" > "$EVAL_OUT/eval_merge.log" 2>&1
"$PYTHON" "$ROOT/scripts/evaluation/check_vsi_complete.py" --out_dir "$EVAL_OUT" --vsi_jsonl "$VSI_JSONL" "$@"
