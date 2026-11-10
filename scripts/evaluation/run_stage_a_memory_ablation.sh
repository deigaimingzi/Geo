#!/usr/bin/env bash
# Stage A Memory ablation (inference-time intervention).
#
# Loads output/v8/stage_a (epoch 20) and runs Full / Zero / Shuffle Memory
# readout conditions on the SAME rays / SAME teacher targets.  Does NOT touch
# the training code or the checkpoint.
#
# Usage:
#   bash scripts/evaluation/run_stage_a_memory_ablation.sh
#   MAX_SAMPLES=128 bash scripts/evaluation/run_stage_a_memory_ablation.sh
set -euo pipefail

ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-$ROOT/output/v8/stage_a}"
MANIFEST="${MANIFEST:-$ROOT/output/v8/manifests/stageA_train.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
QUERY_COUNT="${QUERY_COUNT:-128}"
HELDOUT_COUNT="${HELDOUT_COUNT:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUT="${OUT:-$ROOT/output/v8/stage_a_memory_ablation.json}"

python "$ROOT/scripts/evaluation/eval_stage_a_memory_ablation.py" \
  --ckpt "$CKPT" \
  --manifest "$MANIFEST" \
  --max_samples "$MAX_SAMPLES" \
  --query_count "$QUERY_COUNT" \
  --heldout_count "$HELDOUT_COUNT" \
  --heldout_p 1.0 \
  --batch_size "$BATCH_SIZE" \
  --out "$OUT" \
  "$@"
