#!/usr/bin/env bash
# Stage A Memory ablation: scene-level paired bootstrap + Initial-Memory baseline.
#
# Four conditions: Full / Zero / Init / Shuffle Memory, same rays / same targets.
# Reports per-scene paired bootstrap 95% CI for each Δ and frac(Δ>0).
#
# Usage:
#   bash scripts/evaluation/run_stage_a_ablation_stats.sh
#   MAX_SAMPLES=128 N_BOOT=10000 bash scripts/evaluation/run_stage_a_ablation_stats.sh
set -euo pipefail

ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-$ROOT/output/v8/stage_a}"
MANIFEST="${MANIFEST:-$ROOT/output/v8/manifests/stageA_train.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
QUERY_COUNT="${QUERY_COUNT:-128}"
HELDOUT_COUNT="${HELDOUT_COUNT:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_BOOT="${N_BOOT:-10000}"
OUT="${OUT:-$ROOT/output/v8/stage_a_ablation_stats.json}"

python "$ROOT/scripts/evaluation/eval_stage_a_ablation_stats.py" \
  --ckpt "$CKPT" \
  --manifest "$MANIFEST" \
  --max_samples "$MAX_SAMPLES" \
  --query_count "$QUERY_COUNT" \
  --heldout_count "$HELDOUT_COUNT" \
  --heldout_p 1.0 \
  --batch_size "$BATCH_SIZE" \
  --n_boot "$N_BOOT" \
  --out "$OUT" \
  "$@"
