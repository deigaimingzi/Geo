#!/usr/bin/env bash
# VSI-Bench Stage A Memory ablation from cached features.
# Requires cache_vsi_features.py to have been run first.
#
# Usage:
#   bash scripts/evaluation/run_vsi_stage_a_ablation_cached.sh <ckpt> <out_json>
set -euo pipefail

ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${1:?usage: $0 <ckpt_dir> <out_json>}"
OUT="${2:?usage: $0 <ckpt_dir> <out_json>}"
shift 2 || true

CACHE_DIR="${CACHE_DIR:-/data/juicefs_sharing_data/11188211/VIT-feather/v8_vit512/features/vsi_bench_test}"
MAX_SCENES="${MAX_SCENES:-0}"
QUERY_COUNT="${QUERY_COUNT:-128}"
HELDOUT_COUNT="${HELDOUT_COUNT:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_BOOT="${N_BOOT:-10000}"

python "$ROOT/scripts/evaluation/eval_vsi_stage_a_ablation_cached.py" \
  --ckpt "$CKPT" \
  --cache_dir "$CACHE_DIR" \
  --max_scenes "$MAX_SCENES" \
  --query_count "$QUERY_COUNT" \
  --heldout_count "$HELDOUT_COUNT" \
  --heldout_p 1.0 \
  --batch_size "$BATCH_SIZE" \
  --n_boot "$N_BOOT" \
  --out "$OUT" \
  "$@"
