#!/usr/bin/env bash
# VSI-Bench Stage A Memory ablation (288 test scenes, NOT in training set).
#
# Extracts frames + ViT features + CUT3R teacher on-the-fly, then runs
# full/zero/init Memory ablation with scene-level paired bootstrap.
#
# Usage:
#   bash scripts/evaluation/run_vsi_stage_a_ablation.sh <ckpt_dir> <out_json>
#   bash scripts/evaluation/run_vsi_stage_a_ablation.sh output/v8/stage_a output/v8/vsi_ablation.json
set -euo pipefail

ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
export PYTHONPATH="$ROOT/src:/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${1:?usage: $0 <ckpt_dir> [out_json]}"
shift || true
OUT="${1:-$ROOT/output/vsi_ablation.json}"
shift || true

MAX_SCENES="${MAX_SCENES:-288}"
NUM_FRAMES="${NUM_FRAMES:-32}"
QUERY_COUNT="${QUERY_COUNT:-128}"
HELDOUT_COUNT="${HELDOUT_COUNT:-64}"
N_BOOT="${N_BOOT:-10000}"
RANK="${RANK:-0}"
WORLD="${WORLD:-1}"

PY="${PY:-/root/anaconda3/envs/vllm-cu129/bin/python}"
"$PY" "$ROOT/scripts/evaluation/eval_vsi_stage_a_ablation.py" \
  --ckpt "$CKPT" \
  --max_scenes "$MAX_SCENES" \
  --num_frames "$NUM_FRAMES" \
  --query_count "$QUERY_COUNT" \
  --heldout_count "$HELDOUT_COUNT" \
  --heldout_p 1.0 \
  --n_boot "$N_BOOT" \
  --rank "$RANK" \
  --world "$WORLD" \
  --out "$OUT" \
  "$@"
