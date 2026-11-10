#!/usr/bin/env bash
# v7.1: 8-GPU dense 32-frame CUT3R teacher export.
set -euo pipefail
ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
PY=/root/anaconda3/envs/vllm-cu129/bin/python
OUT="${OUT:-/data/juicefs_sharing_data/11188211/VIT-feather/teacher32}"
mkdir -p "$OUT"
cd "$ROOT"
$PY -m torch.distributed.run --nproc_per_node=8 --master_port=29778 \
  tools/export_teacher32.py \
  --feature-index /data/juicefs_sharing_data/11188211/VIT-feather/v7_features/scene_index.jsonl \
  --output-dir "$OUT" \
  --num-shards 8 \
  "$@"
