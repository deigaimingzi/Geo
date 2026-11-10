#!/usr/bin/env bash
# v7: 8-GPU full ViT feature extraction over all scenes in the flat dataset.
set -euo pipefail
ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
PY=/root/anaconda3/envs/vllm-cu129/bin/python
DATASET=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/datasets/all_no_filter_plus_spar234k_flat.json
MODEL=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B
OUT="${OUT:-/data/juicefs_sharing_data/11188211/VIT-feather/v7_features}"
mkdir -p "$OUT"
cd "$ROOT"
$PY -m torch.distributed.run --nproc_per_node=8 --master_port=29777 \
  tools/extract_vit_features_v7.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --output-dir "$OUT" \
  --num-shards 8 \
  --frame-cache /data/juicefs_sharing_data/11188211/VIT-feather/frame_cache \
  "$@"
