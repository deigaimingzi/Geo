#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
PY=/root/anaconda3/envs/vllm-cu129/bin/python
MODEL=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B
OUT="${OUT:-/data/juicefs_sharing_data/11188211/VIT-feather/v8_vit512}"
cd "$ROOT"
$PY -m torch.distributed.run --nproc_per_node=${NGPU:-8} --master_port=29791 tools/extract_vit_features_v8.py \
  --model "$MODEL" --src-index /data/juicefs_sharing_data/11188211/VIT-feather/v7_features/scene_index.jsonl \
  --output-dir "$OUT" --num-shards ${NGPU:-8} "$@"
