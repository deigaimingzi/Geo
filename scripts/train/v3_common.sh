#!/usr/bin/env bash
# Shared environment for the GEO-Memory v3 runs.
export PYTHONPATH="/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2
MODEL_PATH=/data/juicefs_sharing_data/public_data/11186230/Qinjia-2026/model/Qwen/Qwen3.5-4B
