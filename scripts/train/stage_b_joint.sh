#!/usr/bin/env bash
set -euo pipefail

# --- Thread limits: prevent pthread_create EAGAIN with 8 procs ---
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export TORCH_INTRAOP_PARALLELISM=4
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false
export RAYON_RS_NUM_THREADS=4

: "${MANIFEST:?Set MANIFEST to the prepared training manifest}"
: "${MODEL_PATH:?Set MODEL_PATH to the local/cached Qwen VLM checkpoint}"
: "${GEOMETRY_CHECKPOINT:?Set GEOMETRY_CHECKPOINT to the stage-A checkpoint directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

torchrun --nproc_per_node="${NUM_GPUS}" -m geomemory.train \
  --config_path /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2/configs/geomemory_mvp.json \
  --stage joint \
  --model_name_or_path "${MODEL_PATH}" \
  --geomemory_checkpoint "${GEOMETRY_CHECKPOINT}" \
  --local_files_only True \
  --freeze_language_model True \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2/scripts/zero2_opt.json \
  --bf16 True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --dataloader_num_workers 8 \
  --dataloader_prefetch_factor 4 \
  --report_to none
# Stage B: dynamic TBPTT is auto-enabled by train.py.
# Sequences <=8 frames use full BPTT; >8 frames use detach_every=8.
