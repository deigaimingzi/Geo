#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the prepared training manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

torchrun --nproc_per_node="${NUM_GPUS}" -m geomemory.train \
  --config_path /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2/configs/geomemory_mvp.json \
  --stage geometry \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v2/scripts/zero2_opt.json \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs 24 \
  --gradient_checkpointing False \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 2400 \
  --save_total_limit 6 \
  --dataloader_num_workers 4 \
  --report_to none

