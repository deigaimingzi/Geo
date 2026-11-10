#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the 8/16/32-frame manifest}"
: "${MODEL_PATH:?Set MODEL_PATH to the local/cached Qwen VLM checkpoint}"
: "${CHECKPOINT:?Set CHECKPOINT to the stage-B checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

torchrun --nproc_per_node="${NUM_GPUS}" -m geomemory.train \
  --config_path configs/geomemory_mvp.json \
  --stage long \
  --model_name_or_path "${MODEL_PATH}" \
  --local_files_only True \
  --freeze_language_model True \
  --detach_every 8 \
  --manifest "${MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from_checkpoint "${CHECKPOINT}" \
  --deepspeed scripts/zero2_opt.json \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --dataloader_num_workers 4 \
  --report_to none
# Stage C: detach_every=8 forces TBPTT for all sequences.
# Dynamic detach is also enabled, so short sessions still get full BPTT
# when they are <=8 frames.

