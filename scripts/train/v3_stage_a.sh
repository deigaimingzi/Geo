#!/usr/bin/env bash
# GEO-Memory v3 Stage A: geometry pretraining. 8-frame prefix training, 16-frame validation.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v3/stage_a}"
EXTRA=("$@")
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v3_stage_a.json" \
  --stage geometry \
  --manifest "$ROOT/output/v3/manifests/stageA_train.jsonl" \
  --eval_manifest "$ROOT/output/v3/manifests/stageA_val.jsonl" \
  --max_frames 8 \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 2 --prediction_loss_only True \
  --eval_strategy steps --eval_steps 400 \
  --learning_rate 2e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --num_train_epochs 24 \
  --logging_steps 10 --save_steps 800 --save_total_limit 3 \
  --dataloader_num_workers 4 --report_to none "${EXTRA[@]}"
