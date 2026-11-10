#!/usr/bin/env bash
# GEO-Memory v10 Stage A: from-scratch, expand memory_slots 128 -> 256.
# Backbone: 256 slots / 4-layer / 11D ray readout. Trains 20 epochs, checkpointing every 5.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v10/stage_a}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/output/v8/manifests/stageA_train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$TRAIN_MANIFEST}"
EXTRA=("$@")
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v10_stage_a.json" \
  --stage geometry \
  --manifest "$TRAIN_MANIFEST" \
  --eval_manifest "$EVAL_MANIFEST" \
  --max_frames 0 \
  --frame_choices 4,8,12,15,18,21,23,25,27,28,29,30,31,32 \
  --v7_stratified \
  --heldout_count 64 --heldout_p 0.5 \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 2 --prediction_loss_only True \
  --eval_strategy no \
  --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --num_train_epochs 20 \
  --logging_steps 10 --save_strategy no --save_at_epochs 5,10,15,20 --save_total_limit 4 \
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}" --dataloader_prefetch_factor 1 \
  --report_to none "${EXTRA[@]}"
