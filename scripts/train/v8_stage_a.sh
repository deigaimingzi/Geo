#!/usr/bin/env bash
# GEO-Memory v8 Stage A: 512-long-edge ViT cache (<=256 tok/frame), teacher32 reused, same model/supervision as v7.1.
# Backbone unchanged (64 slots / 2-layer / 11D). Mixed horizons 8/16/24/32.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v8/stage_a}"
INIT="${INIT:-$V2ROOT/output/v3/stage_a}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/output/v8/manifests/stageA_train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-$TRAIN_MANIFEST}"
EXTRA=("$@")
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v7_stage_a.json" \
  --stage geometry \
  --manifest "$TRAIN_MANIFEST" \
  --eval_manifest "$EVAL_MANIFEST" \
  --max_frames 0 \
  --frame_choices 8,16,24,32 \
  --v7_stratified \
  --heldout_count 64 --heldout_p 0.5 \
  --geomemory_checkpoint "$INIT" \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 2 --prediction_loss_only True \
  --eval_strategy epoch --eval_max_samples "${EVAL_MAX_SAMPLES:-256}" \
  --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --num_train_epochs 20 \
  --logging_steps 10 --save_strategy no --save_only_final True \
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}" --dataloader_prefetch_factor 1 \
  --report_to none "${EXTRA[@]}"
