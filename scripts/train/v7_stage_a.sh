#!/usr/bin/env bash
# GEO-Memory v7 Stage A: v3 backbone (64 slots / 2-layer decoder / 11D ray readout),
# new training tasks: stratified history + held-out view + hybrid robust+0.25*NLL loss.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
OUT="${OUT:-$ROOT/output/v7/stage_a}"
INIT="${INIT:-$V2ROOT/output/v3/stage_a}"   # warm start from the trained v3 Stage A
EXTRA=("$@")
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v7_stage_a.json" \
  --stage geometry \
  --manifest "$V2ROOT/output/v3/manifests/stageA_train.jsonl" \
  --eval_manifest "$V2ROOT/output/v3/manifests/stageA_val.jsonl" \
  --max_frames 8 \
  --v7_stratified \
  --heldout_count 64 --heldout_p 0.5 \
  --geomemory_checkpoint "$INIT" \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 2 --prediction_loss_only True \
  --eval_strategy steps --eval_steps 400 \
  --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --num_train_epochs 8 \
  --logging_steps 10 --save_steps 400 --save_total_limit 3 \
  --dataloader_num_workers 4 --report_to none "${EXTRA[@]}"
