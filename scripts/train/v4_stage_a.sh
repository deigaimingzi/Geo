#!/usr/bin/env bash
# GEO-Memory v4 Stage A: CUT3R geometry distillation, 128 slots + 16 camera slots,
# per-visit dynamic rays (re-sampled from raw CUT3R depth every epoch), relation + pose losses.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/stage_a}"
EXTRA=("$@")
mkdir -p "$ROOT/output/v4/logs"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_stage_a.json" \
  --stage geometry \
  --manifest "$ROOT/output/v4/manifests/stageA_train.jsonl" \
  --eval_manifest "$ROOT/output/v4/manifests/stageA_val.jsonl" \
  --dynamic_rays True --query_count 128 --relation_pairs 32 \
  --frame_choices 8,16 --detach_every 8 \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 2 --prediction_loss_only True \
  --eval_strategy steps --eval_steps 400 \
  --learning_rate 5e-5 --weight_decay 0.01 --warmup_ratio 0.05 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --num_train_epochs "${EPOCHS:-24}" \
  --logging_steps 10 --save_steps 800 --save_total_limit 3 \
  --dataloader_num_workers 6 --report_to none "${EXTRA[@]}"
