#!/usr/bin/env bash
# GEO-Memory v3 Stage B: 16-frame Memory + 8-frame fused visual tokens + frozen Qwen3.5-4B.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v3/stage_b}"
STAGE_A="${STAGE_A:-$ROOT/output/v3/stage_a}"
EXTRA=("$@")
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v3_stage_b.json" \
  --stage joint \
  --model_name_or_path "$MODEL_PATH" \
  --geomemory_checkpoint "$STAGE_A" \
  --local_files_only True --freeze_language_model True \
  --manifest "$ROOT/output/v3/manifests/stageB_train.jsonl" \
  --eval_manifest "$ROOT/output/v3/manifests/stageA_val.jsonl" \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 --fusion_learning_rate 1e-4 \
  --weight_decay 0.01 --warmup_ratio 0.05 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --num_train_epochs 1 \
  --logging_steps 10 --save_steps 1000 --save_total_limit 3 \
  --dataloader_num_workers 8 --dataloader_prefetch_factor 4 \
  --report_to none "${EXTRA[@]}"
