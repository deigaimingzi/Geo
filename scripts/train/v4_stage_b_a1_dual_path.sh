#!/usr/bin/env bash
# Stage B-A1 Dual Path: frozen Stage-A recurrent memory + frozen A1 patch geometry.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/stage_b_a1_dual_path}"
STAGE_A="${STAGE_A:-$ROOT/output/v4/stage_a}"
A1_PATCH="${A1_PATCH:-$ROOT/output/v4/a1_patch}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$ROOT/output/v4/manifests/stageB_a1_train.jsonl}"
DEV_MANIFEST="${DEV_MANIFEST:-$ROOT/output/v4/manifests/stageB_a1_dev.jsonl}"
EXTRA=("$@")
mkdir -p "$ROOT/output/v4/logs"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_stage_b_a1_dual_path.json" \
  --stage joint --model_name_or_path "$MODEL_PATH" --attn_implementation flash_attention_2 --local_files_only True \
  --freeze_language_model True --stage_a_checkpoint "$STAGE_A" --a1_patch_checkpoint "$A1_PATCH" \
  --manifest "$TRAIN_MANIFEST" --eval_manifest "$DEV_MANIFEST" --scene_group 4 \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 --fusion_learning_rate 1e-4 --evidence_learning_rate 1e-4 --numeric_learning_rate 5e-5 \
  --weight_decay 0.01 --warmup_ratio 0.03 --max_grad_norm 1.0 --lr_scheduler_type cosine \
  --num_train_epochs 1 --logging_steps 10 --eval_strategy steps --eval_steps 4000 \
  --save_strategy steps --save_steps 4000 --save_total_limit 3 --load_best_model_at_end True \
  --metric_for_best_model eval_loss --greater_is_better False \
  --ddp_timeout 7200 \
  --dataloader_num_workers 8 --dataloader_prefetch_factor 4 --report_to none "${EXTRA[@]}"
