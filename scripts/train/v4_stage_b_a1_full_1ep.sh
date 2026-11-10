#!/usr/bin/env bash
# Final one-epoch fit over every Stage-B scene after dev-set model selection.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/stage_b_a1_full_1ep}"
STAGE_A="${STAGE_A:-$ROOT/output/v4/stage_a}"
A1_PATCH="${A1_PATCH:-$ROOT/output/v4/a1_patch}"
EXTRA=("$@")
mkdir -p "$ROOT/output/v4/logs"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_stage_b_a1_dual_path.json" \
  --stage joint --model_name_or_path "$MODEL_PATH" --attn_implementation flash_attention_2 --local_files_only True \
  --freeze_language_model True --stage_a_checkpoint "$STAGE_A" --a1_patch_checkpoint "$A1_PATCH" \
  --manifest "$ROOT/output/v4/manifests/stageB_train.jsonl" --scene_group 4 \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 --fusion_learning_rate 1e-4 --evidence_learning_rate 1e-4 --numeric_learning_rate 5e-5 \
  --weight_decay 0.01 --warmup_ratio 0.03 --max_grad_norm 1.0 --lr_scheduler_type cosine \
  --num_train_epochs 1 --logging_steps 10 --eval_strategy no --save_strategy no \
  --dataloader_num_workers 8 --dataloader_prefetch_factor 4 --report_to none "${EXTRA[@]}"
