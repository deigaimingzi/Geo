#!/usr/bin/env bash
# GEO-Memory v4 Stage B: scene-grouped joint training (1 scene x 4 QA share one Memory),
# Effective batch: 1 group x 2 accum x 8 GPU = 16 groups x 4 QA = 64 QA.
# NumericReadout + always-on SceneScale (scale supervised only by length/area QA), frozen Qwen3.5-4B.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/stage_b}"
STAGE_A="${STAGE_A:-$ROOT/output/v4/stage_a}"
EXTRA=("$@")
mkdir -p "$ROOT/output/v4/logs"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_stage_b.json" \
  --stage joint \
  --model_name_or_path "$MODEL_PATH" \
  --geomemory_checkpoint "$STAGE_A" \
  --local_files_only True --freeze_language_model True \
  --manifest "$ROOT/output/v4/manifests/stageB_train.jsonl" \
  --eval_manifest "$ROOT/output/v4/manifests/stageA_val.jsonl" \
  --scene_group 4 --dynamic_rays True --query_count 64 --relation_pairs 16 \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" \
  --bf16 True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 --fusion_learning_rate 1e-4 \
  --weight_decay 0.01 --warmup_ratio 0.05 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --num_train_epochs 1 \
  --logging_steps 10 --save_steps 1000 --save_total_limit 3 \
  --dataloader_num_workers 8 --dataloader_prefetch_factor 4 \
  --report_to none "${EXTRA[@]}"
