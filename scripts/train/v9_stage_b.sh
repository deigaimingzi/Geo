#!/usr/bin/env bash
# GEO-Memory v9 Stage B: joint QA on top of v9 Stage A (128 slots / 4-layer memory).
# Same recipe as v8 Stage B; only the backbone depth/slots change to match v9 Stage A.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
MODEL_PATH="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
OUT="${OUT:-$ROOT/output/v9/stage_b}"
STAGE_A="${STAGE_A:-$ROOT/output/v9/stage_a/checkpoint-4950}"
MANIFEST="${MANIFEST:-$ROOT/output/v8/manifests/stageB_train.jsonl}"
torchrun --nproc_per_node="${NUM_GPUS:-8}" --master_port="${MASTER_PORT:-29781}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v9_stage_b.json" --stage joint \
  --model_name_or_path "$MODEL_PATH" --local_files_only True --freeze_language_model True \
  --attn_implementation flash_attention_2 --lm_gradient_checkpointing "${LM_GC:-True}" \
  --qa_micro_batch_size "${QA_MICRO_BATCH_SIZE:-2}" \
  --geomemory_checkpoint "$STAGE_A" --manifest "$MANIFEST" \
  --scene_group 4 --v7_stratified True --heldout_p 0.0 --max_frames 32 \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --model_max_length 32768 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --group_by_length True \
  --learning_rate 3e-5 --fusion_learning_rate 1e-4 --weight_decay 0.01 \
  --warmup_ratio 0.05 --max_grad_norm 1.0 --lr_scheduler_type cosine \
  --num_train_epochs 1 --logging_steps 10 --eval_strategy no \
  --save_strategy steps --save_steps 1000 --save_total_limit 3 \
  --save_only_final "${SAVE_ONLY_FINAL:-True}" \
  --dataloader_num_workers "${DATALOADER_WORKERS:-2}" --dataloader_prefetch_factor 1 \
  --dataloader_persistent_workers True --report_to none "$@"
