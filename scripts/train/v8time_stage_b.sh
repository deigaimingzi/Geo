#!/usr/bin/env bash
# GEO-Memory v8-time Stage B (T1 frame PE): joint QA on top of the v7 Stage A
# checkpoint.  Mirrors v7/scripts/train/v8_stage_b.sh but uses the v8-time
# source tree (which adds the Shared Temporal PE) and the v8time_stage_b
# config that enables memory_time_pe.
#
# Init  : $V7ROOT/output/v8/stage_a   (frozen Stage A from GeoMemory-v7)
# Data  : $V7ROOT/output/v8/manifests/stageB_train.jsonl
# Out   : $ROOT/output/v8time_t1/stage_b
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
MODEL_PATH="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
OUT="${OUT:-$ROOT/output/v8time_t1/stage_b}"
STAGE_A="${STAGE_A:-$V7ROOT/output/v8/stage_a}"
MANIFEST="${MANIFEST:-$V7ROOT/output/v8/manifests/stageB_train.jsonl}"
torchrun --nproc_per_node="${NUM_GPUS:-8}" --master_port="${MASTER_PORT:-29781}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v8time_stage_b.json" --stage joint \
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
