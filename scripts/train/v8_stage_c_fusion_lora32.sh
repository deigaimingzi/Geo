#!/usr/bin/env bash
# OST session training: Stage B initialization, train ONLY visual_fusion plus LLM LoRA (rank 32 alpha 64).
# adapter and memory are frozen; only fusion and LoRA receive gradients.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
MODEL_PATH="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
STAGE_B="${STAGE_B:-$ROOT/output/v8/stage_b}"
WORK="${WORK:-$ROOT/output/v8/ost_stage_c_3ep_fusion_lora32}"
OUT="${OUT:-$WORK/train}"
MANIFEST="${MANIFEST:-$ROOT/output/v8/ost_fulltrain_recovered/data/ost_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-}"
EVAL_ARGS=(--eval_strategy no)
if [[ -n "$VAL_MANIFEST" ]]; then
  EVAL_ARGS=(--eval_manifest "$VAL_MANIFEST" --eval_strategy epoch)
fi
torchrun --nproc_per_node="${NUM_GPUS:-8}" --master_port="${MASTER_PORT:-29842}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v8_stage_c.json" --stage long \
  --model_name_or_path "$MODEL_PATH" --geomemory_checkpoint "$STAGE_B" --reinit_fusion False \
  --local_files_only True --freeze_language_model True --attn_implementation flash_attention_2 \
  --lm_gradient_checkpointing True --qa_micro_batch_size 1 --detach_every 8 \
  --trainable_prefixes visual_fusion \
  --lora_enable True --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj \
  --lora_learning_rate 5e-5 \
  --manifest "$MANIFEST" --expand_qa False --max_frames 0 \
  "${EVAL_ARGS[@]}" \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --model_max_length "${MAX_LENGTH:-0}" --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --per_device_eval_batch_size 1 --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --group_by_length False --learning_rate "${LR:-1e-5}" --fusion_learning_rate "${FUSION_LR:-3e-5}" \
  --weight_decay 0.01 --warmup_ratio 0.05 --max_grad_norm 1.0 --lr_scheduler_type cosine \
  --num_train_epochs "${EPOCHS:-1}" --logging_steps 10 \
  --save_strategy epoch --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_WORKERS:-2}" --dataloader_prefetch_factor 1 \
  --dataloader_persistent_workers True --report_to none "$@"
