#!/usr/bin/env bash
# A1: per-Qwen-token geometry head (xyz, local scale, confidence, same-frame pairwise distance).
# Init = finished v4 Stage A (2328 steps).  Only patch_head trains; Memory / Self head / PoseHead frozen.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/a1_patch}"; INIT="${INIT:-$ROOT/output/v4/stage_a}"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_a1_patch.json" --stage geometry \
  --geomemory_checkpoint "$INIT" --trainable_prefixes "${TRAIN:-patch_head}" \
  --manifest "$ROOT/output/v4/manifests/stageA_train.jsonl" \
  --dynamic_rays True --patch_targets True --query_count 8 --relation_pairs 4 \
  --frame_choices 8,16 --detach_every 8 \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --eval_strategy no --learning_rate "${LR:-2e-4}" --weight_decay 0.01 --warmup_ratio 0.05 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 --max_steps "${STEPS:-1200}" \
  --logging_steps 10 --save_steps 400 --save_total_limit 5 \
  --dataloader_num_workers 6 --report_to none "$@"
