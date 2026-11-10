#!/usr/bin/env bash
# E1: pose-only validation.  Self head / geometry memory frozen (from the finished v4 Stage A);
# only the camera slots + PoseHead learn the absolute CUT3R-scale camera pose.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
OUT="${OUT:-$ROOT/output/v4/e1_pose}"; INIT="${INIT:-$ROOT/output/v4/stage_a}"
TRAIN="${TRAIN:-pose_head,memory.pose_update,memory.initial_pose}"
cd "$ROOT"
torchrun --nproc_per_node="${NUM_GPUS:-8}" -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v4_e1_pose_only.json" --stage geometry \
  --geomemory_checkpoint "$INIT" --trainable_prefixes "$TRAIN" \
  --manifest "$ROOT/output/v4/manifests/stageA_train.jsonl" \
  --dynamic_rays True --query_count 32 --relation_pairs 8 \
  --frame_choices 16 --detach_every 8 \
  --output_dir "$OUT" --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --eval_strategy no --learning_rate "${LR:-1e-4}" --weight_decay 0.01 --warmup_ratio 0.05 \
  --lr_scheduler_type cosine --max_grad_norm 1.0 --max_steps "${STEPS:-600}" \
  --logging_steps 10 --save_steps 200 --save_total_limit 3 \
  --dataloader_num_workers 6 --report_to none "$@"
