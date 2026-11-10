#!/usr/bin/env bash
# OST Stage C: wait for feature cache -> train 5 epochs (save per epoch) from v3 Stage B -> ost-bench eval.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
MODEL_PATH=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B
W="$ROOT/output/v3/ost_stage_c"
OUT="$W/train"
STAGE_B="${STAGE_B:-$ROOT/output/v3/stage_b}"
cd "$ROOT"

echo "[chain] waiting for feature extraction"
while pgrep -f tools/prepare_ost_stage_c.py > /dev/null; do sleep 60; done
python tools/prepare_ost_stage_c.py --merge --out_dir "$W/data"
N=$(wc -l < "$W/data/ost_train.jsonl")
SK=$(cat "$W"/data/skipped_rank*.txt 2>/dev/null | wc -l)
echo "[chain] sessions=$N skipped(corrupt images)=$SK"
[ $((N + SK)) -eq 6390 ] || { echo "[chain] expected 6390 = kept + skipped, got $N + $SK"; exit 1; }

echo "[chain] training"
torchrun --nproc_per_node="${NUM_GPUS:-8}" --master_port 29540 -m geomemory.train \
  --config_path "$ROOT/configs/geomemory_v3_stage_b.json" \
  --stage long --detach_every 8 \
  --model_name_or_path "$MODEL_PATH" \
  --geomemory_checkpoint "$STAGE_B" --reinit_fusion False \
  --local_files_only True --freeze_language_model True \
  --manifest "$W/data/ost_train.jsonl" --expand_qa False \
  --output_dir "$OUT" \
  --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 --fusion_learning_rate 1e-4 \
  --weight_decay 0.01 --warmup_ratio 0.03 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --num_train_epochs 5 \
  --logging_steps 10 --save_strategy epoch --save_total_limit 5 \
  --dataloader_num_workers 4 --model_max_length 16384 \
  --report_to none

echo "[chain] ost-bench eval"
EV="$W/eval"; mkdir -p "$EV"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/evaluation/eval_ost_geomemory.py --ckpt "$OUT" \
    --out_dir "$EV" --rank $i --world 8 > "$W/logs/eval_$i.log" 2>&1 &
done
wait
python scripts/evaluation/eval_ost_geomemory.py --ckpt "$OUT" --out_dir "$EV" --merge
echo "[chain] done"
