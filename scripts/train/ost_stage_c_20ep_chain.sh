#!/usr/bin/env bash
# OST Stage C, 20 epochs from v3 Stage B.  Save every 5 epochs; after training evaluate the epoch-10 / 15 / 20 checkpoints
# on ost-bench sequentially (8-GPU data parallel each), then re-score with the same rules.
# Steps/epoch = 200 (6389 sessions / (8 GPU * bs2 * ga2)), so save_steps=1000 == every 5 epochs.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
MODEL_PATH=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B
W="$ROOT/output/v3/ost_stage_c"
RUN="${RUN:-$W/train_20ep}"
STAGE_B="${STAGE_B:-$ROOT/output/v3/stage_b}"
EPOCHS=20
STEPS_PER_EPOCH=200
SAVE_EVERY_EPOCHS=5
NUM_GPUS="${NUM_GPUS:-8}"
EVAL_EPOCHS=(10 15 20)
cd "$ROOT"
mkdir -p "$W/logs" "$RUN"

[ -s "$W/data/ost_train.jsonl" ] || { echo "[chain] missing $W/data/ost_train.jsonl"; exit 1; }

# ---------------- train ----------------
if [ ! -f "$RUN/model.safetensors" ]; then
  echo "[chain] $(date +%T) training ${EPOCHS} epochs -> $RUN"
  torchrun --nproc_per_node="$NUM_GPUS" --master_port 29541 -m geomemory.train \
    --config_path "$ROOT/configs/geomemory_v3_stage_b.json" \
    --stage long --detach_every 8 \
    --model_name_or_path "$MODEL_PATH" \
    --geomemory_checkpoint "$STAGE_B" --reinit_fusion False \
    --local_files_only True --freeze_language_model True \
    --manifest "$W/data/ost_train.jsonl" --expand_qa False \
    --output_dir "$RUN" \
    --deepspeed "$ROOT/scripts/zero2_opt.json" --bf16 True \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
    --learning_rate 3e-5 --fusion_learning_rate 1e-4 \
    --weight_decay 0.01 --warmup_ratio 0.03 --max_grad_norm 1.0 \
    --lr_scheduler_type cosine --num_train_epochs "$EPOCHS" \
    --logging_steps 10 --save_strategy steps \
    --save_steps $((STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS)) --save_total_limit 10 \
    --dataloader_num_workers 4 --model_max_length 16384 \
    --report_to none
else
  echo "[chain] training already finished, skipping"
fi

# ---------------- eval ----------------
for E in "${EVAL_EPOCHS[@]}"; do
  STEP=$((E * STEPS_PER_EPOCH))
  if [ "$E" -eq "$EPOCHS" ]; then CK="$RUN"; else CK="$RUN/checkpoint-$STEP"; fi
  [ -f "$CK/model.safetensors" ] || { echo "[chain] missing checkpoint $CK"; exit 1; }
  # intermediate checkpoints carry weights only; the eval loader also needs the config
  [ -f "$CK/geomemory_config.json" ] || cp "$RUN/geomemory_config.json" "$CK/geomemory_config.json"
  EV="$W/eval_20ep_epoch${E}"
  mkdir -p "$EV"
  echo "[chain] $(date +%T) eval epoch $E ($CK) -> $EV"
  PIDS=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/evaluation/eval_ost_geomemory.py --ckpt "$CK" \
      --out_dir "$EV" --rank "$i" --world "$NUM_GPUS" > "$W/logs/eval_20ep_e${E}_$i.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0
  for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
  [ "$FAIL" -eq 0 ] || { echo "[chain] eval epoch $E: a rank failed, see $W/logs/eval_20ep_e${E}_*.log"; exit 1; }
  python scripts/evaluation/eval_ost_geomemory.py --ckpt "$CK" --out_dir "$EV" --merge > /dev/null
  python scripts/evaluation/rescore_ost.py "$EV" > "$EV/rescore.log"
  echo "[chain] epoch $E overall: $(python -c "import json;print(json.load(open('$EV/metrics.json'))['overall'])")"
done

echo "[chain] $(date +%T) summary"
python - <<PY
import json
W="$W"
rows = {e: json.load(open(f"{W}/eval_20ep_epoch{e}/metrics.json")) for e in (10, 15, 20)}
keys = [k for k in rows[10] if k != "num_records"]
print("task".ljust(40), *[f"ep{e}".rjust(8) for e in rows])
for k in keys:
    print(k.ljust(40), *[f"{rows[e][k]:8.2f}" for e in rows])
json.dump(rows, open(f"{W}/eval_20ep_summary.json", "w"), indent=2)
PY
echo "[chain] done"
