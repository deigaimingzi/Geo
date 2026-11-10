#!/usr/bin/env bash
# All OST training QA -> 10 epochs -> retain epochs 2/4/6/8/10 -> validate all five on OST-Bench.
set -euo pipefail
ROOT="$(realpath "$(dirname "$0")/../..")"
source "$ROOT/scripts/train/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
WORK="${WORK:-$ROOT/output/v8/ost_stage_c_10ep_fulltrain}"
DATA_SOURCE="${DATA_SOURCE:-$ROOT/output/v8/ost_stage_c/data}"
export NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$WORK/logs"
trap 'printf "Pipeline failed at line %s\n" "$LINENO" >&2; printf "failed\n" > "$WORK/status.txt"' ERR
printf 'preparing_full_train\n' > "$WORK/status.txt"
python "$ROOT/tools/build_ost_fulltrain_manifest.py" --source "$DATA_SOURCE" --out "$WORK/data"
TRAIN_ARGS=(--save_strategy no --save_at_epochs 2,4,6,8,10 --save_total_limit 5 \
            --save_only_final False --eval_saved_checkpoints False --eval_strategy no)
if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi
if [[ ! -f "$WORK/train.complete" || -n "${RESUME_FROM:-}" ]]; then
  printf 'training_10_epochs\n' > "$WORK/status.txt"
  EPOCHS=10 WORK="$WORK" OUT="$WORK/train" MANIFEST="$WORK/data/ost_train.jsonl" VAL_MANIFEST="" \
    bash "$ROOT/scripts/train/v8_stage_c.sh" "${TRAIN_ARGS[@]}" > "$WORK/logs/train.log" 2>&1
  printf 'completed\n' > "$WORK/train.complete"
fi
python "$ROOT/tools/ost_epoch_results.py" --work "$WORK" --index-only > "$WORK/logs/checkpoint_index.log" 2>&1
for epoch in 2 4 6 8 10; do
  printf -v tag 'epoch%02d' "$epoch"
  checkpoint="$(python "$ROOT/tools/ost_epoch_results.py" --work "$WORK" --checkpoint-for "$epoch")"
  printf 'evaluating_%s\n' "$tag" > "$WORK/status.txt"
  CKPT="$checkpoint" OUT="$WORK/eval/$tag" FEATURE_ROOT="$DATA_SOURCE/eval_features" \
    bash "$ROOT/scripts/evaluation/ost_v8_all.sh" > "$WORK/logs/eval_${tag}.log" 2>&1
done
python "$ROOT/tools/ost_epoch_results.py" --work "$WORK" > "$WORK/logs/epoch_comparison.log" 2>&1
python "$ROOT/tools/summarize_ost_stage_c.py" --work "$WORK" --eval-dir "$WORK/eval/epoch10" \
  --stage-b "${STAGE_B:-$ROOT/output/v8/stage_b}" > "$WORK/verification.log" 2>&1
printf 'completed\n' > "$WORK/status.txt"
