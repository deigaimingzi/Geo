#!/usr/bin/env bash
# Resume epoch-3 training from checkpoint-798, then evaluate epoch 3.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
export WORK="$(realpath -m "${WORK:-$ROOT/output/v8/ost_stage_c_3ep_memory_lora}")"
case "$WORK" in "$ROOT/output/"*|/tmp/bluecode/*) ;; *) printf 'Invalid run directory: %s\n' "$WORK" >&2; exit 1;; esac
mkdir -p "$WORK/logs" "$WORK/eval/epoch03"
exec 9>"$WORK/run.lock"
flock -n 9 || { printf 'This pipeline is already running\n' >&2; exit 1; }
printf '%s\n' "$$" > "$WORK/pipeline.pid"
PHASE=preflight
finish() {
  code=$?
  if (( code != 0 )); then
    printf 'failed:%s exit_code=%s time=%s\n' "$PHASE" "$code" "$(date -Is)" > "$WORK/status.txt"
  fi
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
status() {
  PHASE="$1"
  printf '%s time=%s\n' "$PHASE" "$(date -Is)" > "$WORK/status.txt"
  printf '[%s] %s\n' "$(date -Is)" "$PHASE"
}
export NUM_GPUS="${NUM_GPUS:-8}" EPOCHS=3 SAVE_TOTAL_LIMIT=3
export MASTER_PORT="${MASTER_PORT:-29832}"
export MANIFEST="$WORK/data/ost_train_dialogue.jsonl"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/output/v8/ost_fulltrain_recovered/data/ost_train.jsonl}"
if [[ ! -f "$MANIFEST" ]]; then
  python "$ROOT/tools/build_ost_dialogue_manifest.py" --source "$SOURCE_MANIFEST" --out "$MANIFEST" --require-all-supervised
fi

RESUME_CKPT="$WORK/train/checkpoint-798"
if [[ ! -d "$RESUME_CKPT" ]]; then
  printf 'Resume checkpoint not found: %s\n' "$RESUME_CKPT" >&2
  exit 1
fi

status "training_epoch3_resume"
bash "$ROOT/scripts/train/v8_stage_c.sh" --resume_from_checkpoint "$RESUME_CKPT" > "$WORK/train_epoch3.log" 2>&1

python "$ROOT/tools/ost_three_epoch_results.py" index "$WORK" || true

status "evaluating_epoch3"
eval_out="$WORK/eval/epoch03"
EVAL_ARGS=(--scorer official --max_new_tokens "${MAX_NEW_TOKENS:-8192}")
if [[ "${EVAL_LIMIT:-0}" != 0 ]]; then EVAL_ARGS+=(--limit "$EVAL_LIMIT"); fi

# epoch3 checkpoint is the latest one after training completes
EPOCH3_CKPT=$(python3 -c "
import json, os
d = '$WORK/train'
ckpts = [c for c in os.listdir(d) if c.startswith('checkpoint-') and os.path.isdir(os.path.join(d, c))]
ckpts.sort(key=lambda c: int(c.split('-')[1]))
print(os.path.join(d, ckpts[-1]))
")
printf 'Using epoch3 checkpoint: %s\n' "$EPOCH3_CKPT"

CKPT="$EPOCH3_CKPT" OUT="$eval_out" bash "$ROOT/scripts/evaluation/ost_v8_all.sh" "${EVAL_ARGS[@]}" \
  > "$WORK/logs/eval_epoch03.log" 2>&1
touch "$eval_out/eval.complete"

status summarizing
python "$ROOT/tools/ost_three_epoch_results.py" summary "$WORK" || true
status completed
