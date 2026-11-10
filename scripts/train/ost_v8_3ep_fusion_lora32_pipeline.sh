#!/usr/bin/env bash
# Train 3 epochs (fusion + LoRA r32/a64 only; adapter & memory frozen), then evaluate epochs 1/2/3 on OST-bench.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
export WORK="$(realpath -m "${WORK:-$ROOT/output/v8/ost_stage_c_3ep_fusion_lora32}")"
case "$WORK" in "$ROOT/output/"*|/tmp/bluecode/*) ;; *) printf 'Invalid run directory: %s\n' "$WORK" >&2; exit 1;; esac
mkdir -p "$WORK/logs" "$WORK/data"
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
export MASTER_PORT="${MASTER_PORT:-29842}"
export MANIFEST="$WORK/data/ost_train_dialogue.jsonl"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/output/v8/ost_fulltrain_recovered/data/ost_train.jsonl}"
if [[ ! -f "$MANIFEST" ]]; then
  python "$ROOT/tools/build_ost_dialogue_manifest.py" --source "$SOURCE_MANIFEST" --out "$MANIFEST" --require-all-supervised
fi
if [[ ! -f "$WORK/train.complete" ]]; then
  if [[ -d "$WORK/train" ]]; then
    printf 'Existing training output found; choose a new WORK directory instead of overwriting\n' >&2
    exit 1
  fi
  status training
  bash "$ROOT/scripts/train/v8_stage_c_fusion_lora32.sh" "$@" > "$WORK/train.log" 2>&1
  python "$ROOT/tools/ost_three_epoch_results.py" index "$WORK"
  touch "$WORK/train.complete"
else
  python "$ROOT/tools/ost_three_epoch_results.py" index "$WORK"
fi
while IFS=$'\t' read -r epoch checkpoint; do
  status "evaluating_epoch${epoch}"
  printf -v epoch_dir 'epoch%02d' "$epoch"
  eval_out="$WORK/eval/$epoch_dir"
  if [[ ! -f "$eval_out/eval.complete" ]]; then
    EVAL_ARGS=(--scorer official --max_new_tokens "${MAX_NEW_TOKENS:-8192}")
    if [[ "${EVAL_LIMIT:-0}" != 0 ]]; then EVAL_ARGS+=(--limit "$EVAL_LIMIT"); fi
    CKPT="$checkpoint" OUT="$eval_out" bash "$ROOT/scripts/evaluation/ost_v8_all.sh" "${EVAL_ARGS[@]}" \
      > "$WORK/logs/eval_${epoch_dir}.log" 2>&1
    touch "$eval_out/eval.complete"
  fi
done < <(python "$ROOT/tools/ost_three_epoch_results.py" paths "$WORK")
status summarizing
python "$ROOT/tools/ost_three_epoch_results.py" summary "$WORK"
status completed
