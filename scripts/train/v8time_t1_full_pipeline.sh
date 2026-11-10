#!/usr/bin/env bash
# v8-time (T1 frame PE) full sequential pipeline:
#   1. Stage B training (joint, 1 epoch) from v7 Stage A
#   2. Stage B VSI evaluation (m32-v32 and m64-v32)
#   3. Stage C training (LoRA, 3 epochs) from the new Stage B
#   4. Stage C OST evaluation for each of the 3 saved epochs
#
# All phases run sequentially in a single foreground process so the
# pipeline can be backgrounded with a single command and tracked via
# status.txt / pipeline.log.
set -euo pipefail
source "$(dirname "$0")/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"

export WORK="$(realpath -m "${WORK:-$ROOT/output/v8time_t1}")"
case "$WORK" in
  "$ROOT/output/"*|/tmp/bluecode/*) ;;
  *) printf 'Invalid run directory: %s\n' "$WORK" >&2; exit 1;;
esac
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

export NUM_GPUS="${NUM_GPUS:-8}"
export MASTER_PORT="${MASTER_PORT:-29832}"

STAGE_B_OUT="${STAGE_B_OUT:-$WORK/stage_b}"
STAGE_C_WORK="${STAGE_C_WORK:-$WORK/ost_stage_c_3ep_memory_lora}"
STAGE_C_OUT="$STAGE_C_WORK/train"

# ---------------------------------------------------------------------------
# Phase 1 — Stage B training (joint, 1 epoch, init from v7 Stage A)
# ---------------------------------------------------------------------------
status stage_b_training
if [[ ! -f "$STAGE_B_OUT/model.safetensors" ]]; then
  if [[ -d "$STAGE_B_OUT" ]]; then
    printf 'Stage B output directory exists but is incomplete: %s\n' "$STAGE_B_OUT" >&2
    exit 1
  fi
  OUT="$STAGE_B_OUT" bash "$ROOT/scripts/train/v8time_stage_b.sh" \
    > "$WORK/logs/stage_b_train.log" 2>&1
else
  printf '[%s] Stage B already trained, skipping\n' "$(date -Is)"
fi
test -f "$STAGE_B_OUT/model.safetensors"
test -f "$STAGE_B_OUT/trainer_state.json"

# ---------------------------------------------------------------------------
# Phase 2 — Stage B VSI evaluation (m32-v32 and m64-v32)
# ---------------------------------------------------------------------------
status stage_b_vsi_eval
if [[ ! -f "$STAGE_B_OUT/vsi-eval-m64-v32/eval_merge.log" ]]; then
  NGPU="${NUM_GPUS}" QWEN_FRAMES=32 \
    bash "$ROOT/scripts/evaluation/run_v71_vsi.sh" "$STAGE_B_OUT" \
    > "$WORK/logs/stage_b_vsi.log" 2>&1
else
  printf '[%s] Stage B VSI evaluation already complete, skipping\n' "$(date -Is)"
fi

# ---------------------------------------------------------------------------
# Phase 3 — Stage C training (LoRA rank 16, 3 epochs, init from Stage B)
# ---------------------------------------------------------------------------
status stage_c_training
export EPOCHS=3 SAVE_TOTAL_LIMIT=3
export MANIFEST="$WORK/data/ost_train_dialogue.jsonl"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$V7ROOT/output/v8/ost_fulltrain_recovered/data/ost_train.jsonl}"
if [[ ! -f "$MANIFEST" ]]; then
  python "$ROOT/tools/build_ost_dialogue_manifest.py" \
    --source "$SOURCE_MANIFEST" --out "$MANIFEST" --require-all-supervised
fi

if [[ ! -f "$STAGE_C_WORK/train.complete" ]]; then
  if [[ -d "$STAGE_C_OUT" ]]; then
    printf 'Stage C training output exists; refusing to overwrite: %s\n' "$STAGE_C_OUT" >&2
    exit 1
  fi
  WORK="$STAGE_C_WORK" STAGE_B="$STAGE_B_OUT" MANIFEST="$MANIFEST" \
    EPOCHS=3 SAVE_TOTAL_LIMIT=3 \
    bash "$ROOT/scripts/train/v8time_stage_c.sh" \
    > "$WORK/logs/stage_c_train.log" 2>&1
  python "$ROOT/tools/ost_three_epoch_results.py" index "$STAGE_C_WORK"
  touch "$STAGE_C_WORK/train.complete"
else
  python "$ROOT/tools/ost_three_epoch_results.py" index "$STAGE_C_WORK"
fi

# ---------------------------------------------------------------------------
# Phase 4 — Stage C OST evaluation for each saved epoch
# ---------------------------------------------------------------------------
export FEATURE_ROOT="${FEATURE_ROOT:-$V7ROOT/output/v8/ost_stage_c/data/eval_features}"
while IFS=$'\t' read -r epoch checkpoint; do
  status "stage_c_evaluating_epoch${epoch}"
  printf -v epoch_dir 'epoch%02d' "$epoch"
  eval_out="$STAGE_C_WORK/eval/$epoch_dir"
  if [[ ! -f "$eval_out/eval.complete" ]]; then
    EVAL_ARGS=(--scorer official --max_new_tokens "${MAX_NEW_TOKENS:-8192}")
    if [[ "${EVAL_LIMIT:-0}" != 0 ]]; then EVAL_ARGS+=(--limit "$EVAL_LIMIT"); fi
    CKPT="$checkpoint" OUT="$eval_out" \
      bash "$ROOT/scripts/evaluation/ost_v8_all.sh" "${EVAL_ARGS[@]}" \
      > "$WORK/logs/stage_c_eval_${epoch_dir}.log" 2>&1
    touch "$eval_out/eval.complete"
  else
    printf '[%s] Epoch %s OST evaluation already complete, skipping\n' "$(date -Is)" "$epoch"
  fi
done < <(python "$ROOT/tools/ost_three_epoch_results.py" paths "$STAGE_C_WORK")

# ---------------------------------------------------------------------------
# Phase 5 — Summarise Stage C epoch comparison
# ---------------------------------------------------------------------------
status summarizing
python "$ROOT/tools/ost_three_epoch_results.py" summary "$STAGE_C_WORK"
status completed
