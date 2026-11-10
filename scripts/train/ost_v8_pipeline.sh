#!/usr/bin/env bash
# Resumable extraction -> held-out validation -> Stage C -> complete OST benchmark.
set -euo pipefail
ROOT="$(realpath "$(dirname "$0")/../..")"
source "$ROOT/scripts/train/v7_common.sh"
export PYTHONPATH="${STAGE_B_DEPS:-/tmp/bluecode/v71_stage_b_deps}:$PYTHONPATH"
WORK="${WORK:-$ROOT/output/v8/ost_stage_c}"
MODEL="${QWEN_MODEL:-/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/model/Qwen/Qwen3.5-4B}"
export NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$WORK/logs"
trap 'printf "Pipeline failed at line %s\n" "$LINENO" >&2; printf "failed\n" > "$WORK/status.txt"' ERR
printf 'preparing\n' > "$WORK/status.txt"
if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  [[ "$WAIT_FOR_PID" =~ ^[0-9]+$ ]] || exit 1
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 30; done
fi
if [[ -f "$WORK/train.complete" && -z "${RESUME_FROM:-}" ]]; then
  python - "$WORK/train/trainer_state.json" "${EPOCHS:-1}" <<'PY'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    epoch = json.load(handle)["epoch"]
if not math.isclose(float(epoch), float(sys.argv[2]), abs_tol=1e-6):
    raise SystemExit("Existing run has a different epoch count; set RESUME_FROM to continue it or use a new WORK")
PY
fi
python "$ROOT/tools/prepare_ost_stage_c.py" --out "$WORK/data" --audit
for split in train eval; do
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="${PREP_PORT:-29821}" \
    "$ROOT/tools/prepare_ost_stage_c.py" --split "$split" --model "$MODEL" --out "$WORK/data"
  EXTRA=()
  if [[ "$split" == train ]]; then EXTRA+=(--allow-missing); fi
  python "$ROOT/tools/prepare_ost_stage_c.py" --split "$split" --model "$MODEL" \
    --out "$WORK/data" --merge "${EXTRA[@]}"
done
printf 'training\n' > "$WORK/status.txt"
TRAIN_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM"); fi
if [[ ! -f "$WORK/train.complete" || -n "${RESUME_FROM:-}" ]]; then
  WORK="$WORK" OUT="$WORK/train" bash "$ROOT/scripts/train/v8_stage_c.sh" "${TRAIN_ARGS[@]}" \
    > "$WORK/logs/train.log" 2>&1
  printf 'completed\n' > "$WORK/train.complete"
fi
printf 'evaluating\n' > "$WORK/status.txt"
if [[ -n "${RESUME_FROM:-}" && -z "${EVAL_OUT:-}" ]]; then
  EVAL_OUT="$WORK/eval_resume_$(date +%Y%m%d_%H%M%S)"
fi
EVAL_OUT="${EVAL_OUT:-$WORK/eval}"
CKPT="$WORK/train" OUT="$EVAL_OUT" FEATURE_ROOT="$WORK/data/eval_features" \
  bash "$ROOT/scripts/evaluation/ost_v8_all.sh" > "$WORK/logs/eval.log" 2>&1
python "$ROOT/tools/summarize_ost_stage_c.py" --work "$WORK" --eval-dir "$EVAL_OUT" \
  --stage-b "${STAGE_B:-$ROOT/output/v8/stage_b}" > "$WORK/verification.log" 2>&1
printf 'completed\n' > "$WORK/status.txt"
