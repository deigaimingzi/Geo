#!/usr/bin/env bash
# Evaluate every Stage A checkpoint (held-out UV, fixed 299 val scenes) as soon as it is written.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
OUT="${OUT:-output/v4/stage_a}"; EV=output/v4/eval; VAL="${VAL:-/tmp/bluecode/val300_16.jsonl}"
while true; do
  for d in "$OUT"/checkpoint-*; do
    [ -f "$d/model.safetensors" ] || continue
    s="${d##*-}"; r="$EV/step${s}_val.json"; [ -f "$r" ] && continue
    sleep 20
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" PYTHONPATH="$ROOT/src" python scripts/evaluation/eval_geometry_heldout.py \
      --ckpt "$d" --config "$OUT/geomemory_config.json" --manifest "$VAL" --max_samples 300 --out "$r" > "$EV/step${s}.log" 2>&1
  done
  [ -f "$OUT/model.safetensors" ] && [ -f "$EV/final_val.json" ] && break
  if [ -f "$OUT/model.safetensors" ] && [ ! -f "$EV/final_val.json" ]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" PYTHONPATH="$ROOT/src" python scripts/evaluation/eval_geometry_heldout.py \
      --ckpt "$OUT" --manifest "$VAL" --max_samples 300 --out "$EV/final_val.json" > "$EV/final.log" 2>&1
  fi
  sleep 60
done
