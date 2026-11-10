#!/usr/bin/env bash
# Eval epoch 1/2/3 checkpoints (step 200/400/600) of the 5-epoch run in output/v3/ost_stage_c/train, sequentially, same protocol.
set -euo pipefail
source "$(dirname "$0")/v3_common.sh"
W="$ROOT/output/v3/ost_stage_c"; RUN="$W/train"; NUM_GPUS="${NUM_GPUS:-8}"
cd "$ROOT"
for E in 1 2 3; do
  CK="$RUN/checkpoint-$((E * 200))"
  [ -f "$CK/model.safetensors" ] || { echo "[chain] missing $CK"; exit 1; }
  [ -f "$CK/geomemory_config.json" ] || cp "$RUN/geomemory_config.json" "$CK/geomemory_config.json"
  EV="$W/eval_5ep_epoch${E}"; mkdir -p "$EV"
  echo "[chain] $(date +%T) eval epoch $E ($CK)"
  PIDS=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/evaluation/eval_ost_geomemory.py --ckpt "$CK" --out_dir "$EV" \
      --rank "$i" --world "$NUM_GPUS" > "$W/logs/eval_5ep_e${E}_$i.log" 2>&1 &
    PIDS+=($!)
  done
  FAIL=0; for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
  [ "$FAIL" -eq 0 ] || { echo "[chain] epoch $E: a rank failed"; exit 1; }
  python scripts/evaluation/eval_ost_geomemory.py --ckpt "$CK" --out_dir "$EV" --merge > /dev/null
  python scripts/evaluation/rescore_ost.py "$EV" > "$EV/rescore.log"
  echo "[chain] epoch $E overall: $(python -c "import json;print(json.load(open('$EV/metrics.json'))['overall'])")"
done
python - <<PY
import json
W="$W"; rows={e: json.load(open(f"{W}/eval_5ep_epoch{e}/metrics.json")) for e in (1,2,3)}
for k in [k for k in rows[1] if k!="num_records"]: print(k.ljust(40), *[f"{rows[e][k]:8.2f}" for e in rows])
json.dump(rows, open(f"{W}/eval_5ep_ep123_summary.json","w"), indent=2)
PY
echo "[chain] done"
