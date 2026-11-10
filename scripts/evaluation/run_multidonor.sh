#!/usr/bin/env bash
# Multi-donor validation of Delta_State on v3 Stage B (original fusion).
# Each rank processes a slice of samples, testing multiple donor types.
# Usage: NGPU=8 bash scripts/evaluation/run_multidonor.sh output/v3/stage_b [extra args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
CKPT="$(cd "$1" && pwd)"; shift || true
PYTHON="${PYTHON:-/root/anaconda3/envs/geo/bin/python}"
NGPU="${NGPU:-8}"; FRAMES="${FRAMES:-16}"
OUT="${OUT:-$CKPT/multidonor-${FRAMES}f}"; mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
export OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=4
pids=()
for r in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$r "$PYTHON" scripts/evaluation/eval_vsi_multidonor.py \
    --ckpt "$CKPT" --out_dir "$OUT" \
    --rank $r --world $NGPU --max_num_frames "$FRAMES" \
    --max_samples 200 --num_donors 5 "$@" \
    > "$OUT/eval_rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
if [[ $fail == 1 ]]; then
  tail -n 30 "$OUT"/eval_rank*.log
  exit 1
fi
echo "All ranks done. Per-rank summaries:"
for r in $(seq 0 $((NGPU-1))); do
  echo "=== rank$r ==="
  cat "$OUT/multidonor_summary_rank$r.json" 2>/dev/null || echo "(no summary)"
done
