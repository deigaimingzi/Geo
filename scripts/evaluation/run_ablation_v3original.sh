#!/usr/bin/env bash
# V3 Stage B memory ablation using RECONSTRUCTED original fusion (2-layer output_projection).
# Usage: bash scripts/evaluation/run_ablation_v3original.sh output/v3/stage_b [extra args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
CKPT="$(cd "$1" && pwd)"; shift || true
PYTHON="${PYTHON:-/root/anaconda3/envs/geo/bin/python}"
NGPU="${NGPU:-8}"; FRAMES="${FRAMES:-16}"
OUT="${OUT:-$CKPT/ablation-original-${FRAMES}f}"; mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false
export OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=4
pids=()
for r in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$r "$PYTHON" scripts/evaluation/eval_vsi_ablation_v3original.py \
    --ckpt "$CKPT" --out_dir "$OUT" \
    --rank $r --world $NGPU --max_num_frames "$FRAMES" "$@" \
    > "$OUT/eval_rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
if [[ $fail == 1 ]]; then
  tail -n 30 "$OUT"/eval_rank*.log
  exit 1
fi
"$PYTHON" scripts/evaluation/aggregate_ablation.py --out_dir "$OUT" --world $NGPU \
  | tee "$OUT/aggregate.log"
