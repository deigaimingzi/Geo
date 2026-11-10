#!/usr/bin/env bash
# FRAMES=16 (default): strict main protocol, 16 memory + 8 visual tokens-frames. FRAMES=32: generalisation test only.
# Usage: NGPU=8 bash scripts/evaluation/run_vsi_geomemory.sh output/v3/stage_b [extra eval args e.g. --max_samples 16]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
CKPT="$(cd "$1" && pwd)"; shift || true
PYTHON="${PYTHON:-/root/anaconda3/envs/geo/bin/python}"
NGPU="${NGPU:-8}"; FRAMES="${FRAMES:-16}"; STRICT=""; [[ "$FRAMES" == 16 ]] && STRICT="--strict"   # 16 = main protocol (16 memory + 8 visual)
OUT="${OUT:-$CKPT/vsi-eval-${FRAMES}f}"; mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false OPENCV_FOR_THREADS_NUM=1 OMP_NUM_THREADS=4
pids=()
for r in ${RANKS:-$(seq 0 $((NGPU-1)))}; do
  CUDA_VISIBLE_DEVICES=$r "$PYTHON" scripts/evaluation/eval_vsi_geomemory.py --ckpt "$CKPT" --out_dir "$OUT" \
    --rank $r --world $NGPU --max_num_frames "$FRAMES" $STRICT "$@" > "$OUT/eval_rank$r.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail == 1 ]] && { tail -n 20 "$OUT"/eval_rank*.log; exit 1; }
"$PYTHON" /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation/merge_vsi_mra.py --out_dir "$OUT" --world $NGPU | tee "$OUT/eval_merge.log"
