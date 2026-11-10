#!/usr/bin/env bash
# Cache ViT + CUT3R features for 288 VSI-Bench test scenes (8 GPU parallel).
# Run ONCE; then use eval_vsi_stage_a_ablation_cached.py on any checkpoint.
set -euo pipefail

ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7
export PYTHONPATH="$ROOT/src:/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/scripts/evaluation"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True

PY=/root/anaconda3/envs/vllm-cu129/bin/python
OUT_DIR="${OUT_DIR:-/data/juicefs_sharing_data/11188211/VIT-feather/v8_vit512/features/vsi_bench_test}"
NGPU="${NGPU:-8}"

pids=()
for ((r=0; r<NGPU; r++)); do
  CUDA_VISIBLE_DEVICES=$r TMPDIR=/tmp/bluecode/vsi_cache_$r \
    "$PY" "$ROOT/scripts/evaluation/cache_vsi_features.py" \
    --rank $r --world $NGPU --out_dir "$OUT_DIR" \
    --tmp_dir /tmp/bluecode/vsi_cache_tmp_$r \
    > /tmp/bluecode/vsi_cache_rank$r.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if ((fail)); then
  echo "Some ranks failed. Check logs:"
  for ((r=0; r<NGPU; r++)); do echo "--- rank $r ---"; tail -5 /tmp/bluecode/vsi_cache_rank$r.log; done
  exit 1
fi
echo "All done. Cached to $OUT_DIR"
# Merge index files
cat "$OUT_DIR"/scene_index.rank*.jsonl > "$OUT_DIR/scene_index.jsonl"
echo "Merged index: $OUT_DIR/scene_index.jsonl"
