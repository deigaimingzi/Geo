#!/usr/bin/env bash
# Full CUT3R teacher export (pts3d_self/other, conf_self/other, pose, K), 16 frames, 8 GPUs.
set -uo pipefail
source "$(dirname "$0")/v3_common.sh"
cd "$ROOT"
OUT="${OUT:-/data/juicefs_sharing_data/public_data/11186230/Qinjia-2026/datasets/GeoMemory/datasets/v4/cut3r_full}"
mkdir -p "$OUT" output/v4/logs
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=1
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g /root/anaconda3/envs/geo/bin/python tools/export_cut3r_full.py \
    --manifest output/v4/manifests/cut3r_scenes.jsonl --out-dir "$OUT" --gpu 0 --shard $g --nshard 8 \
    >> output/v4/logs/export_cut3r_$g.log 2>&1 &
done
wait
echo "export finished: $(ls "$OUT" | grep -vc tmp) files"
