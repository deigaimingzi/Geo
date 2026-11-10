#!/bin/bash
# Full OST-Bench evaluation for 3 compared models, 8-GPU parallel.
# Usage: bash run_ost_baseline_full.sh [model_name]
#   model_name: sensenova | spatialstack | spatiolm | all  (default: all)
set -uo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7/src:/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/code/SOP-QV-MH-ablation/src

SCRIPT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7/scripts/evaluation/eval_ost_baseline.py
OUT_ROOT=/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/CVPR/my-method/GeoMemory-v7/output/v8/ost_baseline_eval
NUM_GPUS=8

run_model() {
    local NAME=$1
    local MODEL_TAG=$2
    local MODEL_PATH=$3
    local OUT="${OUT_ROOT}/${NAME}-full"
    mkdir -p "$OUT/logs"

    echo "================================================================"
    echo "  OST-Bench Full Eval — ${NAME}"
    echo "  Model: ${MODEL_PATH}"
    echo "  Output: ${OUT}"
    echo "  GPUs: ${NUM_GPUS}"
    echo "  Start: $(date)"
    echo "================================================================"

    # Launch 8 ranks in parallel, each on its own GPU
    for rank in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$rank python3 "$SCRIPT" \
            --model "$MODEL_TAG" \
            --model_path "$MODEL_PATH" \
            --out_dir "$OUT" \
            --max_new_tokens 512 \
            --limit 0 \
            --rank "$rank" --world "$NUM_GPUS" \
            --scorer official \
            > "$OUT/logs/rank${rank}.log" 2>&1 &
    done
    echo "Launched ${NUM_GPUS} ranks for ${NAME}. PIDs: $(jobs -p)"
    wait
    echo "  ${NAME} inference complete: $(date)"

    # Merge
    echo "  Merging ${NAME}..."
    CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT" \
        --model "$MODEL_TAG" \
        --model_path "$MODEL_PATH" \
        --out_dir "$OUT" \
        --merge --world "$NUM_GPUS" --scorer official \
        > "$OUT/logs/merge.log" 2>&1
    echo "  ${NAME} merge complete."
    echo ""
}

TARGET=${1:-all}

if [ "$TARGET" = "sensenova" ] || [ "$TARGET" = "all" ]; then
    run_model "sensenova" "internvl2" \
        "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/compared-model/SenseNova-SI-1.3-InternVL3-8B"
fi

if [ "$TARGET" = "spatialstack" ] || [ "$TARGET" = "all" ]; then
    run_model "spatialstack" "qwen3_5" \
        "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/compared-model/SpatialStack-Qwen3.5-4B"
fi

if [ "$TARGET" = "spatiolm" ] || [ "$TARGET" = "all" ]; then
    run_model "spatiolm" "spatiolm" \
        "/data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188211/AAAI-Geo/compared-model/SpatioLM-Understanding-SenseNovaSI"
fi

echo "================================================================"
echo "  All evaluations complete! $(date)"
echo "  Results: ${OUT_ROOT}"
echo "================================================================"
