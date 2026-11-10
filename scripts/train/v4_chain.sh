#!/usr/bin/env bash
# GEO-Memory v4: Stage A -> Stage B -> VSI-Bench (16f memory + 8f visual, strict), sequential.
# A stage that fails or produces no model stops the chain (no later stage runs on bad input).
set -uo pipefail
source "$(dirname "$0")/v3_common.sh"
cd "$ROOT"
L="$ROOT/output/v4/logs"; mkdir -p "$L"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$L/chain.log"; }
export NUM_GPUS="${NUM_GPUS:-8}"

if [ ! -f output/v4/stage_a/model.safetensors ]; then
  log "Stage A start"
  bash scripts/train/v4_stage_a.sh > "$L/stage_a.log" 2>&1 || { log "Stage A FAILED"; exit 1; }
fi
[ -f output/v4/stage_a/model.safetensors ] || { log "Stage A produced no model"; exit 1; }
log "Stage A done"

if [ ! -f output/v4/stage_b/model.safetensors ]; then
  log "Stage B start"
  bash scripts/train/v4_stage_b.sh > "$L/stage_b.log" 2>&1 || { log "Stage B FAILED"; exit 1; }
fi
[ -f output/v4/stage_b/model.safetensors ] || { log "Stage B produced no model"; exit 1; }
log "Stage B done"

sleep 30   # let training processes release GPU memory
log "VSI eval start"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for try in 1 2 3; do   # eval resumes from predictions_rank*.json, so a retry only redoes the unfinished part
  NGPU="$NUM_GPUS" FRAMES=16 bash scripts/evaluation/run_vsi_geomemory.sh output/v4/stage_b >> "$L/vsi_eval_16f.log" 2>&1 && break
  log "VSI eval attempt $try failed; resuming"; sleep 20
done
[ -f output/v4/stage_b/vsi-eval-16f/eval_merge.log ] || { log "VSI eval FAILED"; exit 1; }
log "VSI eval done: $(grep MERGED output/v4/stage_b/vsi-eval-16f/eval_merge.log)"
