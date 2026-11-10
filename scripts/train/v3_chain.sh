#!/usr/bin/env bash
# Wait for Stage A to finish, then launch Stage B (v3).
set -uo pipefail
source "$(dirname "$0")/v3_common.sh"
cd "$ROOT"
while pgrep -f "geomemory.train" > /dev/null; do sleep 30; done
if [ ! -f output/v3/stage_a/model.safetensors ]; then echo "Stage A produced no model; Stage B not started" >> output/v3/logs/chain.log; exit 1; fi
sleep 20
bash scripts/train/v3_stage_b.sh > output/v3/logs/stage_b.log 2>&1
