#!/usr/bin/env bash
# Train all DeBERTa models on dataset v2. Sequential, one GPU. Safe to stop and rerun:
#   ./stop_training.sh        saves the current run's state within a few seconds and ends the queue
#   ./start_training.sh       starts or resumes the queue in the background (finished runs are skipped,
#                             an interrupted run continues from its last saved step)
# State is also saved every 150 optimiser steps (~2 min), so a crash or power loss costs at most that.
# Pass 1: seed 42 for every config. Pass 2: seeds 43, 44 for the three main models.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs
rm -f results/STOP
run() {  # config seed
  local n="$1"; [ "$2" != 42 ] && n="$1_s$2"
  if [ -f results/STOP ]; then echo "[queue] stopped before $n" | tee -a results/logs/train_all.log; exit 0; fi
  if [ -f "results/runs/$n.json" ]; then echo "[skip] $n already done"; return; fi
  .venv/bin/python train.py --config "configs/$1.yaml" --seed "$2" 2>&1 \
    | grep --line-buffered -E "^\\[(cpmg|content|abl|write)|Error|Traceback|CUDA" | tee -a results/logs/train_all.log
}
for c in cpmg_full content_only abl_no_query abl_no_nbrs abl_no_prov write_only cpmg_noasb cpmg_noehr; do run "$c" 42; done
for s in 43 44; do for c in cpmg_full content_only abl_no_prov; do run "$c" "$s"; done; done
[ -f results/STOP ] || echo "ALL TRAINING DONE" | tee -a results/logs/train_all.log
