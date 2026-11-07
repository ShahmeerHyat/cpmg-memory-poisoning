#!/usr/bin/env bash
# Train all DeBERTa models (CPMG, content-only baseline, ablations, cross-source). Sequential, one GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in cpmg_full content_only abl_no_prov abl_no_query abl_no_nbrs write_only cpmg_noasb; do
  if [ -f "results/runs/$c.json" ]; then echo "[skip] $c already done"; continue; fi
  .venv/bin/python train.py --config "configs/$c.yaml" 2>&1 | grep --line-buffered -vE "Warning|warn\(|LOAD REPORT|UNEXPECTED|MISSING|^Key|^---|^Notes|can be ignored|newly initialized|^\s*$|Loading weights" | tee -a results/logs/train_all.log
done
echo "ALL TRAINING DONE" | tee -a results/logs/train_all.log
