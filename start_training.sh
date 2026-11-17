#!/usr/bin/env bash
# Start or resume the training queue in the background. Progress: tail -f results/logs/train_all.nohup
cd "$(dirname "$0")"
if pgrep -f "train[.]py --config" > /dev/null; then echo "Training is already running."; exit 0; fi
mkdir -p results/logs
setsid nohup ./run_all_training.sh >> results/logs/train_all.nohup 2>&1 < /dev/null &
echo "Training started in the background. Progress: tail -f results/logs/train_all.nohup"
