#!/usr/bin/env bash
# Ask the training queue to save its state and stop. Resume later with ./start_training.sh
cd "$(dirname "$0")"
touch results/STOP
echo "Stop requested. Waiting for the current run to save..."
for i in $(seq 1 90); do
  pgrep -f "train[.]py --config" > /dev/null || { echo "Training stopped. State saved. Resume with ./start_training.sh"; exit 0; }
  sleep 2
done
echo "Still running after 3 minutes; check results/logs/train_all.nohup"
