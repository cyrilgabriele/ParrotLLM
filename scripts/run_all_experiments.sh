#!/bin/bash
# Run all 6 dataset experiments sequentially
# Best HPs from Trial 10 of 40M HP tuning
source $HOME/.local/bin/env
cd /mnt/c/Users/chris/source/repos/ParrotLLM

for ds in c a b d e f; do
    echo "============================================"
    echo "Starting experiment: exp_${ds}"
    echo "============================================"
    uv run python main.py --stage train --config configs/training/train_exp_${ds}.yaml
    echo "Finished experiment: exp_${ds}"
    echo ""
done

echo "ALL EXPERIMENTS COMPLETE"
