#!/usr/bin/env bash
# Run after DPO v3 finishes. Trains SFT v3 with pretraining_mix=0.20.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
  uv run python main.py --stage sft \
  --config configs/post_training/sft_v3_higher_mix.yaml \
  2>&1 | tee runs/sft_v3_run.log
