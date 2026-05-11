#!/usr/bin/env bash
# One-shot chat training pipeline: SFT (chat data) → DPO (HH-RLHF).
# Idempotent: each step is skipped if its output already exists.
# Usage: nohup bash scripts/chat_sft_dpo.sh > runs/chat_sft_dpo.log 2>&1 &

set -euo pipefail

ROOT="/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM"
cd "$ROOT"

CONFIG_SFT="configs/posttraining/sft_chat.yaml"
CONFIG_DPO="configs/posttraining/dpo_chat.yaml"
LOG_DIR="runs/chat_sft_dpo_logs"
mkdir -p "$LOG_DIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/run.log"; }

log "=== Chat SFT+DPO pipeline started ==="

# ---- Step 1: SFT prepare ----
log "Step 1/5: SFT prepare"
if ls data/posttraining/sft_chat_mix/*.jsonl > /dev/null 2>&1; then
  log "  SFT mix already exists, skipping"
else
  uv run python main.py --stage sft-prepare --config "$CONFIG_SFT" \
    >> "$LOG_DIR/01_sft_prepare.log" 2>&1
fi
log "  SFT prepare done"

# ---- Step 2: SFT train ----
log "Step 2/5: SFT train"
SFT_CKPT_PROBE=$(find runs/posttraining/sft_chat -name "best_loss_*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$SFT_CKPT_PROBE" ]]; then
  log "  SFT checkpoint already exists: $SFT_CKPT_PROBE"
else
  uv run python main.py --stage sft --config "$CONFIG_SFT" \
    >> "$LOG_DIR/02_sft_train.log" 2>&1
fi

SFT_CKPT=$(find runs/posttraining/sft_chat -name "best_loss_*.pt" -type f 2>/dev/null \
  | sort | head -1)
if [[ -z "$SFT_CKPT" ]]; then
  log "FATAL: SFT produced no best_loss_*.pt checkpoint"
  exit 1
fi
log "  SFT best checkpoint: $SFT_CKPT"

# ---- Step 3: Inject SFT checkpoint into DPO config ----
log "Step 3/5: Wire SFT checkpoint into DPO config"
uv run python - <<PY
import re
from pathlib import Path
cfg_path = Path("$CONFIG_DPO")
src = cfg_path.read_text()
src = re.sub(
    r"^  reference_checkpoint:.*$",
    f"  reference_checkpoint: $SFT_CKPT",
    src,
    flags=re.M,
)
cfg_path.write_text(src)
print("DPO config updated:", "$SFT_CKPT")
PY
log "  DPO config wired"

# ---- Step 4: DPO prepare ----
log "Step 4/5: DPO prepare (HH-RLHF pairs)"
if [[ -f "data/posttraining/dpo_chat_pairs/train.jsonl" ]]; then
  log "  DPO pairs already exist, skipping"
else
  uv run python main.py --stage dpo-prepare --config "$CONFIG_DPO" \
    >> "$LOG_DIR/04_dpo_prepare.log" 2>&1
fi
log "  DPO prepare done"

# ---- Step 5: DPO train ----
log "Step 5/5: DPO train"
DPO_CKPT_PROBE=$(find runs/posttraining/dpo_chat -name "*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$DPO_CKPT_PROBE" ]]; then
  log "  DPO checkpoint already exists: $DPO_CKPT_PROBE"
else
  uv run python main.py --stage dpo --config "$CONFIG_DPO" \
    >> "$LOG_DIR/05_dpo_train.log" 2>&1
fi

DPO_CKPT=$(find runs/posttraining/dpo_chat -name "best_loss_*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -z "$DPO_CKPT" ]]; then
  DPO_CKPT=$(find runs/posttraining/dpo_chat -name "*.pt" -type f 2>/dev/null | head -1 || true)
fi
if [[ -z "$DPO_CKPT" ]]; then
  log "FATAL: DPO produced no checkpoint"
  exit 1
fi

log "=== Pipeline complete ==="
log "Final chat checkpoint: $DPO_CKPT"
log "Launch demo: uv run python main.py --stage chat --config configs/chat/chat_demo.yaml"
