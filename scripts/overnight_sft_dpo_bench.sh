#!/usr/bin/env bash
# Autonomous overnight pipeline: new-mix SFT -> continuation-pair DPO -> leaderboard bench.
#
# Inputs (must already exist):
#   configs/posttraining/sft_benchmark.yaml       (Plan A SFT recipe)
#   configs/posttraining/dpo_continuation.yaml    (Plan B DPO recipe)
#
# Outputs:
#   data/posttraining/sft_mix_benchmark/                (prepared SFT data)
#   runs/posttraining/sft_benchmark/<run-id>/           (SFT checkpoint)
#   data/posttraining/dpo_pairs_continuation/           (prepared DPO pairs)
#   runs/posttraining/dpo_continuation/<run-id>/        (DPO checkpoint)
#   runs/overnight_sft_dpo_bench/                       (orchestration logs)
#   runs/benchmarks/<sha>__<ckpt>__quick.json           (per-stage bench results)
#   runs/overnight_sft_dpo_bench/summary.md             (final report)

set -euo pipefail

ROOT="/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM"
cd "$ROOT"

LOG_DIR="runs/overnight_sft_dpo_bench"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/run.log"; }

log "=== Overnight SFT/DPO/bench started ==="

# ---------------- Step 1: SFT download (idempotent — reuses cached snapshots) ----------------
log "Step 1/7: SFT download"
uv run python main.py --stage sft-download --config configs/posttraining/sft_benchmark.yaml >> "$LOG_DIR/01_sft_download.log" 2>&1
log "  download done"

# ---------------- Step 2: SFT prepare ----------------
log "Step 2/7: SFT prepare"
if [[ -f "data/posttraining/sft_mix_benchmark/sft_mix.jsonl" ]] || ls data/posttraining/sft_mix_benchmark/*.jsonl > /dev/null 2>&1; then
  log "  prepared SFT mix already exists, skipping"
else
  uv run python main.py --stage sft-prepare --config configs/posttraining/sft_benchmark.yaml >> "$LOG_DIR/02_sft_prepare.log" 2>&1
fi
log "  prepare done"

# ---------------- Step 3: SFT train ----------------
log "Step 3/7: SFT train (this is the long one)"
SFT_CKPT_PROBE=$(find runs/posttraining/sft_benchmark -name "best_loss_*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$SFT_CKPT_PROBE" ]]; then
  log "  SFT checkpoint already exists: $SFT_CKPT_PROBE"
else
  uv run python main.py --stage sft --config configs/posttraining/sft_benchmark.yaml >> "$LOG_DIR/03_sft_train.log" 2>&1
fi

SFT_CKPT=$(find runs/posttraining/sft_benchmark -name "best_loss_*.pt" -type f 2>/dev/null | head -1)
if [[ -z "$SFT_CKPT" ]]; then
  log "FATAL: SFT training produced no best_loss_*.pt checkpoint"
  exit 1
fi
log "  SFT best checkpoint: $SFT_CKPT"

# ---------------- Step 4: Inject SFT checkpoint into DPO config ----------------
log "Step 4/7: Wire SFT checkpoint into DPO config"
uv run python - <<PY
import re
from pathlib import Path
cfg_path = Path("configs/posttraining/dpo_continuation.yaml")
src = cfg_path.read_text()
src = re.sub(
    r"^  reference_checkpoint:.*\$",
    f"  reference_checkpoint: $SFT_CKPT",
    src,
    flags=re.M,
)
cfg_path.write_text(src)
print("DPO config updated with SFT checkpoint")
PY
log "  DPO config wired"

# ---------------- Step 5: DPO prepare ----------------
log "Step 5/7: DPO prepare (continuation pairs)"
if [[ -f "data/posttraining/dpo_pairs_continuation/train.jsonl" ]]; then
  log "  DPO pairs already exist, skipping prepare"
else
  uv run python main.py --stage dpo-prepare --config configs/posttraining/dpo_continuation.yaml >> "$LOG_DIR/05_dpo_prepare.log" 2>&1
fi
log "  DPO prepare done"

# ---------------- Step 6: DPO train ----------------
log "Step 6/7: DPO train"
DPO_CKPT_PROBE=$(find runs/posttraining/dpo_continuation -name "*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$DPO_CKPT_PROBE" ]]; then
  log "  DPO checkpoint already exists: $DPO_CKPT_PROBE"
else
  uv run python main.py --stage dpo --config configs/posttraining/dpo_continuation.yaml >> "$LOG_DIR/06_dpo_train.log" 2>&1
fi

DPO_CKPT=$(find runs/posttraining/dpo_continuation -name "best_*.pt" -type f 2>/dev/null | head -1)
if [[ -z "$DPO_CKPT" ]]; then
  # Some DPO trainers save as last_*.pt. Fall back.
  DPO_CKPT=$(find runs/posttraining/dpo_continuation -name "*.pt" -type f 2>/dev/null | head -1)
fi
if [[ -z "$DPO_CKPT" ]]; then
  log "WARN: DPO training produced no checkpoint; will benchmark SFT only"
fi
log "  DPO checkpoint: ${DPO_CKPT:-<none>}"

# ---------------- Step 7: Leaderboard benchmarks ----------------
log "Step 7/7: Leaderboard benchmarks"
SUBMISSION_DIR="$ROOT/Submissions/parrotlabs_parrotllm"
if [[ ! -d "$SUBMISSION_DIR" ]]; then
  log "WARN: submission dir not found at $SUBMISSION_DIR; skipping bench"
else
  cd "$ROOT/external/PikoGPT_Leaderboard"
  for ckpt_label in "sft:$SFT_CKPT" "dpo:${DPO_CKPT:-NONE}"; do
    label="${ckpt_label%%:*}"
    ckpt="${ckpt_label#*:}"
    if [[ "$ckpt" == "NONE" ]]; then
      log "  skipping $label (no checkpoint)"
      continue
    fi
    log "  benching $label: $ckpt"
    BENCH_OUT="$ROOT/$LOG_DIR/07_bench_${label}.log"
    # Copy checkpoint into the submission's runs/ for the runner to find.
    target_path="Submissions/parrotlabs_parrotllm/runs/overnight_${label}.pt"
    cp "$ckpt" "$target_path" 2>/dev/null || true
    uv run python -m leaderboard.run_benchmarks \
      --submission parrotlabs_parrotllm \
      --checkpoint "runs/overnight_${label}.pt" \
      --bench hellaswag winogrande openbookqa lambada \
      --limit 200 \
      > "$BENCH_OUT" 2>&1 || log "    bench failed for $label (see $BENCH_OUT)"
  done
  cd "$ROOT"
fi

# ---------------- Step 8: Summary ----------------
log "Writing summary"
cat > "$LOG_DIR/summary.md" <<EOF
# Overnight SFT/DPO/Bench Summary

Started: $(head -1 "$LOG_DIR/run.log")
Finished: $(ts)

## Checkpoints

- SFT best: \`$SFT_CKPT\`
- DPO best: \`${DPO_CKPT:-<none>}\`

## Logs

- Step 1 (SFT download): \`$LOG_DIR/01_sft_download.log\`
- Step 2 (SFT prepare):  \`$LOG_DIR/02_sft_prepare.log\`
- Step 3 (SFT train):    \`$LOG_DIR/03_sft_train.log\`
- Step 5 (DPO prepare):  \`$LOG_DIR/05_dpo_prepare.log\`
- Step 6 (DPO train):    \`$LOG_DIR/06_dpo_train.log\`
- Step 7 (bench):        \`$LOG_DIR/07_bench_*.log\`

## Bench results (limit=200 per benchmark)

EOF

for label in sft dpo; do
  bench_log="$LOG_DIR/07_bench_${label}.log"
  if [[ -f "$bench_log" ]]; then
    echo "### $label" >> "$LOG_DIR/summary.md"
    grep -E "%|invalid" "$bench_log" | head -20 >> "$LOG_DIR/summary.md" || true
    echo "" >> "$LOG_DIR/summary.md"
  fi
done

log "=== All steps complete; see $LOG_DIR/summary.md ==="
