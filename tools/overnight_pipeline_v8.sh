#!/usr/bin/env bash
# Overnight pipeline (2026-04-28 night).
#
# Runs each phase serially, keeps every command's exit status, never
# aborts the whole pipeline on a single failure. Continuous status
# updates land in OVERNIGHT_REPORT.md so a human waking up sees the
# latest state regardless of where execution is.
#
# Hard rules baked in:
#   * No git commits, no pushes — file changes only.
#   * No existing checkpoint touched — only writes to runs/soups/ and
#     a new sft_v8 run dir.
#   * No benchmark validation data ever touches training — auto-cloze
#     is decontaminated against the four leaderboard files + wt103 test.

set +e
set -u
shopt -s extglob

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPORT="$ROOT/OVERNIGHT_REPORT.md"
RESULTS_JSON="$ROOT/results/overnight_sweep.json"
mkdir -p "$ROOT/results" "$ROOT/runs/soups" "$ROOT/data/synthetic"

PYTHON=".venv/Scripts/python.exe"
HARNESS="tools/run_public_benchmarks.py"

PRE_8B="runs/big_run/exp_c_8b/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt"
SFT_V6="runs/run_20260428_102441_sft/checkpoints/final_step_0002030_epoch_01_valloss_2p4207.pt"
DPO_V6="runs/run_20260428_104023_dpo/checkpoints/final_step_0000374_epoch_00_valloss_0p6445.pt"
SFT_V7="runs/run_20260428_211931_sft/checkpoints/final_step_0001966_epoch_01_valloss_2p4231.pt"
SFT_V7_900="runs/run_20260428_211931_sft/checkpoints/best_step_0000900_epoch_00_valloss_2p4668.pt"
SFT_V7_1000="runs/run_20260428_211931_sft/checkpoints/best_step_0001000_epoch_01_valloss_2p4576.pt"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$REPORT"
}

# Always start from a clean report header.
{
    echo "# Overnight pipeline report"
    echo "Started: $(date -Iseconds)"
    echo
    echo "Pipeline log follows. Final ranked summary at the bottom."
    echo
    echo '```'
} > "$REPORT"

bench() {
    local label="$1"
    local ckpt="$2"
    local pmi_flag="${3:-}"
    {
        echo
        echo "## bench: $label  pmi=${pmi_flag:-off}  ($(date +%H:%M:%S))"
        PYTHONPATH=. "$PYTHON" "$HARNESS" \
            --checkpoint "$ckpt" \
            --device cuda \
            --limit 500 \
            $pmi_flag \
            --out-json "$RESULTS_JSON" 2>&1
    } >> "$REPORT"
}

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 1: PMI ablation on existing checkpoints (n=500) ==="
> "$RESULTS_JSON"
bench "pre_8b   "          "$PRE_8B"
bench "sft_v6   "          "$SFT_V6"
bench "sft_v6+pmi "        "$SFT_V6"  --pmi
bench "dpo_v6   "          "$DPO_V6"
bench "dpo_v6+pmi "        "$DPO_V6"  --pmi
bench "sft_v7   "          "$SFT_V7"
bench "sft_v7+pmi "        "$SFT_V7"  --pmi

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 2: model souping ==="

V7_LATE_SOUP="runs/soups/sft_v7_late.pt"
V7_V6_SOUP="runs/soups/sft_v7_v6.pt"

PYTHONPATH=. "$PYTHON" tools/soup_checkpoints.py \
    --checkpoints "$SFT_V7" "$SFT_V7_1000" "$SFT_V7_900" \
    --out "$V7_LATE_SOUP" --allow-overwrite 2>&1 | tee -a "$REPORT"

PYTHONPATH=. "$PYTHON" tools/soup_checkpoints.py \
    --checkpoints "$SFT_V7" "$SFT_V6" \
    --out "$V7_V6_SOUP" --allow-overwrite 2>&1 | tee -a "$REPORT"

bench "soup_v7_late "     "$V7_LATE_SOUP"
bench "soup_v7_late+pmi" "$V7_LATE_SOUP" --pmi
bench "soup_v7_v6  "     "$V7_V6_SOUP"
bench "soup_v7_v6+pmi"   "$V7_V6_SOUP"  --pmi

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 3: SFT V8 training (auto-cloze mixin) ==="

V8_LOG="$ROOT/runs/v8_sft.log"
PRE_V8_RUNS=$(ls -d runs/run_* 2>/dev/null | sort -u)

log "launching SFT V8 (config sft_v8_8b.yaml). Log: $V8_LOG"
PYTHONPATH=. "$PYTHON" main.py --stage sft \
    --config configs/post_training/sft_v8_8b.yaml >"$V8_LOG" 2>&1
V8_RC=$?
log "SFT V8 finished with rc=$V8_RC"

# Find the new V8 run directory.
V8_RUN_DIR=""
for d in $(ls -dt runs/run_*_sft 2>/dev/null); do
    if ! grep -q "^$d$" <<<"$PRE_V8_RUNS"; then
        V8_RUN_DIR="$d"
        break
    fi
done
log "V8 run dir: ${V8_RUN_DIR:-MISSING}"

V8_FINAL=""
V8_BEST=""
if [[ -n "$V8_RUN_DIR" && -d "$V8_RUN_DIR/checkpoints" ]]; then
    V8_FINAL=$(ls -t "$V8_RUN_DIR/checkpoints/final_"*.pt 2>/dev/null | head -1)
    V8_BEST=$(ls -t "$V8_RUN_DIR/checkpoints/best_"*.pt 2>/dev/null | head -1)
fi
log "V8 final: ${V8_FINAL:-NOT FOUND}"
log "V8 best:  ${V8_BEST:-NOT FOUND}"

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 4: SFT V8 benchmarks ==="
if [[ -n "$V8_FINAL" ]]; then
    bench "sft_v8_final"       "$V8_FINAL"
    bench "sft_v8_final+pmi"   "$V8_FINAL"  --pmi
fi
if [[ -n "$V8_BEST" && "$V8_BEST" != "$V8_FINAL" ]]; then
    bench "sft_v8_best "       "$V8_BEST"
    bench "sft_v8_best+pmi "   "$V8_BEST"   --pmi
fi

# Optional V8 + V7 soup if V8 produced a usable checkpoint.
if [[ -n "$V8_FINAL" ]]; then
    V8_V7_SOUP="runs/soups/sft_v8_v7.pt"
    PYTHONPATH=. "$PYTHON" tools/soup_checkpoints.py \
        --checkpoints "$V8_FINAL" "$SFT_V7" \
        --out "$V8_V7_SOUP" --allow-overwrite 2>&1 | tee -a "$REPORT"
    bench "soup_v8_v7"          "$V8_V7_SOUP"
    bench "soup_v8_v7+pmi"      "$V8_V7_SOUP" --pmi
fi

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 5: building ranked summary ==="

PYTHONPATH=. "$PYTHON" - <<'PYEOF' 2>&1 | tee -a "$REPORT"
import json, os
rows = []
path = "results/overnight_sweep.json"
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
rows.sort(key=lambda r: r["public_avg"], reverse=True)
print()
print("=== RANKED RESULTS (n=500) ===")
print()
print(f"{'rank':<5}{'public_avg':<12}{'hella':<7}{'wino':<7}{'obqa':<7}{'lamb':<7}  {'pmi':<5} ckpt")
print("-" * 100)
for i, r in enumerate(rows, 1):
    pb = r["per_benchmark"]
    pa = f"{r['public_avg']*100:.1f}%"
    hs = f"{pb.get('hellaswag', {}).get('accuracy', 0)*100:.1f}"
    wg = f"{pb.get('winogrande', {}).get('accuracy', 0)*100:.1f}"
    ob = f"{pb.get('openbookqa', {}).get('accuracy', 0)*100:.1f}"
    lb = f"{pb.get('lambada', {}).get('accuracy', 0)*100:.1f}"
    pm = "ON" if r.get("pmi") else "off"
    ck = r["checkpoint"]
    # Compress the checkpoint path for readability
    ck = ck.replace("\\", "/")
    if "checkpoints/" in ck:
        ck = ck.split("checkpoints/")[-1]
    print(f"{i:<5}{pa:<12}{hs:<7}{wg:<7}{ob:<7}{lb:<7}  {pm:<5} {ck}")
print()
if rows:
    best = rows[0]
    print(f"BEST: public_avg = {best['public_avg']*100:.2f}%")
    print(f"      checkpoint = {best['checkpoint']}")
    print(f"      pmi        = {'ON' if best.get('pmi') else 'OFF'}")
PYEOF

echo '```' >> "$REPORT"

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 6: official leaderboard runner against best ckpt ==="
bash "$ROOT/tools/overnight_official_runner.sh" 2>&1 | tee -a "$REPORT"

# ──────────────────────────────────────────────────────────────────────
log "=== Phase 7: morning brief ==="
PYTHONPATH=. "$PYTHON" "$ROOT/tools/overnight_morning_brief.py" 2>&1 | tee -a "$REPORT"

log "=== Pipeline complete at $(date -Iseconds) ==="
