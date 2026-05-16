#!/usr/bin/env bash
# Run the OFFICIAL PikoGPT leaderboard runner against the best checkpoint
# from the overnight sweep. Appends results to OVERNIGHT_REPORT.md.
#
# Trigger after the main pipeline finishes — kicks off subprocess-per-
# question scoring (the contract the actual leaderboard uses) so the
# numbers in the morning are submission-grade, not just my fast-harness
# approximations.

set +e
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LB_DIR="/c/Users/chris/source/repos/PikoGPT_Leaderboard"

REPORT="$ROOT/OVERNIGHT_REPORT.md"
RESULTS_JSON="$ROOT/results/overnight_sweep.json"
PYTHON="$ROOT/.venv/Scripts/python.exe"

# Pick the best checkpoint by public_avg from the sweep results.
BEST_INFO=$(PYTHONPATH="$ROOT" "$PYTHON" - <<'PYEOF'
import json, os, sys
path = "results/overnight_sweep.json"
if not os.path.exists(path):
    print("MISSING", file=sys.stderr); sys.exit(1)
rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
if not rows:
    print("EMPTY", file=sys.stderr); sys.exit(1)
rows.sort(key=lambda r: r["public_avg"], reverse=True)
best = rows[0]
ck = best["checkpoint"].replace("\\", "/")
print(f"{best['public_avg']:.4f}\t{int(bool(best.get('pmi')))}\t{ck}")
PYEOF
)

if [[ -z "$BEST_INFO" ]]; then
    {
        echo
        echo "## OFFICIAL RUNNER: no overnight results found, skipping."
    } >> "$REPORT"
    exit 1
fi

PA=$(echo "$BEST_INFO" | cut -f1)
PMI=$(echo "$BEST_INFO" | cut -f2)
CKPT=$(echo "$BEST_INFO" | cut -f3)
PMI_LABEL="off"; [[ "$PMI" == "1" ]] && PMI_LABEL="on"

# The official runner has no --pmi flag — it just invokes our main.py.
# main.py's leaderboard mode reads its own behavior; whether PMI helped
# in the sweep, the official run will reflect whatever main.py does
# today (currently PMI is OFF in run_inference because it requires a
# flag). We surface this in the report so the user knows.

{
    echo
    echo "## OFFICIAL RUNNER (n=500)"
    echo "Best harness ckpt: \`$CKPT\`"
    echo "Best harness public_avg: $(awk "BEGIN { printf \"%.2f\", $PA*100 }")% (pmi=$PMI_LABEL in harness)"
    echo
    echo "Note: the official runner invokes main.py which does NOT take"
    echo "a --pmi flag. PMI is therefore OFF for the official numbers"
    echo "below. The cloze MC scoring path and LAMBADA rstrip fix ARE"
    echo "active (both baked into run_inference unconditionally)."
    echo
    echo '```'
} >> "$REPORT"

cd "$LB_DIR" || exit 2

OFFICIAL_LOG="/tmp/overnight_official_runner.log"
PYTHONPATH="$ROOT" "$PYTHON" -m leaderboard.run_benchmarks \
    --submission ParrotLLM \
    --python "$PYTHON" \
    --checkpoint "$CKPT" \
    --limit 500 \
    > "$OFFICIAL_LOG" 2>&1
RC=$?

{
    cat "$OFFICIAL_LOG" 2>/dev/null
    echo
    echo "official runner rc=$RC"
    echo '```'
    echo
    echo "### Aggregated leaderboard.csv after this run:"
    cd "$LB_DIR" && PYTHONPATH="$ROOT" "$PYTHON" leaderboard/leaderboard.py 2>&1 | head -20
} >> "$REPORT"

echo "official runner done rc=$RC"
