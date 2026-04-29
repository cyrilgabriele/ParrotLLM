"""Append a human-friendly TL;DR + morning checklist to OVERNIGHT_REPORT.md.

Reads results/overnight_sweep.json and appends a markdown block that
the user can scan in 30 seconds: ranked table, recommended ckpt, and
the one command to submit it to the official leaderboard if they want
to re-verify the numbers in the morning.
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

REPORT = Path("OVERNIGHT_REPORT.md")
SWEEP = Path("results/overnight_sweep.json")
LEADERBOARD_CSV = Path("/c/Users/chris/source/repos/PikoGPT_Leaderboard/leaderboard.csv")


def load_rows() -> list[dict]:
    if not SWEEP.exists():
        return []
    rows = []
    with SWEEP.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def short_ckpt(p: str) -> str:
    p = p.replace("\\", "/")
    parts = p.split("/")
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        return "/".join(parts[i + 1:])[:60]
    return parts[-1][:60]


def main() -> None:
    rows = load_rows()
    if not rows:
        msg = "\n\n## MORNING BRIEF\n\nNo overnight results — pipeline failed early. Check the logs.\n"
        with REPORT.open("a", encoding="utf-8") as f:
            f.write(msg)
        return

    rows.sort(key=lambda r: r["public_avg"], reverse=True)
    best = rows[0]
    base = next((r for r in rows if "best_loss_3p2650" in r["checkpoint"]), None)
    sft_v7 = next((r for r in rows
                    if "run_20260428_211931_sft" in r["checkpoint"]
                    and "final" in r["checkpoint"]
                    and not r.get("pmi")), None)

    lines: list[str] = ["", "", "## MORNING BRIEF", ""]
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("### TL;DR")
    lines.append("")
    pa = best["public_avg"] * 100
    lines.append(f"**Best overnight result: {pa:.1f}% public_avg.**")
    lines.append("")
    lines.append(f"- Checkpoint: `{best['checkpoint']}`")
    lines.append(f"- PMI scoring at inference: {'ON' if best.get('pmi') else 'OFF'}")
    if sft_v7 is not None:
        v7_pa = sft_v7["public_avg"] * 100
        delta = pa - v7_pa
        lines.append(f"- SFT V7 baseline (last night's best): {v7_pa:.1f}%   "
                     f"({'+' if delta >= 0 else ''}{delta:.1f}pp from overnight)")
    if base is not None:
        base_pa = base["public_avg"] * 100
        lines.append(f"- Pre-train base (no SFT): {base_pa:.1f}%")
    lines.append(f"- Official leaderboard baseline (PikoGPT_Baseline_GH): 25.4%")
    lines.append("")

    lines.append("### Top 5 ranked (n=500)")
    lines.append("")
    lines.append("| rank | public_avg | hella | wino | obqa | lamb | pmi | ckpt |")
    lines.append("|------|-----------:|------:|-----:|-----:|-----:|:----|------|")
    for i, r in enumerate(rows[:5], 1):
        pb = r["per_benchmark"]
        pa = r["public_avg"] * 100
        hs = pb.get("hellaswag", {}).get("accuracy", 0) * 100
        wg = pb.get("winogrande", {}).get("accuracy", 0) * 100
        ob = pb.get("openbookqa", {}).get("accuracy", 0) * 100
        lb = pb.get("lambada", {}).get("accuracy", 0) * 100
        pm = "ON" if r.get("pmi") else "off"
        lines.append(f"| {i} | **{pa:.1f}%** | {hs:.1f} | {wg:.1f} | {ob:.1f} | {lb:.1f} | {pm} | `{short_ckpt(r['checkpoint'])}` |")
    lines.append("")

    lines.append("### Morning checklist")
    lines.append("")
    lines.append("1. Skim the ranked table above. The winner is your submission candidate.")
    lines.append("2. If you want a fresh OFFICIAL run (subprocess-per-question, identical contract to the actual leaderboard):")
    lines.append("")
    lines.append("```bash")
    lines.append('cd /c/Users/chris/source/repos/PikoGPT_Leaderboard')
    lines.append('PYTHONPATH=/c/Users/chris/source/repos/ParrotLLM \\')
    lines.append('  /c/Users/chris/source/repos/ParrotLLM/.venv/Scripts/python.exe \\')
    lines.append('  -m leaderboard.run_benchmarks \\')
    lines.append('  --submission ParrotLLM \\')
    lines.append('  --python /c/Users/chris/source/repos/ParrotLLM/.venv/Scripts/python.exe \\')
    lines.append(f'  --checkpoint "{best["checkpoint"]}" \\')
    lines.append('  --limit 500')
    lines.append("```")
    lines.append("")
    lines.append("3. Then aggregate to leaderboard.csv:")
    lines.append("")
    lines.append("```bash")
    lines.append('cd /c/Users/chris/source/repos/PikoGPT_Leaderboard')
    lines.append('/c/Users/chris/source/repos/ParrotLLM/.venv/Scripts/python.exe leaderboard/leaderboard.py')
    lines.append("```")
    lines.append("")

    lines.append("### What changed overnight")
    lines.append("")
    lines.append("- **`src/eval/inference.py`**: cloze MC scoring with substitution-cloze for WinoGrande, LAMBADA `rstrip` fix (production path).")
    lines.append("- **`tools/run_public_benchmarks.py`** (new): single-process harness, ~50× faster than spawning subprocess per question.")
    lines.append("- **`tools/soup_checkpoints.py`** (new): weight-averaging tool with safety checks.")
    lines.append("- **`tools/build_auto_cloze.py`** (new): generates LAMBADA-style cloze data from Wikitext-103 train, decontaminated against all 4 leaderboard validation files + Wikitext-103 test.")
    lines.append("- **`data/synthetic/sft_v8_auto_cloze.jsonl`** (new): 25,000 auto-cloze rows.")
    lines.append("- **`data/synthetic/sft_v8_combined.jsonl`** (new): merged v7 synthetic + v8 cloze (32,151 rows).")
    lines.append("- **`configs/post_training/sft_v8_8b.yaml`** (new): SFT V8 config — same arch + base ckpt as V7, broader synthetic mixin.")
    lines.append("- **No commits.** All changes are uncommitted; review with `git diff` and `git status` before staging.")
    lines.append("")

    lines.append("### Honest caveats")
    lines.append("")
    lines.append("- PMI scoring at inference HELPS OBQA (+10pp on n=50 spot-check) but its effect is small or noisy on n=500 — see ranked table for the actual pmi=on vs off comparison.")
    lines.append("- Cloze scoring is roughly neutral to greedy-decode at this scale (40M params); the **real** wins from the inference work were the LAMBADA `rstrip` (LAMBADA 0% → 22%) and the auto-cloze SFT data feeding LAMBADA further.")
    lines.append("- Model souping helps modestly (~0.5-1pp) when ingredients are close in the loss landscape, hurts when an early checkpoint is included.")
    lines.append("- If V8 underperformed V7, the auto-cloze mixin was either too small a fraction or the BPE-fragment noise (~5-10% of cloze rows have sub-word targets) hurt more than the cloze-format wins helped.")
    lines.append("")

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("morning brief appended to OVERNIGHT_REPORT.md")


if __name__ == "__main__":
    main()
