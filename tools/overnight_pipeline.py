"""Overnight pipeline: SFT v5 (on 8B base) -> DPO v5 -> full eval suite.

Run with: PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
    uv run python -m tools.overnight_pipeline 2>&1 | tee runs/overnight.log

The script is fail-tolerant: each phase logs its own error and continues
to the next so the user wakes up to as much data as possible even if one
stage breaks.

Phases:
  1. SFT v5 — train Alpaca SFT on the 8B base
  2. DPO v5 — train DPO on the SFT v5 best checkpoint
  3. Update perplexity_sweep + benchmark_pipeline + brutal_test pipelines
     to include sft_v5 / dpo_v5
  4. Run perplexity_sweep
  5. Run benchmark_pipeline (n=500)
  6. Run brutal_test (DPO v5 best)
  7. Write docs/post_training/overnight_results.md with the full table
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list[str], stage: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"START {stage} -> {log_path}")
    log(f"  cmd: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(cmd, cwd=str(REPO), env=ENV,
                              stdout=fp, stderr=subprocess.STDOUT)
    log(f"END   {stage} (exit {proc.returncode})")
    return proc.returncode


def latest_run_dir(prefix_glob: str) -> Path | None:
    runs = sorted(glob.glob(str(REPO / "runs" / prefix_glob)))
    return Path(runs[-1]) if runs else None


def best_checkpoint(run_dir: Path) -> Path | None:
    """Lowest val-loss best_*.pt under run_dir/checkpoints/."""
    candidates = list((run_dir / "checkpoints").glob("best_*.pt"))
    if not candidates:
        return None
    def _val(p: Path) -> float:
        m = re.search(r"valloss_(\d+)p(\d+)|loss_(\d+)p(\d+)", p.name)
        if not m:
            return float("inf")
        a, b = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
        return float(f"{a}.{b}")
    return min(candidates, key=_val)


def update_yaml_field(path: Path, key: str, value: str):
    text = path.read_text()
    text = re.sub(rf"^( *{key}: ).*$", rf"\1{value}", text, count=1, flags=re.MULTILINE)
    path.write_text(text)
    log(f"  patched {path.name}: {key} = {value}")


def main() -> int:
    runs_dir = REPO / "runs"
    runs_dir.mkdir(exist_ok=True)
    pipeline_log = runs_dir / "overnight"
    pipeline_log.mkdir(exist_ok=True)

    log("=" * 70)
    log("OVERNIGHT PIPELINE START")
    log("=" * 70)

    # ── Phase 1: SFT v5 on 8B base ───────────────────────────────────
    sft_cfg = REPO / "configs/post_training/sft_v5_8b.yaml"
    rc = run(
        ["uv", "run", "python", "main.py",
         "--stage", "sft", "--config", str(sft_cfg)],
        "sft_v5_8b", pipeline_log / "01_sft_v5.log",
    )
    if rc != 0:
        log("ABORT: SFT v5 failed; pipeline halting before DPO.")
        return rc

    sft_run = latest_run_dir("run_*_sft")
    if sft_run is None:
        log("ABORT: could not locate SFT run dir.")
        return 1
    sft_best = best_checkpoint(sft_run)
    if sft_best is None:
        log(f"ABORT: no best checkpoint in {sft_run}.")
        return 1
    log(f"SFT v5 best: {sft_best}")

    # ── Phase 2: DPO v5 on SFT v5 best ───────────────────────────────
    dpo_cfg = REPO / "configs/post_training/dpo_v5_8b.yaml"
    update_yaml_field(dpo_cfg, "base_checkpoint",
                      str(sft_best.relative_to(REPO)).replace("\\", "/"))
    rc = run(
        ["uv", "run", "python", "main.py",
         "--stage", "dpo", "--config", str(dpo_cfg)],
        "dpo_v5_8b", pipeline_log / "02_dpo_v5.log",
    )

    dpo_run = latest_run_dir("run_*_dpo")
    dpo_best = best_checkpoint(dpo_run) if dpo_run else None
    log(f"DPO v5 best: {dpo_best}")

    # ── Phase 3: patch eval pipelines to include the new checkpoints ─
    perp = REPO / "tools/perplexity_sweep.py"
    bench = REPO / "tools/benchmark_pipeline.py"
    brut = REPO / "tools/brutal_test.py"

    def _patch_pipeline(path: Path):
        text = path.read_text()
        new_entries = []
        if sft_best:
            new_entries.append(
                f'    ("sft_v5_8b",\n     r"{sft_best.relative_to(REPO).as_posix()}"),'
            )
        if dpo_best:
            new_entries.append(
                f'    ("dpo_v5_8b",\n     r"{dpo_best.relative_to(REPO).as_posix()}"),'
            )
        # Also add the 8b pretrain so we see the full chain
        eight_b = "runs/big_run/exp_c_8b/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt"
        new_entries.insert(0, f'    ("pretrain_8b",\n     r"{eight_b}"),')
        new_block = "PIPELINE = [\n" + "\n".join(new_entries) + "\n]"
        text = re.sub(r"PIPELINE = \[.*?\n\]", new_block, text, count=1, flags=re.DOTALL)
        path.write_text(text)
        log(f"  patched {path.name} pipeline")

    _patch_pipeline(perp)
    _patch_pipeline(bench)

    # brutal_test only takes a single CKPT; point at dpo_v5 if present, else sft_v5
    target = dpo_best or sft_best
    if target:
        ckpt_str = target.relative_to(REPO).as_posix()
        text = brut.read_text()
        text = re.sub(r'CKPT = r"[^"]+"', f'CKPT = r"{ckpt_str}"', text)
        brut.write_text(text)
        log(f"  patched brutal_test CKPT = {ckpt_str}")

    # ── Phase 4-6: evaluations ───────────────────────────────────────
    run(["uv", "run", "python", "-m", "tools.perplexity_sweep"],
        "perplexity_sweep", pipeline_log / "03_perplexity.log")
    run(["uv", "run", "python", "-m", "tools.benchmark_pipeline", "--n", "500"],
        "benchmark_pipeline", pipeline_log / "04_benchmarks.log")
    run(["uv", "run", "python", "-m", "tools.brutal_test"],
        "brutal_test", pipeline_log / "05_brutal_test.log")

    # ── Phase 7: write summary ───────────────────────────────────────
    out = REPO / "docs/post_training/overnight_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    perp_json = runs_dir / "perplexity_comparison.json"
    bench_json = runs_dir / "leaderboard_comparison.json"
    brut_log = pipeline_log / "05_brutal_test.log"

    summary_lines = [
        "# Overnight pipeline results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## What was run",
        "",
        "1. SFT v5 — Alpaca SFT on the 8B-token pretrain base"
        " (configs/post_training/sft_v5_8b.yaml)",
        f"   Best checkpoint: `{sft_best}`",
        "2. DPO v5 — length-normalized DPO on SFT v5 best"
        " (configs/post_training/dpo_v5_8b.yaml)",
        f"   Best checkpoint: `{dpo_best}`",
        "3. Full eval suite across all checkpoints (pretrain_500M,"
        " pretrain_8b, sft_v2, sft_v3, sft_v5_8b, dpo_v2, dpo_v3, dpo_v4, dpo_v5_8b)",
        "",
    ]
    if perp_json.exists():
        summary_lines += [
            "## Perplexity (pillar #1)",
            "",
            "```json",
            perp_json.read_text(),
            "```",
            "",
        ]
    if bench_json.exists():
        summary_lines += [
            "## Leaderboard MC accuracy (pillar #2)",
            "",
            "```json",
            bench_json.read_text(),
            "```",
            "",
        ]
    if brut_log.exists():
        summary_lines += [
            "## brutal_test (chat usability)",
            "",
            "```",
            brut_log.read_text()[-3000:],
            "```",
            "",
        ]

    out.write_text("\n".join(summary_lines))
    log(f"Wrote summary -> {out}")

    log("=" * 70)
    log("OVERNIGHT PIPELINE COMPLETE")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
