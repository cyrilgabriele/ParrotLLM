"""Overnight pipeline v6: SFT v6 (Alpaca + synthetic raw) -> DPO v6 ->
full eval suite (perplexity + leaderboard MC + brutal_test) +
PikoGPT_Leaderboard runner sweep.

Run with: PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
    uv run python -m tools.overnight_pipeline_v6 2>&1 | tee runs/overnight_v6.log

Phases:
  0. Build synthetic data (programmatic + public Q&A) into a single JSONL
  1. SFT v6 — Alpaca + synthetic raw on the 8B base
  2. DPO v6 — length-normalised DPO on SFT v6 best
  3. Patch perplexity_sweep / benchmark_pipeline / brutal_test pipelines
  4. perplexity_sweep
  5. benchmark_pipeline (n=500, internal log-likelihood scoring)
  6. brutal_test (DPO v6 best)
  7. PikoGPT_Leaderboard runner full validation sweep on dpo_v6 + sft_v6
  8. Write docs/post_training/overnight_v6_results.md

Mirrors tools/overnight_pipeline.py for v5; phase 0 and 7 are new.
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
LEADERBOARD = Path("C:/Users/chris/source/repos/PikoGPT_Leaderboard")
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    # tools/build_synthetic_mc_public.py imports `from src.post_training...`,
    # which requires the repo root on sys.path. Subprocesses don't pick up
    # CWD-as-path automatically when invoked through the venv interpreter
    # with an explicit script path, so set it here.
    "PYTHONPATH": str(REPO),
}
VENV_PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list[str], stage: str, log_path: Path, *, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"START {stage} -> {log_path}")
    log(f"  cmd: {' '.join(str(c) for c in cmd)}")
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(cmd, cwd=str(cwd or REPO), env=ENV,
                              stdout=fp, stderr=subprocess.STDOUT)
    log(f"END   {stage} (exit {proc.returncode})")
    return proc.returncode


def latest_run_dir(prefix_glob: str) -> Path | None:
    runs = sorted(glob.glob(str(REPO / "runs" / prefix_glob)))
    return Path(runs[-1]) if runs else None


def best_checkpoint(run_dir: Path) -> Path | None:
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
    pipeline_log = runs_dir / "overnight_v6"
    pipeline_log.mkdir(exist_ok=True)

    log("=" * 70)
    log("OVERNIGHT v6 PIPELINE START")
    log("=" * 70)

    # ── Phase 0: build synthetic data ─────────────────────────────────
    rc = run(
        [str(VENV_PYTHON), "tools/build_synthetic_mc_programmatic.py"],
        "build_synthetic_programmatic", pipeline_log / "00a_synth_prog.log",
    )
    if rc != 0:
        log("ABORT: programmatic synthetic build failed.")
        return rc
    rc = run(
        [str(VENV_PYTHON), "tools/build_synthetic_mc_public.py"],
        "build_synthetic_public", pipeline_log / "00b_synth_public.log",
    )
    if rc != 0:
        log("ABORT: public synthetic build failed.")
        return rc

    # Combine the two JSONL files into one for sft_v6_8b.yaml's
    # synthetic_jsonl_path.
    prog = REPO / "data/synthetic/sft_v6_programmatic.jsonl"
    pub = REPO / "data/synthetic/sft_v6_public.jsonl"
    combined = REPO / "data/synthetic/sft_v6_combined.jsonl"
    combined.write_text(
        prog.read_text(encoding="utf-8") + pub.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    n_lines = sum(1 for _ in combined.open(encoding="utf-8"))
    log(f"Combined synthetic JSONL: {n_lines} examples -> {combined}")

    # ── Phase 1: SFT v6 ──────────────────────────────────────────────
    sft_cfg = REPO / "configs/post_training/sft_v6_8b.yaml"
    rc = run(
        [str(VENV_PYTHON), "main.py", "--stage", "sft", "--config", str(sft_cfg)],
        "sft_v6_8b", pipeline_log / "01_sft_v6.log",
    )
    if rc != 0:
        log("ABORT: SFT v6 failed; pipeline halting before DPO.")
        return rc

    sft_run = latest_run_dir("run_*_sft")
    if sft_run is None:
        log("ABORT: could not locate SFT v6 run dir.")
        return 1
    sft_best = best_checkpoint(sft_run)
    if sft_best is None:
        log(f"ABORT: no best checkpoint in {sft_run}.")
        return 1
    log(f"SFT v6 best: {sft_best}")

    # ── Phase 2: DPO v6 ──────────────────────────────────────────────
    dpo_cfg = REPO / "configs/post_training/dpo_v6_8b.yaml"
    update_yaml_field(dpo_cfg, "base_checkpoint",
                      str(sft_best.relative_to(REPO)).replace("\\", "/"))
    rc = run(
        [str(VENV_PYTHON), "main.py", "--stage", "dpo", "--config", str(dpo_cfg)],
        "dpo_v6_8b", pipeline_log / "02_dpo_v6.log",
    )
    dpo_run = latest_run_dir("run_*_dpo")
    dpo_best = best_checkpoint(dpo_run) if dpo_run else None
    log(f"DPO v6 best: {dpo_best}")

    # ── Phase 3: patch eval pipelines (v6 entries) ───────────────────
    perp = REPO / "tools/perplexity_sweep.py"
    bench = REPO / "tools/benchmark_pipeline.py"
    brut = REPO / "tools/brutal_test.py"

    def _patch_pipeline(path: Path):
        text = path.read_text(encoding="utf-8")
        new_entries = []
        eight_b = ("runs/big_run/exp_c_8b/run_20260410_044337/"
                   "checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt")
        new_entries.append(f'    ("pretrain_8b",\n     r"{eight_b}"),')
        if sft_best:
            new_entries.append(
                f'    ("sft_v6_8b",\n     r"{sft_best.relative_to(REPO).as_posix()}"),'
            )
        if dpo_best:
            new_entries.append(
                f'    ("dpo_v6_8b",\n     r"{dpo_best.relative_to(REPO).as_posix()}"),'
            )
        new_block = "PIPELINE = [\n" + "\n".join(new_entries) + "\n]"
        text = re.sub(r"PIPELINE = \[.*?\n\]", new_block, text, count=1, flags=re.DOTALL)
        path.write_text(text, encoding="utf-8")
        log(f"  patched {path.name} pipeline")

    _patch_pipeline(perp)
    _patch_pipeline(bench)

    target = dpo_best or sft_best
    if target:
        ckpt_str = target.relative_to(REPO).as_posix()
        text = brut.read_text(encoding="utf-8")
        text = re.sub(r'CKPT = r"[^"]+"', f'CKPT = r"{ckpt_str}"', text)
        brut.write_text(text, encoding="utf-8")
        log(f"  patched brutal_test CKPT = {ckpt_str}")

    # ── Phase 4-6: internal evaluations ──────────────────────────────
    run([str(VENV_PYTHON), "-m", "tools.perplexity_sweep"],
        "perplexity_sweep", pipeline_log / "03_perplexity.log")
    run([str(VENV_PYTHON), "-m", "tools.benchmark_pipeline", "--n", "500"],
        "benchmark_pipeline", pipeline_log / "04_benchmarks.log")
    run([str(VENV_PYTHON), "-m", "tools.brutal_test"],
        "brutal_test", pipeline_log / "05_brutal_test.log")

    # ── Phase 7: PikoGPT_Leaderboard runner sweep ────────────────────
    if dpo_best:
        run(
            [str(VENV_PYTHON), "-m", "leaderboard.run_benchmarks",
             "--submission", "ParrotLLM",
             "--python", str(VENV_PYTHON),
             "--checkpoint", dpo_best.relative_to(REPO).as_posix()],
            "leaderboard_dpo_v6", pipeline_log / "06_leaderboard_dpo.log",
            cwd=LEADERBOARD,
        )
    if sft_best:
        run(
            [str(VENV_PYTHON), "-m", "leaderboard.run_benchmarks",
             "--submission", "ParrotLLM",
             "--python", str(VENV_PYTHON),
             "--checkpoint", sft_best.relative_to(REPO).as_posix()],
            "leaderboard_sft_v6", pipeline_log / "07_leaderboard_sft.log",
            cwd=LEADERBOARD,
        )

    # ── Phase 8: write summary ───────────────────────────────────────
    out = REPO / "docs/post_training/overnight_v6_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    perp_json = runs_dir / "perplexity_comparison.json"
    bench_json = runs_dir / "leaderboard_comparison.json"
    brut_log = pipeline_log / "05_brutal_test.log"

    summary_lines = [
        "# Overnight v6 results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## What ran",
        "",
        f"- SFT v6 best: `{sft_best}`",
        f"- DPO v6 best: `{dpo_best}`",
        f"- Synthetic JSONL: {combined} ({n_lines} rows)",
        "",
    ]
    if perp_json.exists():
        summary_lines += ["## Perplexity (Pillar #1)", "", "```json",
                          perp_json.read_text(), "```", ""]
    if bench_json.exists():
        summary_lines += ["## Leaderboard MC accuracy — internal LL scoring (Pillar #2 internal)",
                          "", "```json", bench_json.read_text(), "```", ""]
    if brut_log.exists():
        summary_lines += ["## brutal_test (chat usability)",
                          "", "```", brut_log.read_text()[-3000:], "```", ""]
    # Leaderboard runner outputs
    for log_file, label in (
        (pipeline_log / "06_leaderboard_dpo.log", "PikoGPT_Leaderboard runner — DPO v6"),
        (pipeline_log / "07_leaderboard_sft.log", "PikoGPT_Leaderboard runner — SFT v6"),
    ):
        if log_file.exists():
            summary_lines += [f"## {label}", "", "```",
                              log_file.read_text()[-4000:], "```", ""]
    out.write_text("\n".join(summary_lines))
    log(f"Wrote summary -> {out}")

    log("=" * 70)
    log("OVERNIGHT v6 PIPELINE COMPLETE")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
