"""Tier-aware wrapper around the official PikoGPT_Leaderboard runner.

Smoke: 5 items per benchmark, ~30s, used by CI smoke tests.
Quick: 200 items at fixed seed, ~10-20 min, used as the iteration loop.
Full:  the official complete suite, ~8h, used only for final acceptance.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .registry import BenchmarkResult, save_result


TIER_LIMITS: dict[str, int | None] = {
    "smoke": 5,
    "quick": 200,
    "full": None,
}

NAMED_BENCHMARKS = ("hellaswag", "openbookqa", "winogrande", "lambada")


@dataclass(slots=True)
class BenchmarkRunSpec:
    checkpoint: Path
    tier: str  # "smoke" | "quick" | "full"
    submission_name: str
    leaderboard_repo: Path
    registry_dir: Path
    git_sha: str
    # Python interpreter the leaderboard uses to invoke our submission's main.py.
    # Must point at the ParrotLLM venv (which has dotenv/tiktoken/etc.) — the
    # leaderboard's own .venv lacks these. Defaults to whatever Python is
    # running the harness, which is the right answer 99% of the time.
    python_executable: str = field(default_factory=lambda: sys.executable)


def _invoke_leaderboard(cmd: list[str], cwd: Path) -> dict[str, float]:
    """Run the leaderboard CLI and parse its overview.json results.

    The leaderboard exits with code 1 whenever any benchmark scores at-or-below
    random chance — at low --limit tiers this is essentially guaranteed even for
    a working model. So we use the existence of overview.json (not the return
    code) as the success signal.

    Returns a dict mapping benchmark name -> accuracy_pct in [0, 100].
    """
    completed = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    # The leaderboard writes to Results/<submission>/<checkpoint_stem>/...;
    # locate the most recent overview.json regardless of the exact subdirectory.
    results_dir = cwd / "Results"
    candidates = sorted(results_dir.rglob("*overview.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(
            f"Leaderboard exited rc={completed.returncode} and produced no overview.json under {results_dir}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    overview = json.loads(candidates[-1].read_text())
    # Per-benchmark scores live under overview["benchmarks"] as a list of dicts;
    # top-level numeric fields are config metadata (seed, limit, timeout_s, ...).
    out: dict[str, float] = {}
    for entry in overview.get("benchmarks", []):
        name = str(entry.get("benchmark", "")).lower()
        acc = entry.get("accuracy_pct")
        if name and isinstance(acc, (int, float)):
            out[name] = float(acc)
    return out


def run_benchmark(spec: BenchmarkRunSpec) -> BenchmarkResult:
    if spec.tier not in TIER_LIMITS:
        raise ValueError(f"Unknown tier {spec.tier!r}; must be one of {list(TIER_LIMITS)}")

    cmd: list[str] = [
        "uv", "run", "python", "-m", "leaderboard.run_benchmarks",
        "--submission", spec.submission_name,
        "--checkpoint", str(spec.checkpoint),
        # Forward the parent venv's python so the leaderboard subprocess can
        # import ParrotLLM's deps (dotenv/tiktoken/datasets/...) when it spawns
        # our Submissions/ParrotLLM/main.py.
        "--python", spec.python_executable,
    ]
    limit = TIER_LIMITS[spec.tier]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    t0 = time.time()
    scores = _invoke_leaderboard(cmd, cwd=spec.leaderboard_repo)
    wall_clock = time.time() - t0

    # Keep only the four named benchmarks for PII; the leaderboard may emit more.
    named = {k: scores.get(k, 0.0) for k in NAMED_BENCHMARKS}
    pii_named = sum(named.values())

    result = BenchmarkResult(
        git_sha=spec.git_sha,
        checkpoint_basename=spec.checkpoint.name,
        tier=spec.tier,
        scores=named,
        pii_named=pii_named,
        wall_clock_seconds=wall_clock,
    )
    save_result(result, spec.registry_dir)
    return result
