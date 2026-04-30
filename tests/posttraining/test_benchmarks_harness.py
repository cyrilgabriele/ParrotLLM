"""Tier-aware wrapper around the official leaderboard runner."""
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from src.posttraining.benchmarks.harness import (
    TIER_LIMITS,
    BenchmarkRunSpec,
    run_benchmark,
)


def test_tier_limits_pinned_to_spec_values() -> None:
    # Smoke = 5, quick = 200, full = None (no limit).
    assert TIER_LIMITS == {"smoke": 5, "quick": 200, "full": None}


def test_run_benchmark_invokes_leaderboard_with_correct_limit_and_python(tmp_path: Path) -> None:
    spec = BenchmarkRunSpec(
        checkpoint=Path("runs/some/ckpt.pt"),
        tier="quick",
        submission_name="ParrotLLM",
        leaderboard_repo=tmp_path / "PikoGPT_Leaderboard",
        registry_dir=tmp_path / "registry",
        git_sha="abc1234",
    )
    fake_results = {
        "hellaswag": 35.0,
        "openbookqa": 18.0,
        "winogrande": 68.0,
        "lambada": 5.0,
    }
    with mock.patch(
        "src.posttraining.benchmarks.harness._invoke_leaderboard",
        return_value=fake_results,
    ) as invoke:
        result = run_benchmark(spec)
    invoke.assert_called_once()
    args, _ = invoke.call_args
    cmd = args[0]
    # The leaderboard must be invoked with limit=200 for the quick tier.
    assert "--limit" in cmd
    limit_index = cmd.index("--limit")
    assert cmd[limit_index + 1] == "200"
    assert "--submission" in cmd
    submission_index = cmd.index("--submission")
    assert cmd[submission_index + 1] == "ParrotLLM"
    # Critical: the leaderboard subprocess must use our parent venv's python so
    # imports of ParrotLLM-only deps (dotenv, tiktoken, ...) succeed.
    assert "--python" in cmd
    python_index = cmd.index("--python")
    assert cmd[python_index + 1] == sys.executable
    assert result.tier == "quick"
    assert result.scores == fake_results
    assert result.pii_named == pytest.approx(126.0)


def test_run_benchmark_full_tier_omits_limit_flag(tmp_path: Path) -> None:
    spec = BenchmarkRunSpec(
        checkpoint=Path("runs/some/ckpt.pt"),
        tier="full",
        submission_name="ParrotLLM",
        leaderboard_repo=tmp_path / "PikoGPT_Leaderboard",
        registry_dir=tmp_path / "registry",
        git_sha="abc1234",
    )
    with mock.patch(
        "src.posttraining.benchmarks.harness._invoke_leaderboard",
        return_value={"hellaswag": 0, "openbookqa": 0, "winogrande": 0, "lambada": 0},
    ) as invoke:
        run_benchmark(spec)
    cmd = invoke.call_args.args[0]
    assert "--limit" not in cmd  # full tier = no limit
    # --python should still be present at every tier.
    assert "--python" in cmd


def test_run_benchmark_parses_real_overview_schema(tmp_path: Path) -> None:
    """Regression: the leaderboard writes scores under overview['benchmarks'][i]['accuracy_pct'],
    not as top-level numeric keys. Top-level numerics are config metadata (seed, limit, ...).
    """
    from src.posttraining.benchmarks import harness as harness_mod

    leaderboard_root = tmp_path / "PikoGPT_Leaderboard"
    results_dir = leaderboard_root / "Results" / "ParrotLLM" / "ckpt"
    results_dir.mkdir(parents=True)
    overview = results_dir / "ParrotLLM__ckpt__overview.json"
    overview.write_text(json.dumps({
        "submission": "ParrotLLM",
        "checkpoint": "/abs/path/to/ckpt.pt",
        "limit": 5,
        "seed": 0,
        "temperature": 0.0,
        "timeout_s": 60,
        "mc_max_tokens": 3,
        "lambada_max_tokens": 5,
        "benchmarks": [
            {"benchmark": "hellaswag",  "total": 5, "correct": 1, "invalid": 0, "accuracy_pct": 20.0},
            {"benchmark": "winogrande", "total": 5, "correct": 3, "invalid": 0, "accuracy_pct": 60.0},
            {"benchmark": "openbookqa", "total": 5, "correct": 0, "invalid": 0, "accuracy_pct":  0.0},
            {"benchmark": "lambada",    "total": 5, "correct": 0, "invalid": 0, "accuracy_pct":  0.0},
        ],
    }))

    spec = harness_mod.BenchmarkRunSpec(
        checkpoint=Path("/abs/path/to/ckpt.pt"),
        tier="smoke",
        submission_name="ParrotLLM",
        leaderboard_repo=leaderboard_root,
        registry_dir=tmp_path / "registry",
        git_sha="deadbee",
    )

    # Simulate: leaderboard runs and exits with code 1 (because at smoke tier scores
    # are at-or-below random chance — the runner's "ok" gate fails). Our wrapper must
    # tolerate this and use the overview.json's existence as the success signal.
    fake_completed = subprocess.CompletedProcess(args=["uv"], returncode=1, stdout="", stderr="")
    with mock.patch("src.posttraining.benchmarks.harness.subprocess.run", return_value=fake_completed):
        result = harness_mod.run_benchmark(spec)

    assert result.scores == {
        "hellaswag": 20.0, "winogrande": 60.0, "openbookqa": 0.0, "lambada": 0.0
    }
    assert result.pii_named == 80.0
    # Crucially: returncode=1 must NOT raise; overview.json existing is the success signal.
