"""Tier-aware wrapper around the official leaderboard runner."""
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
