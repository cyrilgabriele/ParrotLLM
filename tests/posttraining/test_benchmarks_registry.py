"""The registry persists benchmark results to disk and reads them back.

File-naming convention: <git_sha>__<checkpoint_basename>__<tier>.json
"""
import json
from pathlib import Path

import pytest

from src.posttraining.benchmarks.registry import (
    BenchmarkResult,
    save_result,
    load_results,
)


def test_save_result_writes_expected_filename(tmp_path: Path) -> None:
    result = BenchmarkResult(
        git_sha="abc1234",
        checkpoint_basename="best_loss_3p2650_step_0096000.pt",
        tier="quick",
        scores={
            "hellaswag": 33.5,
            "openbookqa": 12.0,
            "winogrande": 67.2,
            "lambada": 1.5,
        },
        pii_named=114.2,
        wall_clock_seconds=812.3,
    )
    path = save_result(result, registry_dir=tmp_path)
    assert path.name == "abc1234__best_loss_3p2650_step_0096000.pt__quick.json"
    payload = json.loads(path.read_text())
    assert payload["git_sha"] == "abc1234"
    assert payload["scores"]["hellaswag"] == 33.5
    assert payload["pii_named"] == pytest.approx(114.2)


def test_load_results_returns_all_json_in_directory(tmp_path: Path) -> None:
    for sha in ("aaa", "bbb"):
        save_result(
            BenchmarkResult(
                git_sha=sha,
                checkpoint_basename="ckpt.pt",
                tier="smoke",
                scores={"hellaswag": 30.0, "openbookqa": 10.0,
                        "winogrande": 50.0, "lambada": 0.0},
                pii_named=90.0,
                wall_clock_seconds=10.0,
            ),
            registry_dir=tmp_path,
        )
    results = load_results(tmp_path)
    assert len(results) == 2
    assert {r.git_sha for r in results} == {"aaa", "bbb"}


def test_load_results_handles_empty_directory(tmp_path: Path) -> None:
    assert load_results(tmp_path) == []
