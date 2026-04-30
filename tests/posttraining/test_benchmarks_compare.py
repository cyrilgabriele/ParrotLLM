"""Comparison table includes our runs, external groups, and variance budget."""
from pathlib import Path

import yaml

from src.posttraining.benchmarks.registry import BenchmarkResult, save_result
from src.posttraining.benchmarks.compare import (
    build_comparison_markdown,
    quick_tier_variance_pp,
)


def test_quick_tier_variance_at_n_200_around_three_pp() -> None:
    # Standard error sqrt(p*(1-p)/N) at p=0.33, N=200 ~= 0.0333 -> 3.3pp.
    var = quick_tier_variance_pp(p=0.33, n=200)
    assert 3.0 < var < 3.6


def test_build_markdown_includes_our_runs_external_groups_and_variance(tmp_path: Path) -> None:
    save_result(
        BenchmarkResult(
            git_sha="abc1234",
            checkpoint_basename="sft_best.pt",
            tier="quick",
            scores={"hellaswag": 35.0, "openbookqa": 18.0,
                    "winogrande": 68.0, "lambada": 5.0},
            pii_named=126.0,
            wall_clock_seconds=900.0,
        ),
        registry_dir=tmp_path / "registry",
    )
    external = tmp_path / "external_groups.yaml"
    external.write_text(
        yaml.safe_dump(
            [{
                "name": "PikoGPT_GH",
                "hellaswag": 33.33,
                "openbookqa": 0.0,
                "winogrande": 66.67,
                "lambada": 0.0,
                "pii_named": 100.0,
                "source": "VL08 slide 29",
            }]
        )
    )
    md = build_comparison_markdown(
        registry_dir=tmp_path / "registry",
        external_groups_path=external,
    )
    assert "sft_best.pt" in md
    assert "PikoGPT_GH" in md
    assert "126.0" in md
    assert "variance" in md.lower()
    assert "hellaswag" in md.lower()
