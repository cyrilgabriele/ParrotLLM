"""Pins the canonical SFT recipe values so misconfigurations fail loudly.

Reference points:
  - SmolLM2-135M SFT (HF, 2024): LR 5e-5..1e-4, 1-3 epochs
  - Ibrahim et al. 2024: 5-10% replay during continued pretraining
  - Alpaca: 3 epochs, warmup_ratio 0.03
"""
from pathlib import Path

import yaml


SFT_YAML = Path(__file__).resolve().parents[2] / "configs/posttraining/sft.yaml"


def _load() -> dict:
    return yaml.safe_load(SFT_YAML.read_text())


def test_learning_rates_in_canonical_range_for_small_model():
    sft = _load()["sft"]
    lrs = sft["learning_rates"]
    assert lrs, "expected at least one LR"
    for lr in lrs:
        assert 2e-5 <= lr <= 3e-4, (
            f"LR {lr} outside SmolLM2-class small-model SFT range. "
            f"Canonical: 5e-5..1e-4. See plan 2026-04-30-sft-fixes."
        )


def test_replay_ratio_within_continual_pretraining_range():
    sft = _load()["sft"]
    ratio = sft["replay_ratio"]
    assert 0.0 <= ratio <= 0.15, (
        f"replay_ratio={ratio} too high. Ibrahim et al. 2024 finds 5-10% "
        f"sufficient; >15% dilutes SFT signal too much for tight step budgets."
    )


def test_polish_pass_enabled():
    sft = _load()["sft"]
    assert sft["polish_epochs"] > 0.0, (
        "polish_epochs=0 disables the second-stage tight-quality SFT pass. "
        "Re-enable to match the original recipe design."
    )


def test_format_score_weight_strong_enough_to_steer_selection():
    sft = _load()["sft"]
    weight = sft["format_score_weight"]
    assert weight >= 0.25, (
        f"format_score_weight={weight} too low; checkpoint selection "
        f"is dominated by dev_loss alone. Canonical 0.3-0.5 puts format "
        f"in the same OOM as a 1-nat dev_loss change."
    )


def test_prompt_suite_uses_richer_default():
    sft = _load()["sft"]
    suite = Path(sft["prompt_suite_path"]).name
    assert "ifeval" in suite, (
        f"prompt_suite_path={suite} still points at the 8-prompt strict "
        f"exact-match suite. Use ifeval-lite for finer granularity."
    )
