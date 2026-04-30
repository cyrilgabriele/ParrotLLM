"""WeightedRandomSampler must oversample by per-example quality_score."""
import json
from pathlib import Path

import torch

from src.posttraining.trainer import (
    PackedConversationDataset,
    _build_quality_sampler,
)


def test_weighted_sampler_proportional_to_quality(tmp_path: Path):
    records = [
        {"tokens": [1, 2, 3], "loss_mask": [0, 0, 1], "quality_score": 1.0},
        {"tokens": [4, 5, 6], "loss_mask": [0, 0, 1], "quality_score": 1.0},
        {"tokens": [7, 8, 9], "loss_mask": [0, 0, 1], "quality_score": 2.0},
    ]
    path = tmp_path / "packed.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    ds = PackedConversationDataset(path)
    sampler = _build_quality_sampler(ds, seed=0, num_samples=20_000)
    counts = [0, 0, 0]
    for idx in sampler:
        counts[idx] += 1
    # Index 2 has quality 2.0, twice the weight of indices 0 and 1 (each 1.0).
    # Expected ratios: index 0 = index 1 = 0.25, index 2 = 0.50.
    # n=20_000, p=0.5 gives std ≈ 0.0035; tolerance 0.02 is ~5σ headroom.
    expected_ratio_2 = 2.0 / 4.0
    observed_ratio_2 = counts[2] / 20_000
    assert abs(observed_ratio_2 - expected_ratio_2) < 0.02
    # Indices 0 and 1 share equal weight — their counts should be within
    # statistical noise of each other.
    assert abs(counts[0] - counts[1]) / 20_000 < 0.02


def test_weighted_sampler_handles_missing_quality_score(tmp_path: Path):
    """Records without quality_score default to weight=1.0 (uniform)."""
    records = [
        {"tokens": [1, 2], "loss_mask": [0, 1]},
        {"tokens": [3, 4], "loss_mask": [0, 1]},
    ]
    path = tmp_path / "packed.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    ds = PackedConversationDataset(path)
    sampler = _build_quality_sampler(ds, seed=0, num_samples=10_000)
    counts = [0, 0]
    for idx in sampler:
        counts[idx] += 1
    # ~uniform within ±5%
    assert abs(counts[0] / 10_000 - 0.5) < 0.05
