"""Tests for randomized training-window sampling and strided eval windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.training.trainer import RandomWindowSampler, StridedWindowDataset


def _write_tokens(path: Path, count: int) -> None:
    np.arange(count, dtype=np.uint16).tofile(path)


def test_random_window_sampler_is_deterministic_and_epoch_sensitive():
    sampler = RandomWindowSampler(
        num_start_positions=17,
        samples_per_epoch=8,
        seed=123,
    )

    epoch0_first = list(iter(sampler))
    epoch0_second = list(iter(sampler))

    assert epoch0_first == epoch0_second
    assert len(epoch0_first) == 8
    assert len(set(epoch0_first)) == 8
    assert all(0 <= start < 17 for start in epoch0_first)

    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert epoch1 != epoch0_first
    assert len(set(epoch1)) == 8
    assert all(0 <= start < 17 for start in epoch1)


def test_random_window_sampler_shards_a_single_permutation_across_ranks():
    rank0 = RandomWindowSampler(
        num_start_positions=19,
        samples_per_epoch=10,
        seed=999,
        num_replicas=2,
        rank=0,
        drop_last=True,
    )
    rank1 = RandomWindowSampler(
        num_start_positions=19,
        samples_per_epoch=10,
        seed=999,
        num_replicas=2,
        rank=1,
        drop_last=True,
    )

    rank0.set_epoch(4)
    rank1.set_epoch(4)

    rank0_indices = list(iter(rank0))
    rank1_indices = list(iter(rank1))
    merged = rank0_indices + rank1_indices

    assert len(rank0_indices) == len(rank1_indices) == 5
    assert len(set(merged)) == 10
    assert set(rank0_indices).isdisjoint(rank1_indices)


def test_strided_window_dataset_uses_contiguous_eval_windows(tmp_path: Path):
    token_path = tmp_path / "tokens.bin"
    _write_tokens(token_path, count=9)

    dataset = StridedWindowDataset(str(token_path), context_length=4, stride=4)

    assert len(dataset) == 2

    x0, y0 = dataset[0]
    x1, y1 = dataset[1]

    torch.testing.assert_close(x0, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    torch.testing.assert_close(y0, torch.tensor([1, 2, 3, 4], dtype=torch.long))
    torch.testing.assert_close(x1, torch.tensor([4, 5, 6, 7], dtype=torch.long))
    torch.testing.assert_close(y1, torch.tensor([5, 6, 7, 8], dtype=torch.long))
