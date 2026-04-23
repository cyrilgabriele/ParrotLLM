from __future__ import annotations

import torch

from src.posttraining.trainer import _loss_chunk_rows_for_device, _resolve_runtime_batching


def test_resolve_runtime_batching_keeps_non_mps_unchanged():
    train_bs, eval_bs, grad_accum = _resolve_runtime_batching(
        device=torch.device("cpu"),
        train_batch_size=8,
        eval_batch_size=8,
        gradient_accumulation_steps=8,
    )

    assert train_bs == 8
    assert eval_bs == 8
    assert grad_accum == 8


def test_resolve_runtime_batching_caps_mps_and_preserves_effective_batch():
    train_bs, eval_bs, grad_accum = _resolve_runtime_batching(
        device=torch.device("mps"),
        train_batch_size=8,
        eval_batch_size=8,
        gradient_accumulation_steps=8,
    )

    assert train_bs == 1
    assert eval_bs == 2
    assert grad_accum == 64


def test_loss_chunk_rows_for_mps_are_small():
    assert _loss_chunk_rows_for_device(torch.device("mps")) == 16
