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


def test_resolve_runtime_batching_mps_disables_grad_accum_and_caps_batch():
    """MPS has a non-deterministic gradient-accumulation bug (p.grad +=
    across multiple .backward() calls produces grad norms 10²-10⁵× wrong
    intermittently). Workaround: force gradient_accumulation_steps=1 on
    MPS and cap batch at 8 to keep transient logits memory under ~2 GB."""
    train_bs, eval_bs, grad_accum = _resolve_runtime_batching(
        device=torch.device("mps"),
        train_batch_size=8,
        eval_batch_size=8,
        gradient_accumulation_steps=8,
    )
    assert train_bs == 8
    assert eval_bs == 8
    assert grad_accum == 1


def test_resolve_runtime_batching_mps_caps_oversize_batch():
    train_bs, eval_bs, grad_accum = _resolve_runtime_batching(
        device=torch.device("mps"),
        train_batch_size=64,
        eval_batch_size=64,
        gradient_accumulation_steps=4,
    )
    assert train_bs == 8
    assert eval_bs == 8
    assert grad_accum == 1


def test_loss_chunk_rows_for_mps_are_small():
    assert _loss_chunk_rows_for_device(torch.device("mps")) == 16
