"""Tests for rank-local evaluation behavior."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from src.training.trainer import estimate_loss


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([idx, idx + 1], dtype=torch.long)
        y = torch.tensor([idx + 1, idx + 2], dtype=torch.long)
        return x, y


class _InnerEvalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
        **_: object,
    ) -> tuple[None, torch.Tensor]:
        assert targets is not None
        assert return_logits is False
        self.forward_calls += 1
        return None, torch.tensor(1.5, dtype=torch.float32, device=idx.device)


class _WrapperThatMustNotRun(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):  # pragma: no cover - exercised only on regression
        raise AssertionError("estimate_loss should bypass the distributed wrapper")


def test_estimate_loss_uses_unwrapped_module_for_rank_local_eval():
    inner = _InnerEvalModel()
    wrapped = _WrapperThatMustNotRun(inner)

    metrics = estimate_loss(
        wrapped,
        _TinyDataset(),
        device=torch.device("cpu"),
        autocast_ctx=nullcontext(),
        batch_size=1,
        max_batches=2,
    )

    assert inner.forward_calls == 2
    assert metrics["loss"] == 1.5
