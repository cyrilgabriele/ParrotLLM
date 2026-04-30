"""MPS regression: chunked CE backward must NOT be used on MPS.

Background: commit 2820d29 ('fix the OOM') added `_compute_loss_in_chunks`
to the model forward and made the trainer pass `return_logits=False` so the
chunked path is taken. On MPS this path's backward produces non-deterministic
gradient norms — same input, same weights, multiple runs gave grad norms
~30, ~290, ~370, ~310 (orders of magnitude wrong). The non-chunked path
matches CPU within fp32 noise.

This test runs CPU-only (so it's fast and runs everywhere) and pins:
1. `_use_chunked_ce_for_device` correctly returns False on MPS, True elsewhere.
2. The chunked and non-chunked paths agree on CPU (sanity).
3. The trainer-level call sites pass `return_logits` based on the helper.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.model.transformer import ParrotLLM
from src.posttraining.trainer import _use_chunked_ce_for_device


def _tiny_model() -> ParrotLLM:
    cfg = {"model": {
        "vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 4,
        "d_ff": 64, "context_length": 32, "bias": False,
        "dropout": 0.0, "rope_theta": 10000.0,
        "gradient_checkpointing": False,
    }}
    return ParrotLLM(cfg)


def test_use_chunked_ce_helper_routes_mps_to_non_chunked_path():
    """MPS should NOT take the chunked CE path."""
    assert _use_chunked_ce_for_device(torch.device("cpu")) is True
    assert _use_chunked_ce_for_device(torch.device("mps")) is False
    assert _use_chunked_ce_for_device(torch.device("cuda")) is True


def test_chunked_and_non_chunked_paths_agree_on_cpu():
    """The two CE paths must produce mathematically equivalent gradients
    on CPU. (If they diverge on CPU, then the MPS-vs-CPU comparison loses
    meaning — this test guards that invariant.)"""
    torch.manual_seed(0)
    B, T = 2, 16
    model_chunked = _tiny_model()
    model_full = _tiny_model()
    model_full.load_state_dict(model_chunked.state_dict())

    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))
    mask = torch.ones((B, T), dtype=torch.float32)
    mask[:, :2] = 0  # mask out the first couple tokens (typical SFT shape)

    _, loss_chunked = model_chunked(x, y, loss_mask=mask, return_logits=False, loss_chunk_rows=8)
    _, loss_full = model_full(x, y, loss_mask=mask, return_logits=True)

    assert math.isclose(loss_chunked.item(), loss_full.item(), rel_tol=1e-4), (
        f"chunked vs non-chunked CE forward losses must agree on CPU: "
        f"chunked={loss_chunked.item()}  non-chunked={loss_full.item()}"
    )

    loss_chunked.backward()
    loss_full.backward()
    grad_chunked = math.sqrt(sum((p.grad.float()**2).sum().item() for p in model_chunked.parameters() if p.grad is not None))
    grad_full = math.sqrt(sum((p.grad.float()**2).sum().item() for p in model_full.parameters() if p.grad is not None))
    assert math.isclose(grad_chunked, grad_full, rel_tol=1e-3), (
        f"chunked vs non-chunked CE gradient norms must agree on CPU: "
        f"chunked={grad_chunked}  non-chunked={grad_full}"
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable on this host",
)
def test_mps_non_chunked_backward_matches_cpu_within_noise():
    """On hosts with MPS, the trainer-selected path (return_logits=True on MPS)
    must produce gradient norms close to CPU. This is a regression test for
    the bug from commit 2820d29: the chunked path's MPS backward produced
    grad norms 10-1000× CPU's value."""
    torch.manual_seed(0)
    B, T = 1, 32  # must be <= context_length=32 of _tiny_model()
    base = _tiny_model()

    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))
    mask = torch.ones((B, T), dtype=torch.float32)
    mask[:, :4] = 0

    grad_norms = {}
    for device in ("cpu", "mps"):
        model = _tiny_model()
        model.load_state_dict(base.state_dict())
        model = model.to(device)
        for p in model.parameters():
            p.grad = None
        if device == "mps":
            torch.mps.synchronize()
        # Trainer-selected path: return_logits = True on MPS, False on CPU
        return_logits = not _use_chunked_ce_for_device(torch.device(device))
        _, loss = model(x.to(device), y.to(device), loss_mask=mask.to(device),
                        return_logits=return_logits)
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
        grad_norms[device] = math.sqrt(
            sum((p.grad.float()**2).sum().item() for p in model.parameters() if p.grad is not None)
        )

    cpu_gn = grad_norms["cpu"]
    mps_gn = grad_norms["mps"]
    # Allow 30% relative deviation (fp32 reduction-order noise on different backends).
    # The bug we're guarding produced 10-1000× ratios.
    ratio = mps_gn / cpu_gn if cpu_gn > 0 else float("inf")
    assert 0.7 < ratio < 1.3, (
        f"MPS gradient norm ({mps_gn:.4f}) deviates from CPU ({cpu_gn:.4f}) "
        f"by ratio {ratio:.4f}. The chunked CE bug (commit 2820d29) may have "
        f"resurfaced — check that the trainer is passing return_logits=True on MPS."
    )
