# tests/dashboard/test_app_helpers.py
import pytest
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.app import _compute_eta


def _metrics_with_tps(steps, tps, max_steps=1000, batch=64, ctx=1024, grad_acc=1):
    m = TrainingMetrics()
    m.steps = steps
    m.tokens_per_sec = tps
    m.train_losses = [4.0] * len(steps)
    m.lrs = [1e-4] * len(steps)
    m.grad_norms = [0.5] * len(steps)
    m.config = {
        "max_steps": max_steps,
        "batch_size": batch,
        "context_length": ctx,
        "gradient_accumulation_steps": grad_acc,
    }
    return m


def test_eta_includes_seconds_hours():
    # 900 steps remaining × 1024 tokens/step ÷ 1024 tok/s = 900s = 0h 15m 0s
    m = _metrics_with_tps([100], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    assert "m" in eta and "s" in eta


def test_eta_large_remaining_shows_hours():
    # 7200s remaining → ~2h 0m 0s
    m = _metrics_with_tps([100], [1024.0], max_steps=7300, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    assert "h" in eta


def test_eta_seconds_only_when_under_a_minute():
    # 30s remaining
    m = _metrics_with_tps([970], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    assert "s" in eta


def test_eta_done_when_no_remaining():
    m = _metrics_with_tps([1000], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    assert _compute_eta(m) == "Done"


def test_eta_dash_when_no_tps():
    m = TrainingMetrics()
    m.steps = [100]
    m.train_losses = [4.0]
    m.lrs = [1e-4]
    m.grad_norms = [0.5]
    m.config = {"max_steps": 1000}
    assert _compute_eta(m) == "—"
