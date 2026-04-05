# tests/dashboard/test_problem_detector.py
import pytest
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.problem_detector import detect_problems, Severity


def _make(steps, losses, grad_norms=None, val_losses=None, eval_train_losses=None, lrs=None):
    m = TrainingMetrics()
    m.steps = steps
    m.train_losses = losses
    m.grad_norms = grad_norms or [0.5] * len(steps)
    m.lrs = lrs or [3e-4] * len(steps)
    if val_losses:
        m.eval_steps = steps[-len(val_losses):]
        m.val_losses = val_losses
    if eval_train_losses:
        m.eval_train_losses = eval_train_losses
    return m


def test_no_alerts_clean_run():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8])
    assert detect_problems(m) == []


def test_grad_explosion_detected():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8], grad_norms=[0.5, 15.0, 12.0])
    alerts = detect_problems(m)
    codes = [a.code for a in alerts]
    assert "GRAD_EXPLOSION" in codes
    assert any(a.severity == Severity.ERROR for a in alerts if a.code == "GRAD_EXPLOSION")


def test_no_explosion_below_threshold():
    m = _make([100, 200], [4.5, 4.1], grad_norms=[9.9, 9.8])
    assert all(a.code != "GRAD_EXPLOSION" for a in detect_problems(m))


def test_overfitting_detected():
    m = _make(
        list(range(100, 600, 100)),
        [4.5, 4.3, 4.1, 3.9, 3.8],
        val_losses=[4.6, 4.7, 4.9, 5.2, 5.6],
        eval_train_losses=[4.5, 4.3, 4.1, 3.9, 3.8],
    )
    codes = [a.code for a in detect_problems(m)]
    assert "OVERFITTING" in codes


def test_stagnation_detected():
    m = _make(
        list(range(100, 600, 100)),
        [4.5, 4.3, 4.1, 3.9, 3.8],
        val_losses=[3.5001, 3.5002, 3.5000, 3.5001, 3.5000],
        eval_train_losses=[3.4, 3.4, 3.4, 3.4, 3.4],
    )
    codes = [a.code for a in detect_problems(m)]
    assert "STAGNATION" in codes


def test_high_loss_detected():
    m = _make(list(range(200, 500, 100)), [8.0, 8.1, 7.9])
    codes = [a.code for a in detect_problems(m)]
    assert "HIGH_LOSS" in codes


def test_lr_zero_detected():
    m = _make([100, 200, 300], [4.5, 4.1, 3.8], lrs=[0.0, 0.0, 0.0])
    codes = [a.code for a in detect_problems(m)]
    assert "LR_ZERO" in codes


def test_lr_zero_not_flagged_early():
    m = _make([5, 10], [9.0, 8.5], lrs=[0.0, 0.0])
    assert all(a.code != "LR_ZERO" for a in detect_problems(m))
