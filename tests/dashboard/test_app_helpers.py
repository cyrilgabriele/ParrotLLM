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


from src.dashboard.app import _fmt_status_banner


def _metrics_with_grad_explosion():
    m = TrainingMetrics()
    m.steps = [100, 200, 300]
    m.train_losses = [4.5, 4.1, 3.8]
    m.lrs = [3e-4] * 3
    m.grad_norms = [0.5, 15.0, 12.0]
    return m


def test_status_banner_green_when_no_alerts():
    m = TrainingMetrics()
    html = _fmt_status_banner(m)
    assert "No errors" in html or "No problems" in html or "✅" in html
    assert "green" in html or "#" in html


def test_status_banner_shows_error_code():
    m = _metrics_with_grad_explosion()
    html = _fmt_status_banner(m)
    assert "GRAD_EXPLOSION" in html


def test_status_banner_red_on_error():
    m = _metrics_with_grad_explosion()
    html = _fmt_status_banner(m)
    assert "🔴" in html or "red" in html.lower()


from src.dashboard.app import _fmt_progress_detail


def _metrics_with_history():
    m = TrainingMetrics()
    m.steps = list(range(100, 600, 100))      # [100,200,300,400,500]
    m.train_losses = [4.5, 4.3, 4.1, 3.9, 3.7]
    m.lrs = [3e-4] * 5
    m.grad_norms = [0.8, 0.7, 0.65, 0.6, 0.55]
    m.tokens_per_sec = [12000.0] * 5
    m.eval_steps = [200, 400]
    m.val_losses = [4.4, 3.95]
    m.best_step = 400
    m.config = {"max_steps": 1000, "batch_size": 64, "context_length": 1024,
                "gradient_accumulation_steps": 4}
    return m


def test_progress_detail_shows_step():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "500" in html


def test_progress_detail_shows_loss_delta():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "Δ" in html or "delta" in html.lower() or "-0." in html


def test_progress_detail_shows_progress_bar():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "50" in html or "50.0" in html


def test_progress_detail_shows_val_loss():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "3.95" in html or "3.9500" in html


def test_progress_detail_empty_state():
    m = TrainingMetrics()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "No runs found" in html
