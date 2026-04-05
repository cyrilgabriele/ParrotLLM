# src/dashboard/problem_detector.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.dashboard.metrics_reader import TrainingMetrics


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Alert:
    severity: Severity
    code: str
    message: str
    detail: str


def detect_problems(metrics: TrainingMetrics) -> list[Alert]:
    """Analyze training metrics and return a list of Alerts."""
    alerts: list[Alert] = []

    # GRAD_EXPLOSION: grad_norm > 10.0 in any of the last 3 steps
    if metrics.grad_norms:
        recent = metrics.grad_norms[-3:]
        if any(g > 10.0 for g in recent):
            worst = max(recent)
            alerts.append(Alert(
                severity=Severity.ERROR, code="GRAD_EXPLOSION",
                message="Gradient explosion detected",
                detail=f"Grad norm {worst:.2f} in the last 3 steps (threshold: 10.0). "
                       "Consider reducing lr or increasing grad_clip.",
            ))

    # OVERFITTING: val-train gap > 0.5 AND widening over last 3 evals
    if len(metrics.val_losses) >= 3 and len(metrics.eval_train_losses) >= 3:
        gaps = [v - t for v, t in zip(metrics.val_losses[-3:], metrics.eval_train_losses[-3:])]
        if gaps[-1] > 0.5 and gaps[-1] > gaps[0]:
            alerts.append(Alert(
                severity=Severity.WARNING, code="OVERFITTING",
                message="Overfitting detected",
                detail=f"Val–train gap is {gaps[-1]:.3f} and widening "
                       f"(was {gaps[0]:.3f} three evals ago). "
                       "Consider adding dropout or early stopping.",
            ))

    # STAGNATION: val_loss not improved by > 0.001 in last 5 evals
    if len(metrics.val_losses) >= 5:
        window = metrics.val_losses[-5:]
        if max(window) - min(window) < 0.001:
            alerts.append(Alert(
                severity=Severity.WARNING, code="STAGNATION",
                message="Training stagnation detected",
                detail=f"Val loss range over last 5 evals is only "
                       f"{max(window) - min(window):.5f}. Training may have plateaued.",
            ))

    # HIGH_LOSS: train_loss > 7.0 after step 200
    if metrics.steps and metrics.steps[-1] >= 200:
        recent = [l for s, l in zip(metrics.steps, metrics.train_losses) if s >= 200]
        if recent and all(l > 7.0 for l in recent[-3:]):
            alerts.append(Alert(
                severity=Severity.ERROR, code="HIGH_LOSS",
                message="Abnormally high training loss",
                detail=f"Train loss is {recent[-1]:.2f} after step 200. "
                       "Model may not be learning — check data or lr.",
            ))

    # LR_ZERO: learning rate is 0 after step 20
    if metrics.steps and metrics.steps[-1] > 20:
        if all(lr == 0.0 for lr in metrics.lrs[-3:]):
            alerts.append(Alert(
                severity=Severity.ERROR, code="LR_ZERO",
                message="Learning rate is zero",
                detail="LR has been 0 for the last 3 logged steps after step 20. "
                       "Check scheduler configuration.",
            ))

    return alerts
