"""Generate matplotlib figures for the dashboard (light theme)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from src.dashboard.metrics_reader import TrainingMetrics

TRAIN_COLOR = "#2563EB"   # blue
VAL_COLOR   = "#EA580C"   # orange
LR_COLOR    = "#16A34A"   # green
GRAD_COLOR  = "#D97706"   # amber


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def build_training_figure(metrics: TrainingMetrics) -> Optional[plt.Figure]:
    """Return a 2×2 Figure from TrainingMetrics, or None if no data."""
    if not metrics.steps:
        return None

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    steps = metrics.steps
    eval_steps = metrics.eval_steps

    # ── [0,0] Train & Val Loss ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, metrics.train_losses, color=TRAIN_COLOR, linewidth=1.5, label="Train")
    if eval_steps and metrics.val_losses:
        ax1.plot(eval_steps, metrics.val_losses, color=VAL_COLOR, linewidth=1.5,
                 linestyle="--", label="Val")
    if metrics.best_step:
        ax1.axvline(metrics.best_step, color="#999", linewidth=0.8, linestyle=":",
                    label=f"Best @ {metrics.best_step}")
    _style(ax1)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss")
    ax1.legend(fontsize=7, frameon=False)

    # ── [0,1] Validation Perplexity ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if eval_steps and metrics.val_ppls:
        ax2.plot(eval_steps, metrics.val_ppls, color=VAL_COLOR, linewidth=1.5)
        ax2.set_yscale("log")
    _style(ax2)
    ax2.set_ylabel("Perplexity (log)")
    ax2.set_title("Validation Perplexity")

    # ── [1,0] Learning Rate + Grad Norm (twin axis) ───────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, metrics.lrs, color=LR_COLOR, linewidth=1.5, label="LR")
    ax3.set_ylabel("Learning Rate", color=LR_COLOR)
    ax3.tick_params(axis="y", labelcolor=LR_COLOR)
    ax3.set_xlabel("Step")
    ax3.set_title("LR & Grad Norm")
    _style(ax3)
    if metrics.grad_norms:
        ax3b = ax3.twinx()
        ax3b.plot(steps, metrics.grad_norms, color=GRAD_COLOR, linewidth=1.0,
                  alpha=0.7, label="Grad Norm")
        ax3b.set_ylabel("Grad Norm", color=GRAD_COLOR)
        ax3b.tick_params(axis="y", labelcolor=GRAD_COLOR)
        ax3b.spines["top"].set_visible(False)

    # ── [1,1] Train–Val Gap ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if eval_steps and metrics.val_losses and metrics.eval_train_losses:
        gap = [v - t for v, t in zip(metrics.val_losses, metrics.eval_train_losses)]
        ax4.plot(eval_steps, gap, color=VAL_COLOR, linewidth=1.5)
        ax4.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
    _style(ax4)
    ax4.set_ylabel("Val − Train Loss")
    ax4.set_xlabel("Step")
    ax4.set_title("Generalization Gap")

    fig.tight_layout(pad=1.5)
    return fig
