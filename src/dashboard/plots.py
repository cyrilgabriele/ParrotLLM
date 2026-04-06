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
    """Return a 2×3 Figure from TrainingMetrics, or None if no data.

    Top row (most active): Train & Val Loss | Tokens/sec | LR & Grad Norm
    Bottom row (contextual): Val Perplexity | Generalization Gap | (hidden)
    """
    if not metrics.steps:
        return None

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    steps = metrics.steps
    eval_steps = metrics.eval_steps
    n_eval = len(eval_steps)

    # ── [0,0] Train & Val Loss (most important) ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, metrics.train_losses, color=TRAIN_COLOR, linewidth=1.8, label="Train")
    if eval_steps and metrics.val_losses:
        ax1.plot(eval_steps, metrics.val_losses[:n_eval], color=VAL_COLOR, linewidth=1.8,
                 linestyle="--", label="Val")
    if metrics.best_step:
        ax1.axvline(metrics.best_step, color="#999", linewidth=0.8, linestyle=":",
                    label=f"Best @ {metrics.best_step}")
    _style(ax1)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss", fontweight="bold")
    ax1.legend(fontsize=8, frameon=False)

    # ── [0,1] Tokens per second ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.tokens_per_sec:
        ax2.plot(steps[:len(metrics.tokens_per_sec)], metrics.tokens_per_sec,
                 color=TRAIN_COLOR, linewidth=1.8)
    _style(ax2)
    ax2.set_ylabel("Tok/s")
    ax2.set_title("Tokens per Second", fontweight="bold")

    # ── [0,2] LR + Grad Norm (twin axis) ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, metrics.lrs, color=LR_COLOR, linewidth=1.8, label="LR")
    ax3.set_ylabel("Learning Rate", color=LR_COLOR)
    ax3.tick_params(axis="y", labelcolor=LR_COLOR)
    ax3.set_xlabel("Step")
    ax3.set_title("LR & Grad Norm", fontweight="bold")
    _style(ax3)
    if metrics.grad_norms:
        ax3b = ax3.twinx()
        ax3b.plot(steps, metrics.grad_norms, color=GRAD_COLOR, linewidth=1.2,
                  alpha=0.7, label="Grad Norm")
        ax3b.set_ylabel("Grad Norm", color=GRAD_COLOR)
        ax3b.tick_params(axis="y", labelcolor=GRAD_COLOR)
        ax3b.spines["top"].set_visible(False)

    # ── [1,0] Validation Perplexity ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if eval_steps and metrics.val_ppls:
        ax4.plot(eval_steps, metrics.val_ppls[:n_eval], color=VAL_COLOR, linewidth=1.8)
        ax4.set_yscale("log")
    _style(ax4)
    ax4.set_ylabel("Perplexity (log)")
    ax4.set_xlabel("Step")
    ax4.set_title("Validation Perplexity")

    # ── [1,1] Generalization Gap ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if eval_steps and metrics.val_losses and metrics.eval_train_losses:
        gap = [v - t for v, t in zip(metrics.val_losses[:n_eval],
                                      metrics.eval_train_losses[:n_eval])]
        ax5.plot(eval_steps[:len(gap)], gap, color=VAL_COLOR, linewidth=1.8)
        ax5.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
    _style(ax5)
    ax5.set_ylabel("Val − Train Loss")
    ax5.set_xlabel("Step")
    ax5.set_title("Generalization Gap")

    # ── [1,2] hidden ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_visible(False)

    return fig
