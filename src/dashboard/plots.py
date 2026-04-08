"""Generate matplotlib figures for the dashboard."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

from src.dashboard.metrics_reader import TrainingMetrics, read_metrics

TRAIN_COLOR = "#2563EB"   # blue
VAL_COLOR   = "#EA580C"   # orange
LR_COLOR    = "#16A34A"   # green
GRAD_COLOR  = "#D97706"   # amber
TOKSEC_COLOR = "#7C3AED"  # purple
LRxGN_COLOR  = "#92400E"  # brown


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

    # ── [1,2] Combined Metrics: Tokens/sec + LR × Grad Norm ────────
    ax6 = fig.add_subplot(gs[1, 2])
    if metrics.tokens_per_sec and metrics.grad_norms:
        n = min(len(steps), len(metrics.tokens_per_sec))
        ax6.plot(steps[:n], metrics.tokens_per_sec[:n],
                 color=TOKSEC_COLOR, linewidth=1.2, alpha=0.8, label="Tokens/sec")
        ax6.set_ylabel("Tokens/sec", color=TOKSEC_COLOR)
        ax6.tick_params(axis="y", labelcolor=TOKSEC_COLOR)
        ax6.set_xlabel("Step")
        ax6.set_title("Combined Metrics", fontweight="bold")
        _style(ax6)

        # LR × Grad Norm on right axis
        n_lg = min(len(steps), len(metrics.lrs), len(metrics.grad_norms))
        lr_x_gn = [lr * gn for lr, gn in
                    zip(metrics.lrs[:n_lg], metrics.grad_norms[:n_lg])]
        ax6b = ax6.twinx()
        ax6b.plot(steps[:n_lg], lr_x_gn,
                  color=LRxGN_COLOR, linewidth=1.2, alpha=0.7, label="LR × Grad Norm")
        ax6b.set_ylabel("LR × Grad Norm", color=LRxGN_COLOR)
        ax6b.tick_params(axis="y", labelcolor=LRxGN_COLOR)
        ax6b.spines["top"].set_visible(False)

        # Combined legend
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6b.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False)
    else:
        ax6.set_visible(False)

    return fig


# ── Compare overlay plot ──────────────────────────────────────────────────────

_COMPARE_COLORS = [
    "#2563EB", "#EA580C", "#16A34A", "#D97706", "#7C3AED",
    "#DC2626", "#0891B2", "#4F46E5", "#059669", "#E11D48",
]


def build_compare_figure(
    run_dirs: list[Path],
    dark: bool = False,
) -> Optional[plt.Figure]:
    """Overlay train+val loss curves from multiple runs on a single figure."""
    if not run_dirs:
        return None

    bg = "#1e1e2e" if dark else "white"
    fg = "#cdd6f4" if dark else "#111"
    grid_color = "#45475a" if dark else "#e0e0e0"

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor(bg)
    fig.suptitle("Run Comparison", fontweight="bold", color=fg, fontsize=14)

    for ax in (ax_train, ax_val):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.grid(axis="y", color=grid_color, linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)

    has_train = False
    has_val = False

    for i, run_dir in enumerate(run_dirs):
        m = read_metrics(run_dir)
        color = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
        label = run_dir.name

        if m.steps and m.train_losses:
            ax_train.plot(m.steps, m.train_losses, color=color, linewidth=1.5,
                          label=label, alpha=0.85)
            has_train = True

        if m.eval_steps and m.val_losses:
            n = min(len(m.eval_steps), len(m.val_losses))
            ax_val.plot(m.eval_steps[:n], m.val_losses[:n], color=color,
                        linewidth=1.5, label=label, alpha=0.85)
            has_val = True

    ax_train.set_title("Train Loss", fontweight="bold")
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    if has_train:
        ax_train.legend(fontsize=7, frameon=False, labelcolor=fg)

    ax_val.set_title("Validation Loss", fontweight="bold")
    ax_val.set_xlabel("Step")
    ax_val.set_ylabel("Loss")
    if has_val:
        ax_val.legend(fontsize=7, frameon=False, labelcolor=fg)

    if not has_train and not has_val:
        plt.close(fig)
        return None

    fig.tight_layout()
    return fig
