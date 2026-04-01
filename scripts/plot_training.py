"""
Training log plotter for ParrotLLM.

Usage:
    # Single run
    uv run scripts/plot_training.py runs/run_20260320_145519/

    # Multi-run comparison
    uv run scripts/plot_training.py runs/run_A/ runs/run_B/ runs/run_C/

    # Custom output path
    uv run scripts/plot_training.py runs/run_A/ --output results/plots.pdf
"""

import re
import json
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_RE_STEP = re.compile(
    r"step\s+(\d+)\s*\|.*?loss\s+([\d.]+)\s*\|.*?lr\s+([\d.e+-]+)\s*\|.*?grad\s+([\d.]+)"
)
_RE_PPL = re.compile(r"step\s+(\d+)\s*\|\s*ppl\s+([\d.]+)")
_RE_EVAL_TRAIN = re.compile(r"Train:\s*loss=([\d.]+),\s*ppl=([\d.]+)")
_RE_EVAL_VAL = re.compile(r"Val:\s*loss=([\d.]+),\s*ppl=([\d.]+)")
_RE_BEST = re.compile(r"\*\* New best validation loss! \*\*")
_RE_EVAL_START = re.compile(r"Starting evaluation\.\.\.")


def _label_from_config(run_dir: Path) -> str | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text())
        lr = cfg.get("training", {}).get("learning_rate", "?")
        layers = cfg.get("model", {}).get("n_layers", "?")
        d_model = cfg.get("model", {}).get("d_model", "?")
        return f"lr={lr}, layers={layers}, d={d_model}"
    except Exception:
        return None


def parse_log(log_path: Path, label: str | None = None) -> dict:
    """Parse a train.log file into a dict of time-series lists."""
    text = log_path.read_text()
    lines = text.splitlines()

    steps = []
    train_loss = []
    lr = []
    grad_norm = []
    train_ppl = []
    eval_steps = []
    val_loss = []
    val_ppl = []
    eval_train_loss = []
    eval_train_ppl = []
    best_val_step = None

    # Track current step for associating eval blocks
    current_step = None
    in_eval = False
    pending_train_loss = None
    pending_train_ppl = None

    for line in lines:
        m = _RE_STEP.search(line)
        if m:
            current_step = int(m.group(1))
            steps.append(current_step)
            train_loss.append(float(m.group(2)))
            lr.append(float(m.group(3)))
            grad_norm.append(float(m.group(4)))
            in_eval = False
            continue

        m = _RE_PPL.search(line)
        if m and not in_eval:
            train_ppl.append(float(m.group(2)))
            continue

        if _RE_EVAL_START.search(line):
            in_eval = True
            pending_train_loss = None
            pending_train_ppl = None
            continue

        if in_eval:
            m = _RE_EVAL_TRAIN.search(line)
            if m:
                pending_train_loss = float(m.group(1))
                pending_train_ppl = float(m.group(2))
                continue

            m = _RE_EVAL_VAL.search(line)
            if m:
                eval_steps.append(current_step)
                val_loss.append(float(m.group(1)))
                val_ppl.append(float(m.group(2)))
                eval_train_loss.append(pending_train_loss)
                eval_train_ppl.append(pending_train_ppl)
                continue

            if _RE_BEST.search(line):
                best_val_step = current_step
                continue

    if label is None:
        label = _label_from_config(log_path.parent) or log_path.parent.name

    return {
        "steps": steps,
        "train_loss": train_loss,
        "lr": lr,
        "grad_norm": grad_norm,
        "train_ppl": train_ppl,
        "eval_steps": eval_steps,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "eval_train_loss": eval_train_loss,
        "eval_train_ppl": eval_train_ppl,
        "best_val_step": best_val_step,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Qualitative color palette (up to 8 runs)
_PALETTE = [
    "#e05c5c", "#5c7ae0", "#5cba7a", "#e0a35c",
    "#a35ce0", "#5ce0d4", "#c9e05c", "#e05cb8",
]


def _apply_style(ax):
    """Apply NeurIPS/Nature minimal style to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def build_figure(runs: list[dict]) -> plt.Figure:
    """
    Build a 4-subplot figure from one or more parsed run dicts.

    Subplots (top to bottom):
      0: Train & Val Loss
      1: Val Perplexity (log scale)
      2: LR & Gradient Norm (dual y-axis)
      3: Train–Val Gap
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 11), sharex=False)
    is_comparison = len(runs) > 1

    for i, data in enumerate(runs):
        color = _PALETTE[i % len(_PALETTE)]
        label = data["label"] if is_comparison else None
        steps = data["steps"]
        eval_steps = data["eval_steps"]

        # --- Subplot 0: Train & Val Loss ---
        ax0 = axes[0]
        ax0.plot(steps, data["train_loss"], color=color, linewidth=1.5,
                 label=f"{label} train" if is_comparison else "train")
        if data["val_loss"]:
            ax0.plot(eval_steps, data["val_loss"], color=color, linewidth=1.5,
                     linestyle="--",
                     label=f"{label} val" if is_comparison else "val")
        if data["best_val_step"] is not None and not is_comparison:
            ax0.axvline(data["best_val_step"], color="#999", linewidth=0.8,
                        linestyle=":", label=f"best val (step {data['best_val_step']})")
        _apply_style(ax0)
        ax0.set_ylabel("Loss")
        ax0.set_title("Train & Validation Loss")
        ax0.legend(fontsize=7, frameon=False)

        # --- Subplot 1: Val Perplexity (log scale) ---
        ax1 = axes[1]
        if data["val_ppl"]:
            ax1.plot(eval_steps, data["val_ppl"], color=color, linewidth=1.5,
                     label=label)
        _apply_style(ax1)
        ax1.set_yscale("log")
        ax1.set_ylabel("Val Perplexity (log)")
        ax1.set_title("Validation Perplexity")
        if is_comparison:
            ax1.legend(fontsize=7, frameon=False)

        # --- Subplot 2: LR & Gradient Norm (dual y-axis) ---
        ax2 = axes[2]
        if i == 0:
            ax2_r = ax2.twinx()
            ax2._twin = ax2_r
        else:
            ax2_r = ax2._twin
        ax2.plot(steps, data["lr"], color=color, linewidth=1.5,
                 label=f"{label} LR" if is_comparison else "LR")
        ax2_r.plot(steps, data["grad_norm"], color=color, linewidth=1.5,
                   linestyle="--", alpha=0.7,
                   label=f"{label} grad norm" if is_comparison else "grad norm")
        _apply_style(ax2)
        ax2.spines["top"].set_visible(False)
        ax2_r.spines["top"].set_visible(False)
        ax2_r.spines["right"].set_color("#aaa")
        ax2.set_ylabel("Learning Rate", color="#333")
        ax2_r.set_ylabel("Gradient Norm", color="#888")
        ax2.set_title("Learning Rate & Gradient Norm")
        # Combine legends on last iteration
        if i == len(runs) - 1:
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_r.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False)

        # --- Subplot 3: Train–Val Gap ---
        ax3 = axes[3]
        if data["val_loss"] and data["eval_train_loss"]:
            gap = [v - t for v, t in zip(data["val_loss"], data["eval_train_loss"])]
            ax3.plot(eval_steps, gap, color=color, linewidth=1.5, label=label)
        _apply_style(ax3)
        ax3.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
        ax3.set_ylabel("Val − Train Loss")
        ax3.set_xlabel("Step")
        ax3.set_title("Train–Val Gap")
        if is_comparison:
            ax3.legend(fontsize=7, frameon=False)

    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_run(run_dir: Path) -> Path:
    """Return the train.log path inside a run directory."""
    log = run_dir / "train.log"
    if not log.exists():
        raise FileNotFoundError(f"No train.log found in {run_dir}")
    return log


def main():
    parser = argparse.ArgumentParser(
        description="Plot ParrotLLM training logs as a PDF."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="One or more run directories containing train.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <first_run_dir>/training_plots.pdf or comparison_plots.pdf)",
    )
    args = parser.parse_args()

    runs = []
    for run_dir in args.run_dirs:
        log_path = _resolve_run(run_dir)
        runs.append(parse_log(log_path))

    is_comparison = len(runs) > 1

    if args.output:
        out_path = args.output
    elif is_comparison:
        out_path = args.run_dirs[0] / "comparison_plots.pdf"
    else:
        out_path = args.run_dirs[0] / "training_plots.pdf"

    matplotlib.use("Agg")
    fig = build_figure(runs)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
