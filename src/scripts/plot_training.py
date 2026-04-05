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
import math
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported
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
        "tokens_per_sec": [math.nan] * len(steps),   # not tracked in train.log
        "eval_steps": eval_steps,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "eval_train_loss": eval_train_loss,
        "eval_train_ppl": eval_train_ppl,
        "best_val_step": best_val_step,
        "label": label,
    }


def parse_metrics(run_dir: Path, label: str | None = None) -> dict:
    """Parse a metrics.jsonl file into the same dict format as parse_log."""
    metrics_path = run_dir / "metrics.jsonl"
    text = metrics_path.read_text()

    steps: list[int] = []
    train_loss: list[float] = []
    lr: list[float] = []
    grad_norm: list[float] = []
    train_ppl: list[float] = []
    tokens_per_sec: list[float] = []
    eval_steps: list[int] = []
    val_loss: list[float] = []
    val_ppl: list[float] = []
    eval_train_loss: list = []
    eval_train_ppl: list = []
    best_val_step: int | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = entry.get("type")

        if etype == "step":
            steps.append(entry["step"])
            train_loss.append(entry["train_loss"])
            lr.append(entry["lr"])
            train_ppl.append(entry["perplexity"])
            grad_norm.append(float(entry["grad_norm"]) if entry.get("grad_norm") is not None else math.nan)
            tokens_per_sec.append(float(entry["tokens_per_sec"]) if entry.get("tokens_per_sec") is not None else math.nan)

        elif etype == "eval":
            eval_steps.append(entry["step"])
            val_loss.append(entry["val_loss"])
            val_ppl.append(entry["val_ppl"])
            eval_train_loss.append(entry.get("eval_train_loss"))
            eval_train_ppl.append(entry.get("eval_train_ppl"))

        elif etype == "best_checkpoint":
            best_val_step = entry["step"]

    if label is None:
        label = _label_from_config(run_dir) or run_dir.name

    return {
        "steps": steps,
        "train_loss": train_loss,
        "lr": lr,
        "grad_norm": grad_norm,
        "train_ppl": train_ppl,
        "tokens_per_sec": tokens_per_sec,
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

# Single-run colours: blue = train, orange = validation
TRAIN_COLOR = "#2563EB"
VAL_COLOR   = "#EA580C"

# Multi-run comparison palette (up to 8 runs; solid=train, dashed=val)
_PALETTE = [
    "#e05c5c", "#5c7ae0", "#5cba7a", "#e0a35c",
    "#a35ce0", "#5ce0d4", "#c9e05c", "#e05cb8",
]


def _apply_style(ax):
    """Apply minimal style to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _ema(values: list[float], alpha: float = 0.98) -> list[float]:
    """Exponential moving average. s[0]=values[0]; s[i]=alpha*s[i-1]+(1-alpha)*values[i]."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


def build_figure(runs: list[dict]) -> plt.Figure:
    """
    Build a 3×2 subplot figure from one or more parsed run dicts.

    Layout:
      [0,0] Train & Val Loss (raw dimmed + EMA overlay)   [0,1] Train & Val Perplexity (log, raw dimmed + EMA)
      [1,0] Learning Rate                                  [1,1] Gradient Norm
      [2,0] Training Throughput (tokens/sec)               [2,1] Train–Val Gap
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=False)
    is_comparison = len(runs) > 1

    for i, data in enumerate(runs):
        run_col = _PALETTE[i % len(_PALETTE)] if is_comparison else None
        tc = run_col if is_comparison else TRAIN_COLOR
        vc = run_col if is_comparison else VAL_COLOR
        label = data["label"] if is_comparison else None
        steps = data["steps"]
        eval_steps = data["eval_steps"]

        # ── [0,0] Train & Val Loss ────────────────────────────────────────
        ax = axes[0, 0]
        if steps and data["train_loss"]:
            ax.plot(steps, data["train_loss"], color=tc, linewidth=0.8, alpha=0.25)
            ax.plot(steps, _ema(data["train_loss"]), color=tc, linewidth=1.5,
                    label=(f"{label} train" if is_comparison else "train"))
        if data["val_loss"]:
            ax.plot(eval_steps, data["val_loss"], color=vc, linewidth=1.5,
                    linestyle="--",
                    label=(f"{label} val" if is_comparison else "val"))
        if data["best_val_step"] is not None and not is_comparison:
            ax.axvline(data["best_val_step"], color="#999", linewidth=0.8,
                       linestyle=":",
                       label=f"best val (step {data['best_val_step']})")
        _apply_style(ax)
        ax.set_ylabel("Loss")
        ax.set_title("Train & Validation Loss")
        ax.legend(fontsize=7, frameon=False)

        # ── [0,1] Train & Val Perplexity (log scale) ─────────────────────
        ax = axes[0, 1]
        if steps and data["train_ppl"]:
            ax.plot(steps, data["train_ppl"], color=tc, linewidth=0.8, alpha=0.25)
            ax.plot(steps, _ema(data["train_ppl"]), color=tc, linewidth=1.5,
                    label=(f"{label} train" if is_comparison else "train"))
        if data["val_ppl"]:
            ax.plot(eval_steps, data["val_ppl"], color=vc, linewidth=1.5,
                    linestyle="--",
                    label=(f"{label} val" if is_comparison else "val"))
        _apply_style(ax)
        ax.set_yscale("log")
        ax.set_ylabel("Perplexity (log)")
        ax.set_title("Train & Validation Perplexity")
        ax.legend(fontsize=7, frameon=False)

        # ── [1,0] Learning Rate ───────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(steps, data["lr"], color=tc, linewidth=1.5, label=label)
        _apply_style(ax)
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        if is_comparison:
            ax.legend(fontsize=7, frameon=False)

        # ── [1,1] Gradient Norm ───────────────────────────────────────────
        ax = axes[1, 1]
        valid_gn = [
            (s, g) for s, g in zip(steps, data["grad_norm"])
            if not math.isnan(g)
        ]
        if valid_gn:
            s_v, g_v = zip(*valid_gn)
            ax.plot(s_v, g_v, color=tc, linewidth=1.5, alpha=0.85, label=label)
        _apply_style(ax)
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm")
        if is_comparison:
            ax.legend(fontsize=7, frameon=False)

        # ── [2,0] Training Throughput (tokens / second) ───────────────────
        ax = axes[2, 0]
        tps = data.get("tokens_per_sec", [])
        valid_tps = [
            (s, t) for s, t in zip(steps, tps)
            if not math.isnan(t) and t > 0
        ]
        if valid_tps:
            s_v, t_v = zip(*valid_tps)
            ax.plot(s_v, t_v, color=tc, linewidth=1.5, label=label)
        _apply_style(ax)
        ax.set_ylabel("Tokens / second")
        ax.set_xlabel("Step")
        ax.set_title("Training Throughput")
        if is_comparison:
            ax.legend(fontsize=7, frameon=False)

        # ── [2,1] Train–Val Gap ───────────────────────────────────────────
        ax = axes[2, 1]
        if data["val_loss"] and data["eval_train_loss"] and all(
            x is not None for x in data["eval_train_loss"]
        ):
            gap = [v - t for v, t in zip(data["val_loss"], data["eval_train_loss"])]
            ax.plot(eval_steps, gap, color=vc, linewidth=1.5, label=label)
        _apply_style(ax)
        ax.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Val − Train Loss")
        ax.set_xlabel("Step")
        ax.set_title("Train–Val Gap")
        if is_comparison:
            ax.legend(fontsize=7, frameon=False)

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


def plot_run_dir(run_dir: Path, output: Path | None = None) -> Path:
    """Generate training plots for a run directory and save as PDF.

    Reads ``metrics.jsonl`` when present; falls back to ``train.log`` for
    older runs that pre-date structured JSONL logging.

    Returns the path of the saved PDF.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        data = parse_metrics(run_dir)
    else:
        data = parse_log(_resolve_run(run_dir))

    if output is None:
        output = run_dir / "training_plots.pdf"

    fig = build_figure([data])
    fig.savefig(output, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Plot ParrotLLM training logs as a PDF."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="One or more run directories containing metrics.jsonl or train.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <first_run_dir>/training_plots.pdf or comparison_plots.pdf)",
    )
    args = parser.parse_args()

    if len(args.run_dirs) == 1 and args.output is None:
        out = plot_run_dir(args.run_dirs[0])
        print(f"Saved: {out}")
        return

    runs = []
    for run_dir in args.run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            runs.append(parse_metrics(run_dir))
        else:
            runs.append(parse_log(_resolve_run(run_dir)))

    out_path = args.output or (args.run_dirs[0] / "comparison_plots.pdf")
    fig = build_figure(runs)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
