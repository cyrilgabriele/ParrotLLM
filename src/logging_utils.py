"""Logging utilities for ParrotLLM.

Two output channels:
1. Python logging -> human-readable .log files (+ console), configurable per component.
2. JSONL writer  -> machine-readable metrics per step/epoch for plot generation.
"""

import json
import logging
import os
from datetime import datetime, timezone


def init_logging(
    *,
    console_level: str = "INFO",
    component_levels: dict[str, str] | None = None,
) -> logging.Logger:
    """Set up console-only logging for non-training stages."""
    logger = logging.getLogger("parrotllm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if component_levels:
        for component, level in component_levels.items():
            child = logging.getLogger(f"parrotllm.{component}")
            child.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    return logger


def setup_logger(
    run_dir: str,
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    component_levels: dict[str, str] | None = None,
) -> logging.Logger:
    """Configure the root 'parrotllm' logger with console + file handlers.

    Args:
        run_dir: directory for this run; the log file is written here.
        console_level: minimum level printed to stderr.
        file_level: minimum level written to the .log file.
        component_levels: optional per-component overrides,
            e.g. {"data_preprocessing": "DEBUG", "model_initialization": "WARNING"}.

    Returns:
        The configured root logger.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    logger = logging.getLogger("parrotllm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Per-component log levels
    if component_levels:
        for component, level in component_levels.items():
            child = logging.getLogger(f"parrotllm.{component}")
            child.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    logger.info(f"Logging initialised -> {log_path}")
    return logger


class JSONLLogger:
    """Append-only JSONL writer for structured metrics.

    Each line is a self-contained JSON object with at least
    ``stage``, ``type``, and ``timestamp`` fields.
    """

    def __init__(self, run_dir: str, filename: str = "metrics.jsonl"):
        os.makedirs(run_dir, exist_ok=True)
        self._path = os.path.join(run_dir, filename)
        self._file = open(self._path, "a", encoding="utf-8")

    def log(self, stage: str, entry_type: str, **kwargs) -> None:
        record = {
            "stage": stage,
            "type": entry_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def fmt_param_count(n: int) -> str:
    """Format a parameter count with human-readable suffix."""
    if n >= 1_000_000:
        return f"{n:,} ({n / 1e6:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1e3:.1f}K)"
    return f"{n:,}"


def fmt_model_summary(
    mc: dict,
    *,
    n_params: int,
    n_non_emb: int,
    pos_emb_params: int,
    n_trainable: int,
    n_non_trainable: int,
    params_size_mb: float,
    torchinfo: str | None = None,
) -> str:
    """Build a multi-line model architecture summary string."""
    parts = [
        "\n" + "=" * 60,
        "MODEL ARCHITECTURE SUMMARY",
        "=" * 60,
        "",
        "Configuration:",
        f"  Vocab size: {mc['vocab_size']}",
        f"  Block size (context): {mc['context_length']}",
        f"  Layers: {mc['n_layers']}",
        f"  Heads: {mc['n_heads']}",
        f"  Embedding dim: {mc['d_model']}",
        f"  FFN hidden dim: {mc['d_ff']}",
        f"  Dropout: {mc.get('dropout', 0.0)}",
        f"  Bias: {mc.get('bias', False)}",
        "",
        "Parameters (unique, weight-tied layers counted once):",
        f"  Total: {fmt_param_count(n_params)}",
        f"  Trainable: {fmt_param_count(n_trainable)}",
        f"  Non-trainable: {fmt_param_count(n_non_trainable)}",
        f"  Non-embedding: {fmt_param_count(n_non_emb)}",
        f"  Position embeddings: {fmt_param_count(pos_emb_params)}",
        f"  Size (MB): {params_size_mb:.2f}",
    ]
    if torchinfo:
        parts += [
            "",
            "Layer-wise breakdown (torchinfo):",
            "  Note: torchinfo double-counts weight-tied layers (tok_emb/lm_head).",
            torchinfo,
        ]
    parts.append("=" * 60)
    return "\n".join(parts)


def fmt_training_start(steps_per_epoch: int, max_steps: int) -> str:
    """Build the training-start banner string."""
    return (
        "\n" + "=" * 60
        + "\nStarting training..."
        + f"\n  Steps per epoch (approx): {steps_per_epoch}"
        + f"\n  Max steps: {max_steps}"
        + "\n" + "=" * 60
    )


def fmt_training_complete(
    epochs: int, total_steps: int, total_hours: float,
    best_val_loss: float, run_dir: str,
) -> str:
    """Build the training-complete summary string."""
    return (
        "\n" + "=" * 60
        + "\nTRAINING COMPLETE"
        + "\n" + "=" * 60
        + f"\n  Epochs: {epochs}"
        + f"\n  Total steps: {total_steps}"
        + f"\n  Total time: {total_hours:.2f} hours"
        + f"\n  Best validation loss: {best_val_loss:.4f}"
        + f"\n  Run directory: {run_dir}"
        + "\n" + "=" * 60
    )


def render_ascii_loss_curve(
    loss_history: list[tuple[int, float]], width: int = 50, height: int = 8,
) -> str:
    """Render an ASCII loss curve and return it as a single string."""
    if not loss_history:
        return ""

    steps = [s for s, _ in loss_history]
    losses = [l for _, l in loss_history]
    min_loss, max_loss = min(losses), max(losses)
    min_step, max_step = min(steps), max(steps)

    if max_loss == min_loss:
        max_loss = min_loss + 1

    grid = [[" " for _ in range(width)] for _ in range(height)]
    step_range = max(max_step - min_step, 1)
    loss_range = max_loss - min_loss

    for s, l in loss_history:
        x = int((s - min_step) / step_range * (width - 1))
        y = int((l - min_loss) / loss_range * (height - 1))
        y = height - 1 - y
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"

    lines = ["Loss Curve:", "  Loss", "  ^"]
    for i, row in enumerate(grid):
        if i == 0:
            label = f"{max_loss:>8.2f}"
        elif i == height - 1:
            label = f"{min_loss:>8.2f}"
        else:
            label = "        "
        lines.append(f"{label} |{''.join(row)}")
    lines.append(f"         +{'-' * width}> step")
    lines.append(f"         {min_step:<{width // 2}}{max_step:>{width - width // 2}}")
    return "\n".join(lines)


def make_run_dir(runs_dir: str = "runs", tag: str | None = None) -> str:
    """Create a timestamped run directory, e.g. runs/run_20260306_143641[_tag]/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" if tag is None else f"run_{ts}_{tag}"
    run_dir = os.path.join(runs_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
