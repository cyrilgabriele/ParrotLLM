"""Logging utilities for ParrotLLM.

Two output channels:
1. Python logging -> human-readable .log files (+ console), configurable per component.
2. JSONL writer  -> machine-readable metrics per step/epoch for plot generation.
"""

import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Mapping

try:  # Optional import to avoid forcing torch during preprocess-only workflows
    import torch
except Exception:  # pragma: no cover - torch is a runtime dependency for training
    torch = None  # type: ignore

try:  # torch.profiler is available in PyTorch >= 1.8
    from torch import profiler as torch_profiler
except Exception:  # pragma: no cover
    torch_profiler = None  # type: ignore


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


class TorchProfiler:
    """Thin wrapper around torch.profiler with ParrotLLM logging hooks."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None,
        run_dir: str | None,
        logger: logging.Logger | None = None,
        json_logger: JSONLLogger | None = None,
        enabled: bool = True,
    ) -> None:
        self._logger = logger or logging.getLogger("parrotllm")
        self._json_logger = json_logger
        self._config = dict(config or {})
        self._run_dir = run_dir
        self._enabled = bool(enabled and self._config.get("enabled", False))
        self._profiler = None
        self._trace_dir = None
        self._tb_handler = None
        self._trace_index = 0
        self._last_meta: dict[str, Any] = {}

        if not self._enabled:
            return
        if run_dir is None:
            self._logger.warning("Profiler requested without a run directory; disabling.")
            self._enabled = False
            return
        if torch is None or torch_profiler is None:
            self._logger.warning("PyTorch profiler unavailable; install torch>=1.8 to enable it.")
            self._enabled = False
            return

        activities = self._resolve_activities(self._config.get("activities"))
        if not activities:
            self._logger.warning("Profiler requested zero activities; disabling.")
            self._enabled = False
            return

        wait = int(self._config.get("wait", 1))
        warmup = int(self._config.get("warmup", 1))
        active = max(1, int(self._config.get("active", 1)))
        repeat = max(1, int(self._config.get("repeat", 1)))
        skip_first = int(self._config.get("skip_first", 0))
        self._schedule = torch_profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first
        )

        trace_subdir = self._config.get("trace_subdir", "profiler")
        self._trace_dir = os.path.join(run_dir, trace_subdir)
        os.makedirs(self._trace_dir, exist_ok=True)

        if self._config.get("export_tensorboard", False):
            tb_subdir = self._config.get("tensorboard_subdir", "tensorboard")
            tb_dir = os.path.join(self._trace_dir, tb_subdir)
            os.makedirs(tb_dir, exist_ok=True)
            self._tb_handler = torch_profiler.tensorboard_trace_handler(tb_dir)

        self._export_chrome = bool(self._config.get("export_chrome_trace", True))
        self._summary_enabled = bool(self._config.get("log_summary", True))
        self._summary_sort = self._config.get("summary_sort_by", "self_cpu_time_total")
        self._summary_rows = int(self._config.get("summary_row_limit", 20))

        self._profiler = torch_profiler.profile(
            activities=activities,
            schedule=self._schedule,
            record_shapes=bool(self._config.get("record_shapes", True)),
            profile_memory=bool(self._config.get("profile_memory", True)),
            with_stack=bool(self._config.get("with_stack", False)),
            with_modules=bool(self._config.get("with_modules", False)),
            on_trace_ready=self._handle_trace,
        )

        self._logger.info(
            "PyTorch profiler enabled: wait=%d warmup=%d active=%d repeat=%d activities=%s",
            wait,
            warmup,
            active,
            repeat,
            ",".join(self._config.get("activities", ["cpu", "cuda"])),
        )
        if self._json_logger is not None:
            self._json_logger.log(
                "profiling",
                "enabled",
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
                activities=self._config.get("activities", ["cpu", "cuda"]),
            )

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._profiler is not None)

    def record_function(self, name: str):
        if not self.enabled or torch_profiler is None:
            return nullcontext()
        return torch_profiler.record_function(name)

    def step(self, *, step: int, epoch: int | None = None) -> None:
        if not self.enabled:
            return
        self._last_meta = {"step": step, "epoch": epoch}
        self._profiler.step()

    def __enter__(self):
        if self.enabled:
            self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._profiler.__exit__(exc_type, exc, tb)
            self._log_summary()
        return False

    # Internal helpers -----------------------------------------------------

    def _resolve_activities(self, values: list[str] | None):
        if torch_profiler is None:
            return []
        requested = values or ["cpu", "cuda"]
        resolved = []
        for name in requested:
            key = name.lower()
            if key == "cpu":
                resolved.append(torch_profiler.ProfilerActivity.CPU)
            elif key == "cuda":
                if torch is not None and torch.cuda.is_available():
                    resolved.append(torch_profiler.ProfilerActivity.CUDA)
            else:
                self._logger.warning("Unknown profiler activity '%s'", name)
        return resolved

    def _handle_trace(self, prof) -> None:
        if not self.enabled:
            return
        self._trace_index += 1
        step = self._last_meta.get("step")
        epoch = self._last_meta.get("epoch")
        trace_name = f"trace_{self._trace_index:04d}_step_{step if step is not None else 'na'}.json"
        if self._export_chrome and self._trace_dir is not None:
            trace_path = os.path.join(self._trace_dir, trace_name)
            try:
                prof.export_chrome_trace(trace_path)
                self._logger.info("Profiler trace #%d saved to %s", self._trace_index, trace_path)
                if self._json_logger is not None:
                    self._json_logger.log(
                        "profiling",
                        "trace",
                        step=step,
                        epoch=epoch,
                        path=trace_path,
                    )
            except Exception as exc:  # pragma: no cover - export errors should be rare
                self._logger.error("Failed to export profiler trace: %s", exc)
        if self._tb_handler is not None:
            try:
                self._tb_handler(prof)
            except Exception as exc:  # pragma: no cover
                self._logger.error("Failed to write profiler TensorBoard data: %s", exc)

    def _log_summary(self) -> None:
        if not self.enabled or not self._summary_enabled:
            return
        try:
            table = self._profiler.key_averages().table(
                sort_by=self._summary_sort,
                row_limit=self._summary_rows,
            )
            self._logger.info(
                "PyTorch profiler summary (sort=%s, rows=%d):\n%s",
                self._summary_sort,
                self._summary_rows,
                table,
            )
            if self._json_logger is not None:
                self._json_logger.log(
                    "profiling",
                    "summary",
                    sort_by=self._summary_sort,
                    row_limit=self._summary_rows,
                )
        except Exception as exc:
            self._logger.warning("Unable to render profiler summary: %s", exc)

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
        f"  Position embeddings: {fmt_param_count(pos_emb_params)} (RoPE: 0 learned params)",
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
