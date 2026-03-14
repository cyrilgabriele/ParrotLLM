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


def make_run_dir(runs_dir: str = "runs", tag: str | None = None) -> str:
    """Create a timestamped run directory, e.g. runs/run_20260306_143641[_tag]/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" if tag is None else f"run_{ts}_{tag}"
    run_dir = os.path.join(runs_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
