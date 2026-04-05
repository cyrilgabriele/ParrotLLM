# src/dashboard/metrics_reader.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingMetrics:
    steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_ppls: list[float] = field(default_factory=list)
    eval_train_losses: list[float] = field(default_factory=list)
    architecture: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    best_step: Optional[int] = None


def read_metrics(run_dir: Path) -> TrainingMetrics:
    """Parse metrics.jsonl from a run directory. Returns empty TrainingMetrics if missing."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return TrainingMetrics()

    m = TrainingMetrics()
    for raw in metrics_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        t = entry.get("type", "")
        if t == "model_architecture":
            m.architecture = {k: v for k, v in entry.items()
                              if k not in ("stage", "type", "timestamp")}
        elif t == "config":
            m.config = {k: v for k, v in entry.items()
                        if k not in ("stage", "type", "timestamp")}
        elif t == "step":
            m.steps.append(entry["step"])
            m.train_losses.append(entry["train_loss"])
            m.lrs.append(entry["lr"])
            m.grad_norms.append(entry.get("grad_norm", float("nan")))
            if "tokens_per_sec" in entry:
                m.tokens_per_sec.append(entry["tokens_per_sec"])
        elif t in ("eval", "initial_validation"):
            if "step" in entry:
                m.eval_steps.append(entry["step"])
            if "val_loss" in entry:
                m.val_losses.append(entry["val_loss"])
            if "val_ppl" in entry:
                m.val_ppls.append(entry["val_ppl"])
            if "eval_train_loss" in entry:
                m.eval_train_losses.append(entry["eval_train_loss"])
        elif t == "best_checkpoint":
            m.best_step = entry.get("step")

    return m


def is_metrics_stale(run_dir: Path, threshold: int = 60) -> tuple[bool, int]:
    """Return (is_stale, seconds_since_update). Returns (False, 0) if file absent."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return False, 0
    age = int(time.time() - metrics_path.stat().st_mtime)
    return age > threshold, age
