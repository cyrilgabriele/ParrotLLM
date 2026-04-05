# src/dashboard/run_manager.py
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.dashboard.metrics_reader import read_metrics


@dataclass
class RunInfo:
    name: str
    run_dir: Path
    last_step: Optional[int]
    best_val_loss: Optional[float]


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Return all runs sorted newest-first by directory name."""
    if not runs_dir.exists():
        return []
    dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    result = []
    for d in dirs:
        m = read_metrics(d)
        result.append(RunInfo(
            name=d.name,
            run_dir=d,
            last_step=m.steps[-1] if m.steps else None,
            best_val_loss=m.val_losses[m.eval_steps.index(m.best_step)]
                if m.best_step and m.best_step in m.eval_steps and m.val_losses else None,
        ))
    return result


def get_latest_run_dir(runs_dir: Path) -> Optional[Path]:
    """Return the most recent run directory, or None if none exist."""
    runs = list_runs(runs_dir)
    return runs[0].run_dir if runs else None


def launch_training(
    config_path: Path,
    resume_run_dir: Optional[Path] = None,
) -> subprocess.Popen:
    """Launch training as a subprocess. Returns the Popen handle."""
    cmd = [
        "uv", "run", "python", "main.py",
        "--stage", "train",
        "--config", str(config_path),
    ]
    if resume_run_dir is not None:
        cmd += ["--resume", str(resume_run_dir)]
    return subprocess.Popen(cmd)


def kill_training(proc: subprocess.Popen) -> None:
    """Send SIGTERM to proc if it is still alive."""
    if proc is not None and proc.poll() is None:
        proc.terminate()


def get_log_lines(run_dir: Path, n: int = 20) -> list[str]:
    """Return the last n lines from train.log in run_dir, or [] if unavailable."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []
