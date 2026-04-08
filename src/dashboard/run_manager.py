# src/dashboard/run_manager.py
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.dashboard.metrics_reader import read_metrics, is_metrics_stale, TrainingMetrics


@dataclass
class CheckpointInfo:
    path: Path
    name: str
    step: Optional[int]
    size_mb: float
    is_best: bool


@dataclass
class RunInfo:
    name: str
    rel_path: str          # relative path from runs_dir
    run_dir: Path
    last_step: Optional[int]
    best_val_loss: Optional[float]
    status: str            # "live", "completed", "crashed", "stopped", "empty"
    max_steps: Optional[int] = None


# ── Status detection ──────────────────────────────────────────────────────────

_STEP_RE = re.compile(r"step_(\d+)")
_LOSS_RE = re.compile(r"loss_([\dp]+)")


def _detect_status(run_dir: Path, metrics: TrainingMetrics) -> str:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return "empty"

    max_steps = metrics.config.get("max_steps")
    if metrics.steps and max_steps and metrics.steps[-1] >= max_steps:
        return "completed"

    stale, age = is_metrics_stale(run_dir, threshold=30)
    if not stale:
        return "live"
    if age > 300:
        return "crashed"
    return "stopped"


# ── Run listing ───────────────────────────────────────────────────────────────

def _build_run_info(run_dir: Path, runs_dir: Path) -> RunInfo:
    m = read_metrics(run_dir)
    rel = str(run_dir.relative_to(runs_dir))
    status = _detect_status(run_dir, m)
    best_val = None
    if m.best_step and m.best_step in m.eval_steps and m.val_losses:
        idx = m.eval_steps.index(m.best_step)
        if idx < len(m.val_losses):
            best_val = m.val_losses[idx]
    return RunInfo(
        name=run_dir.name,
        rel_path=rel,
        run_dir=run_dir,
        last_step=m.steps[-1] if m.steps else None,
        best_val_loss=best_val,
        status=status,
        max_steps=m.config.get("max_steps"),
    )


def list_runs(runs_dir: Path, subdir: str = "") -> list[RunInfo]:
    """Return runs at the given level (non-recursive), live runs first."""
    target = runs_dir / subdir if subdir else runs_dir
    if not target.exists():
        return []

    result = []
    for d in target.iterdir():
        if not d.is_dir():
            continue
        if not (d / "metrics.jsonl").exists():
            continue
        result.append(_build_run_info(d, runs_dir))

    live = sorted([r for r in result if r.status == "live"], key=lambda r: r.name, reverse=True)
    rest = sorted([r for r in result if r.status != "live"], key=lambda r: r.name, reverse=True)
    return live + rest


def list_subdirs(runs_dir: Path, subdir: str = "") -> list[str]:
    """Return subdirectory names at the given level that are not runs themselves."""
    target = runs_dir / subdir if subdir else runs_dir
    if not target.exists():
        return []

    subdirs = []
    for d in sorted(target.iterdir()):
        if not d.is_dir():
            continue
        if (d / "metrics.jsonl").exists():
            continue  # this is a run, not a folder
        # Check if it has any descendant runs
        try:
            if any(True for _ in d.rglob("metrics.jsonl")):
                subdirs.append(d.name)
        except PermissionError:
            continue
    return subdirs


def find_all_live_runs(runs_dir: Path) -> list[RunInfo]:
    """Recursively find all currently-active runs."""
    result = []
    try:
        for mp in runs_dir.rglob("metrics.jsonl"):
            run_dir = mp.parent
            m = read_metrics(run_dir)
            if _detect_status(run_dir, m) == "live":
                result.append(_build_run_info(run_dir, runs_dir))
    except PermissionError:
        pass
    return sorted(result, key=lambda r: r.name, reverse=True)


def get_latest_run_dir(runs_dir: Path) -> Optional[Path]:
    """Return the most recent run directory (by mtime of metrics.jsonl), or None."""
    best, best_mtime = None, 0.0
    try:
        for mp in runs_dir.rglob("metrics.jsonl"):
            mt = mp.stat().st_mtime
            if mt > best_mtime:
                best_mtime = mt
                best = mp.parent
    except PermissionError:
        pass
    return best


# ── Checkpoint listing ────────────────────────────────────────────────────────

def list_checkpoints(run_dir: Path) -> list[CheckpointInfo]:
    """List checkpoint files in a run directory."""
    checkpoints: list[CheckpointInfo] = []
    ckpt_dir = run_dir / "checkpoints"

    search_dirs = [run_dir]
    if ckpt_dir.exists():
        search_dirs.append(ckpt_dir)

    for sd in search_dirs:
        for f in sd.glob("*.pt"):
            step_m = _STEP_RE.search(f.stem)
            step = int(step_m.group(1)) if step_m else None
            size_mb = f.stat().st_size / (1024 * 1024)
            is_best = "best" in f.stem.lower()
            checkpoints.append(CheckpointInfo(
                path=f, name=f.name, step=step,
                size_mb=size_mb, is_best=is_best,
            ))

    # Also check numbered subdirectories (epoch_*/step_*)
    for d in run_dir.iterdir():
        if d.is_dir() and ("epoch" in d.name or "step" in d.name):
            for f in d.glob("*.pt"):
                step_m = _STEP_RE.search(d.name) or _STEP_RE.search(f.stem)
                step = int(step_m.group(1)) if step_m else None
                size_mb = f.stat().st_size / (1024 * 1024)
                is_best = "best" in f.stem.lower()
                checkpoints.append(CheckpointInfo(
                    path=f, name=f"{d.name}/{f.name}", step=step,
                    size_mb=size_mb, is_best=is_best,
                ))

    checkpoints.sort(key=lambda c: (c.step or 0), reverse=True)
    return checkpoints


# ── Config diff ───────────────────────────────────────────────────────────────

def get_config_diff(run_dirs: list[Path]) -> dict[str, dict[str, object]]:
    """Compare configs across runs. Returns {key: {run_name: value}} for keys that differ."""
    configs: dict[str, dict] = {}
    for d in run_dirs:
        m = read_metrics(d)
        configs[d.name] = m.config

    all_keys: set[str] = set()
    for cfg in configs.values():
        all_keys.update(cfg.keys())

    diff: dict[str, dict[str, object]] = {}
    for key in sorted(all_keys):
        values = {name: cfg.get(key, "—") for name, cfg in configs.items()}
        if len(set(str(v) for v in values.values())) > 1:
            diff[key] = values
    return diff


# ── Training control ──────────────────────────────────────────────────────────

def launch_training(
    config_path: Path,
    resume_run_dir: Optional[Path] = None,
) -> subprocess.Popen:
    cmd = [
        "uv", "run", "python", "main.py",
        "--stage", "train",
        "--config", str(config_path),
    ]
    if resume_run_dir is not None:
        cmd += ["--resume", str(resume_run_dir)]
    return subprocess.Popen(cmd)


def kill_training(proc: subprocess.Popen) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()


def get_log_lines(run_dir: Path, n: int = 20) -> list[str]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []
