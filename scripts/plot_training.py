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
        label = log_path.parent.name

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
