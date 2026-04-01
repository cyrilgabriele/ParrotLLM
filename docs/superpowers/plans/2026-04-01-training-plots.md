# Training Plots Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single script `scripts/plot_training.py` that parses ParrotLLM training logs and produces a 4-subplot NeurIPS-style PDF, supporting single-run and multi-run comparison modes.

**Architecture:** Single flat script (~200 lines). Regex-based log parser builds plain dicts of time-series data. A plotting function takes one or more data dicts and produces a single-column 4-subplot PDF. Mode (single vs comparison) is inferred from the number of positional CLI arguments.

**Tech Stack:** Python stdlib (`re`, `argparse`, `pathlib`, `json`), `matplotlib`, `numpy`

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `scripts/plot_training.py` | CLI entry point + parser + plotter, all in one |
| Modify | `pyproject.toml` | Add `matplotlib` dependency |
| Create | `tests/test_plot_training.py` | Unit tests for the parser and plot output |

---

## Task 1: Add matplotlib dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add matplotlib to pyproject.toml**

In `pyproject.toml`, add `"matplotlib>=3.9.0"` to the `dependencies` list:

```toml
dependencies = [
    "torch>=2.6.0",
    "transformers>=4.48.0",
    "datasets>=3.0.0",
    "numpy>=2.0.0",
    "matplotlib>=3.9.0",
    ...
]
```

- [ ] **Step 2: Sync the venv**

```bash
uv sync
```

Expected: output shows `matplotlib` and its deps installed, no errors.

- [ ] **Step 3: Verify import**

```bash
uv run python -c "import matplotlib; print(matplotlib.__version__)"
```

Expected: prints a version string like `3.9.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add matplotlib dependency for training plots"
```

---

## Task 2: Write the log parser (TDD)

**Files:**
- Create: `tests/test_plot_training.py`
- Create: `scripts/plot_training.py` (parser portion only)

The parser reads a `train.log` file and returns a dict with these keys:

```python
{
    "steps":      [0, 25, 50, ...],        # int list — all logged training steps
    "train_loss": [10.87, 9.78, ...],      # float list — loss at each logged step
    "lr":         [2e-05, 5.2e-04, ...],   # float list — LR at each logged step
    "grad_norm":  [3.65, 1.01, ...],       # float list — grad norm at each logged step
    "train_ppl":  [52964, 17793, ...],     # float list — train ppl at each logged step (from DEBUG lines)
    "eval_steps": [100, 200, 300, ...],    # int list — steps where eval ran
    "val_loss":   [7.17, 6.66, ...],       # float list — val loss at each eval step
    "val_ppl":    [1300, 787, ...],        # float list — val ppl at each eval step
    "eval_train_loss": [7.05, 6.43, ...],  # float list — train loss recorded at eval time
    "eval_train_ppl":  [1159, 625, ...],   # float list — train ppl recorded at eval time
    "best_val_step": 400,                  # int — step where best val loss was achieved
    "label": "run_20260320_145519",        # str — run label for legends
}
```

- [ ] **Step 1: Create the test file with a minimal fixture log**

Create `tests/test_plot_training.py`:

```python
import sys
from pathlib import Path
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from plot_training import parse_log

SAMPLE_LOG = """
2026-03-20 14:55:19 - INFO - parrotllm.training - Logging initialised -> runs/run_test/train.log
2026-03-20 14:55:29 - INFO - parrotllm.training - step      0 | epoch 0 | loss 10.8774 | lr 2.00e-05 | grad 3.6559
2026-03-20 14:55:29 - DEBUG - parrotllm.training - step      0 | ppl 52964.37 | dt 3.7s
2026-03-20 14:55:49 - INFO - parrotllm.training - step     25 | epoch 0 | loss 9.7866 | lr 5.20e-04 | grad 1.0129
2026-03-20 14:55:49 - DEBUG - parrotllm.training - step     25 | ppl 17793.38 | dt 16.3s
2026-03-20 14:56:37 - INFO - parrotllm.training - step    100 | epoch 0 | loss 7.0557 | lr 1.00e-03 | grad 0.8274
2026-03-20 14:56:37 - DEBUG - parrotllm.training - step    100 | ppl 1159.44 | dt 16.0s
2026-03-20 14:56:37 - INFO - parrotllm.training - Starting evaluation...
2026-03-20 14:56:37 - INFO - parrotllm.training - ------------------------------------------------------------
2026-03-20 14:56:39 - INFO - parrotllm.training -   Train: loss=7.0557, ppl=1159.44
2026-03-20 14:56:39 - INFO - parrotllm.training -   Val:   loss=7.1702, ppl=1300.12
2026-03-20 14:56:39 - INFO - parrotllm.training -   ** New best validation loss! **
2026-03-20 14:56:39 - INFO - parrotllm.training - ------------------------------------------------------------
""".strip()


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(SAMPLE_LOG)
    return p


def test_parse_steps(log_file):
    data = parse_log(log_file)
    assert data["steps"] == [0, 25, 100]


def test_parse_train_loss(log_file):
    data = parse_log(log_file)
    assert data["train_loss"] == pytest.approx([10.8774, 9.7866, 7.0557])


def test_parse_lr(log_file):
    data = parse_log(log_file)
    assert data["lr"] == pytest.approx([2e-05, 5.2e-04, 1e-03])


def test_parse_grad_norm(log_file):
    data = parse_log(log_file)
    assert data["grad_norm"] == pytest.approx([3.6559, 1.0129, 0.8274])


def test_parse_train_ppl(log_file):
    data = parse_log(log_file)
    assert data["train_ppl"] == pytest.approx([52964.37, 17793.38, 1159.44])


def test_parse_eval(log_file):
    data = parse_log(log_file)
    assert data["eval_steps"] == [100]
    assert data["val_loss"] == pytest.approx([7.1702])
    assert data["val_ppl"] == pytest.approx([1300.12])
    assert data["eval_train_loss"] == pytest.approx([7.0557])
    assert data["eval_train_ppl"] == pytest.approx([1159.44])


def test_parse_best_val_step(log_file):
    data = parse_log(log_file)
    assert data["best_val_step"] == 100


def test_parse_label_fallback(log_file):
    data = parse_log(log_file, label="my_run")
    assert data["label"] == "my_run"
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: `ModuleNotFoundError: No module named 'plot_training'` or similar — script doesn't exist yet.

- [ ] **Step 3: Create `scripts/plot_training.py` with the `parse_log` function**

Create `scripts/plot_training.py`:

```python
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


def parse_log(log_path: Path, label: str = None) -> dict:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: add training log parser with tests"
```

---

## Task 3: Run label from config.json

**Files:**
- Modify: `scripts/plot_training.py`
- Modify: `tests/test_plot_training.py`

- [ ] **Step 1: Add test for config-derived label**

Append to `tests/test_plot_training.py`:

```python
def test_parse_label_from_config(log_file):
    import json
    config = {
        "training": {"learning_rate": 0.001},
        "model": {"n_layers": 4, "d_model": 128}
    }
    config_path = log_file.parent / "config.json"
    config_path.write_text(json.dumps(config))
    data = parse_log(log_file)
    assert data["label"] == "lr=0.001, layers=4, d=128"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_plot_training.py::test_parse_label_from_config -v
```

Expected: FAIL — label is the directory name, not the config string.

- [ ] **Step 3: Add config label extraction to `parse_log`**

Add a helper and update `parse_log` in `scripts/plot_training.py`. Replace the final `if label is None:` block:

```python
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
```

And in `parse_log`, change the label fallback:

```python
    if label is None:
        label = _label_from_config(log_path.parent) or log_path.parent.name
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: derive run label from config.json"
```

---

## Task 4: Build the 4-subplot figure

**Files:**
- Modify: `scripts/plot_training.py`
- Modify: `tests/test_plot_training.py`

The `build_figure` function takes a list of data dicts (one per run) and returns a `matplotlib.figure.Figure`.

- [ ] **Step 1: Add a smoke test for figure creation**

Append to `tests/test_plot_training.py`:

```python
def test_build_figure_single_run(log_file):
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for tests
    from plot_training import build_figure
    data = parse_log(log_file)
    fig = build_figure([data])
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) == 5  # 4 subplots + 1 twinx on the LR/GradNorm subplot


def test_build_figure_comparison(log_file):
    import matplotlib
    matplotlib.use("Agg")
    from plot_training import build_figure
    data1 = parse_log(log_file, label="run_A")
    data2 = parse_log(log_file, label="run_B")
    fig = build_figure([data1, data2])
    assert fig is not None
    assert len(fig.get_axes()) == 5  # 4 subplots + 1 twinx on the LR/GradNorm subplot
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_plot_training.py::test_build_figure_single_run tests/test_plot_training.py::test_build_figure_comparison -v
```

Expected: FAIL — `build_figure` not defined.

- [ ] **Step 3: Add `build_figure` to `scripts/plot_training.py`**

Add after the `parse_log` function:

```python
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
        ax2_r = ax2.twinx() if i == 0 else axes[2].get_shared_x_axes()
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
        # Combine legends
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
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: add 4-subplot figure builder"
```

---

## Task 5: CLI entry point and PDF output

**Files:**
- Modify: `scripts/plot_training.py`

- [ ] **Step 1: Add `main()` and `__main__` block to `scripts/plot_training.py`**

Append to the end of `scripts/plot_training.py`:

```python
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
```

- [ ] **Step 2: Run a smoke test against a real log**

```bash
uv run scripts/plot_training.py runs/run_20260320_145519/
```

Expected: prints `Saved: runs/run_20260320_145519/training_plots.pdf`. Open the PDF to verify it looks correct.

- [ ] **Step 3: Run comparison mode smoke test**

```bash
uv run scripts/plot_training.py runs/run_20260318_161637/ runs/run_20260320_145519/
```

Expected: prints `Saved: runs/run_20260318_161637/comparison_plots.pdf`.

- [ ] **Step 4: Run all tests one final time**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_training.py
git commit -m "feat: add CLI entry point and PDF output for training plots"
```

---

## Self-Review

**Spec coverage:**
- ✅ Single run mode (`uv run scripts/plot_training.py runs/run/`)
- ✅ Comparison mode (multiple positional args)
- ✅ PDF output with default path logic
- ✅ `--output` override
- ✅ All 5 metrics parsed (loss, ppl, lr, grad, gap)
- ✅ 4 subplots: Train/Val Loss, Val PPL (log scale), LR+Grad (dual axis), Train-Val Gap
- ✅ Best val step vertical line (single run only)
- ✅ NeurIPS style (minimal spines, grey grid, tight layout)
- ✅ Run label from config.json with fallback to directory name
- ✅ `matplotlib` added to deps

**Placeholder scan:** No TBDs, all code blocks complete.

**Type consistency:** `parse_log` returns dict with keys used consistently in `build_figure`. `build_figure` takes `list[dict]`, returns `Figure`. `main()` calls both correctly.
