# Automatic Training Plots with Enriched Metrics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-generate `training_plots.pdf` at each checkpoint save, adding tokens/sec throughput, training perplexity, a dedicated gradient-norm subplot, and a consistent blue/orange train-vs-val colour scheme.

**Architecture:** Enrich `metrics.jsonl` with `grad_norm`, `tokens_per_sec`, `eval_train_loss`, and `eval_train_ppl` fields in `trainer.py`; add a structured `parse_metrics()` JSONL parser and a `plot_run_dir()` callable API to `plot_training.py`; redesign `build_figure()` from a 4×1 to a 3×2 grid; call `plot_run_dir()` from the trainer after every checkpoint save.

**Tech Stack:** `matplotlib`, `pathlib`, existing `JSONLLogger`, `CheckpointManager`.

---

## Approved Additional Metrics

| Metric | Status | Where computed |
|--------|--------|----------------|
| **Smoothed train loss (EMA, α=0.98)** | ✅ approved — implement in Task 3 | In plotter: overlay on loss and perplexity subplots |
| **Z-loss** | ❌ rejected — do not implement | n/a |

**Smoothed loss spec:** In `build_figure`, after plotting the raw `train_loss` line (dimmed to `alpha=0.25`, `linewidth=0.8`), compute and overlay an EMA-smoothed version at full opacity (`linewidth=1.5`). Apply the same treatment to `train_ppl` in the perplexity subplot. EMA formula: `s[0] = raw[0]`; `s[i] = 0.98 * s[i-1] + 0.02 * raw[i]`.

---

## File Map

| File | Change |
|------|--------|
| `src/scripts/plot_training.py` | Add `import math`; add `parse_metrics()`; add `plot_run_dir()`; new colour constants; redesign `build_figure()` (3×2 grid); update `main()` |
| `src/training/trainer.py` | Compute `tokens_per_step`; add per-step timer; enrich JSONL step/eval entries; call `plot_run_dir` after each checkpoint |
| `tests/test_plot_training.py` | Add `parse_metrics` tests; add `plot_run_dir` tests; update subplot count assertion |

---

## Task 1: Add `parse_metrics()` JSONL parser

**Files:**
- Modify: `src/scripts/plot_training.py`
- Test: `tests/test_plot_training.py`

- [ ] **Step 1: Write failing tests for `parse_metrics`**

Add these tests to `tests/test_plot_training.py`. Update the import line at the top to also import `parse_metrics`:

```python
# change line 8 from:
from plot_training import parse_log, _resolve_run
# to:
from plot_training import parse_log, parse_metrics, _resolve_run
```

Then add the following new fixtures and tests (place after the existing `log_file` fixture):

```python
import math

SAMPLE_JSONL = "\n".join([
    json.dumps({"stage": "pretraining", "type": "step", "step": 0, "epoch": 0,
                "train_loss": 10.8774, "perplexity": 52964.37, "lr": 2e-5,
                "grad_norm": 3.6559, "tokens_per_sec": 8192}),
    json.dumps({"stage": "pretraining", "type": "step", "step": 25, "epoch": 0,
                "train_loss": 9.7866, "perplexity": 17793.38, "lr": 5.2e-4,
                "grad_norm": 1.0129, "tokens_per_sec": 9500}),
    json.dumps({"stage": "pretraining", "type": "step", "step": 100, "epoch": 0,
                "train_loss": 7.0557, "perplexity": 1159.44, "lr": 1e-3,
                "grad_norm": 0.8274, "tokens_per_sec": 10100}),
    json.dumps({"stage": "pretraining", "type": "eval", "step": 100, "epoch": 0,
                "val_loss": 7.1702, "val_ppl": 1300.12,
                "eval_train_loss": 7.0557, "eval_train_ppl": 1159.44}),
    json.dumps({"stage": "pretraining", "type": "best_checkpoint", "step": 100,
                "epoch": 0, "val_loss": 7.1702}),
])


@pytest.fixture
def metrics_dir(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(SAMPLE_JSONL)
    return tmp_path


def test_parse_metrics_steps(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["steps"] == [0, 25, 100]


def test_parse_metrics_train_loss(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["train_loss"] == pytest.approx([10.8774, 9.7866, 7.0557])


def test_parse_metrics_lr(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["lr"] == pytest.approx([2e-5, 5.2e-4, 1e-3])


def test_parse_metrics_grad_norm(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["grad_norm"] == pytest.approx([3.6559, 1.0129, 0.8274])


def test_parse_metrics_tokens_per_sec(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["tokens_per_sec"] == pytest.approx([8192, 9500, 10100])


def test_parse_metrics_train_ppl(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["train_ppl"] == pytest.approx([52964.37, 17793.38, 1159.44])


def test_parse_metrics_eval(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["eval_steps"] == [100]
    assert data["val_loss"] == pytest.approx([7.1702])
    assert data["val_ppl"] == pytest.approx([1300.12])
    assert data["eval_train_loss"] == pytest.approx([7.0557])
    assert data["eval_train_ppl"] == pytest.approx([1159.44])


def test_parse_metrics_best_val_step(metrics_dir):
    data = parse_metrics(metrics_dir)
    assert data["best_val_step"] == 100


def test_parse_metrics_label_fallback(metrics_dir):
    data = parse_metrics(metrics_dir, label="my_run")
    assert data["label"] == "my_run"


def test_parse_metrics_missing_optional_fields(tmp_path):
    """Old-format JSONL entries without grad_norm/tokens_per_sec parse as nan."""
    old_entry = json.dumps({"stage": "pretraining", "type": "step", "step": 10,
                            "epoch": 0, "train_loss": 5.0, "perplexity": 148.4,
                            "lr": 1e-3})
    (tmp_path / "metrics.jsonl").write_text(old_entry + "\n")
    data = parse_metrics(tmp_path)
    assert math.isnan(data["grad_norm"][0])
    assert math.isnan(data["tokens_per_sec"][0])


def test_parse_metrics_eval_without_train_fields(tmp_path):
    """Old eval entries without eval_train_loss/ppl still parse cleanly."""
    old_eval = json.dumps({"stage": "pretraining", "type": "eval", "step": 50,
                           "epoch": 0, "val_loss": 3.1, "val_ppl": 22.2})
    (tmp_path / "metrics.jsonl").write_text(old_eval + "\n")
    data = parse_metrics(tmp_path)
    assert data["eval_steps"] == [50]
    assert data["eval_train_loss"] == [None]
    assert data["eval_train_ppl"] == [None]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_plot_training.py -k "parse_metrics" -v
```

Expected: `ImportError` or `AttributeError` — `parse_metrics` does not exist yet.

- [ ] **Step 3: Implement `parse_metrics()` in `plot_training.py`**

First, add `import math` to the imports section at the top of `src/scripts/plot_training.py` (after `import re`):

```python
import math
```

Then add `parse_metrics` as a new function directly after `parse_log` (before the `# Plotting` section):

```python
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
```

Also update the `return` statement in the existing `parse_log()` function to include `tokens_per_sec` (so `build_figure` receives a uniform dict from both parsers):

```python
# In parse_log(), change the return statement to add this key:
return {
    "steps": steps,
    "train_loss": train_loss,
    "lr": lr,
    "grad_norm": grad_norm,
    "train_ppl": train_ppl,
    "tokens_per_sec": [math.nan] * len(steps),   # <-- add this line
    "eval_steps": eval_steps,
    "val_loss": val_loss,
    "val_ppl": val_ppl,
    "eval_train_loss": eval_train_loss,
    "eval_train_ppl": eval_train_ppl,
    "best_val_step": best_val_step,
    "label": label,
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_plot_training.py -k "parse_metrics" -v
```

Expected: all 11 `parse_metrics` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: add parse_metrics() JSONL parser to plot_training.py"
```

---

## Task 2: Enrich trainer JSONL logs

**Files:**
- Modify: `src/training/trainer.py`

- [ ] **Step 1: Add `tokens_per_step` computation and per-step timer before the training loop**

In `src/training/trainer.py`, find the two lines around line 1040–1046:

```python
grad_accum = tc["gradient_accumulation_steps"]
...
train_start = time.time()
t0 = train_start
```

Add directly after `grad_accum = ...`:

```python
grad_accum = tc["gradient_accumulation_steps"]
tokens_per_step: int = tc["batch_size"] * mc["context_length"] * grad_accum
```

And add directly after `t0 = train_start`:

```python
t0 = train_start
_step_t = train_start
```

- [ ] **Step 2: Capture per-step elapsed time after the optimizer update**

Find line ~1153:

```python
optimizer_updated = _apply_optimizer_step(optimizer, scaler)
if optimizer_updated:
    scheduler.step()
```

Change to:

```python
optimizer_updated = _apply_optimizer_step(optimizer, scaler)
_step_dt = time.time() - _step_t
_step_t = time.time()
if optimizer_updated:
    scheduler.step()
```

- [ ] **Step 3: Enrich the JSONL step log**

Find the existing `jlog.log("pretraining", "step", ...)` block (around lines 1195–1200):

```python
if jlog is not None:
    jlog.log(
        "pretraining", "step",
        epoch=current_epoch, step=completed_steps,
        train_loss=accum_loss, perplexity=ppl, lr=lr,
    )
```

Replace with:

```python
if jlog is not None:
    jlog.log(
        "pretraining", "step",
        epoch=current_epoch, step=completed_steps,
        train_loss=accum_loss, perplexity=ppl, lr=lr,
        grad_norm=round(grad_norm, 6),
        tokens_per_sec=round(tokens_per_step / _step_dt) if _step_dt > 0 else 0,
    )
```

- [ ] **Step 4: Enrich the JSONL eval log**

Find the existing `jlog.log("pretraining", "eval", ...)` block (around lines 1233–1238):

```python
if jlog is not None:
    jlog.log(
        "pretraining", "eval",
        step=completed_steps, epoch=eval_epoch,
        val_loss=val_loss, val_ppl=val_ppl,
    )
```

Replace with:

```python
if jlog is not None:
    jlog.log(
        "pretraining", "eval",
        step=completed_steps, epoch=eval_epoch,
        val_loss=val_loss, val_ppl=val_ppl,
        eval_train_loss=accum_loss, eval_train_ppl=ppl,
    )
```

- [ ] **Step 5: Run trainer tests to confirm nothing broken**

```bash
uv run pytest tests/training/ -v
```

Expected: all tests PASS (added fields are additive; no logic changed).

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: enrich JSONL with grad_norm, tokens_per_sec, eval_train_loss/ppl"
```

---

## Task 3: Redesign `build_figure()` — new colours and 3×2 layout

**Files:**
- Modify: `src/scripts/plot_training.py`
- Test: `tests/test_plot_training.py`

- [ ] **Step 1: Update subplot-count assertions to their new expected values**

In `tests/test_plot_training.py`, change both existing `build_figure` tests:

```python
def test_build_figure_single_run(log_file):
    import matplotlib
    matplotlib.use("Agg")
    from plot_training import build_figure
    data = parse_log(log_file)
    fig = build_figure([data])
    assert fig is not None
    assert len(fig.get_axes()) == 6   # 6 independent subplots, no twinx


def test_build_figure_comparison(log_file):
    import matplotlib
    matplotlib.use("Agg")
    from plot_training import build_figure
    data1 = parse_log(log_file, label="run_A")
    data2 = parse_log(log_file, label="run_B")
    fig = build_figure([data1, data2])
    assert fig is not None
    assert len(fig.get_axes()) == 6
```

- [ ] **Step 2: Run to confirm the tests fail**

```bash
uv run pytest tests/test_plot_training.py -k "build_figure" -v
```

Expected: FAIL — `assert 5 == 6` (old layout had 4 subplots + 1 twinx = 5 axes).

- [ ] **Step 3: Replace colour constants and `build_figure()` in `plot_training.py`**

Replace the block from `# Qualitative color palette` down to (and including) the closing `return fig` of the current `build_figure` function with:

```python
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


def build_figure(runs: list[dict]) -> plt.Figure:
    """
    Build a 3×2 subplot figure from one or more parsed run dicts.

    Layout:
      [0,0] Train & Val Loss          [0,1] Train & Val Perplexity (log)
      [1,0] Learning Rate             [1,1] Gradient Norm
      [2,0] Training Throughput       [2,1] Train–Val Gap
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=False)
    is_comparison = len(runs) > 1

    for i, data in enumerate(runs):
        # comparison: one palette colour per run, solid=train / dashed=val
        # single run: TRAIN_COLOR for training metrics, VAL_COLOR for validation
        run_col = _PALETTE[i % len(_PALETTE)] if is_comparison else None
        tc = run_col if is_comparison else TRAIN_COLOR
        vc = run_col if is_comparison else VAL_COLOR
        label = data["label"] if is_comparison else None
        steps = data["steps"]
        eval_steps = data["eval_steps"]

        # ── [0,0] Train & Val Loss ────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(steps, data["train_loss"], color=tc, linewidth=1.5,
                linestyle="-",
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
        if data["train_ppl"]:
            ax.plot(steps, data["train_ppl"], color=tc, linewidth=1.5,
                    linestyle="-",
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
```

- [ ] **Step 4: Run figure tests to confirm pass**

```bash
uv run pytest tests/test_plot_training.py -k "build_figure" -v
```

Expected: both PASS.

- [ ] **Step 5: Run full plot test suite**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all tests PASS (regex-based tests are untouched; `_resolve_run` test unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: redesign build_figure() — 3x2 grid, train=blue/val=orange, train perplexity, grad-norm subplot"
```

---

## Task 4: Add `plot_run_dir()` API function and update CLI

**Files:**
- Modify: `src/scripts/plot_training.py`
- Test: `tests/test_plot_training.py`

- [ ] **Step 1: Write failing tests for `plot_run_dir`**

Update the import at the top of `tests/test_plot_training.py`:

```python
from plot_training import parse_log, parse_metrics, plot_run_dir, _resolve_run
```

Add these tests:

```python
def test_plot_run_dir_reads_metrics_jsonl(metrics_dir):
    """plot_run_dir() reads metrics.jsonl and saves a PDF at the default path."""
    import matplotlib
    matplotlib.use("Agg")
    out = plot_run_dir(metrics_dir)
    assert out == metrics_dir / "training_plots.pdf"
    assert out.exists()


def test_plot_run_dir_custom_output(metrics_dir, tmp_path):
    """plot_run_dir() respects a custom output path."""
    import matplotlib
    matplotlib.use("Agg")
    out_path = tmp_path / "custom.pdf"
    out = plot_run_dir(metrics_dir, output=out_path)
    assert out == out_path
    assert out.exists()


def test_plot_run_dir_falls_back_to_train_log(log_file):
    """plot_run_dir() uses train.log when metrics.jsonl is absent."""
    import matplotlib
    matplotlib.use("Agg")
    out = plot_run_dir(log_file.parent)
    assert out.exists()
    assert out.suffix == ".pdf"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_plot_training.py -k "plot_run_dir" -v
```

Expected: `ImportError` — `plot_run_dir` does not exist yet.

- [ ] **Step 3: Implement `plot_run_dir()` and update `main()` in `plot_training.py`**

Add `plot_run_dir` right before the `# CLI` section (after `build_figure`):

```python
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
```

Update `main()` to use `plot_run_dir` for single-run invocations and prefer `metrics.jsonl` for multi-run comparisons:

```python
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
```

- [ ] **Step 4: Run `plot_run_dir` tests**

```bash
uv run pytest tests/test_plot_training.py -k "plot_run_dir" -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Run full plot test suite**

```bash
uv run pytest tests/test_plot_training.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/plot_training.py tests/test_plot_training.py
git commit -m "feat: add plot_run_dir() API; update CLI to prefer metrics.jsonl"
```

---

## Task 5: Auto-generate plots at each checkpoint save

**Files:**
- Modify: `src/training/trainer.py`

- [ ] **Step 1: Add auto-plot call after `save_last` checkpoint**

Find the block around line 1345 that saves the `last` checkpoint (inside the `if is_master` block):

```python
if ckpt_path is not None:
    log.info(f"Saved checkpoint: {ckpt_path}")
    if jlog is not None:
        jlog.log(
            "pretraining", "checkpoint",
            step=completed_steps,
            epoch=eval_epoch,
            path=ckpt_path,
            category="last",
        )
```

Extend it to:

```python
if ckpt_path is not None:
    log.info(f"Saved checkpoint: {ckpt_path}")
    if jlog is not None:
        jlog.log(
            "pretraining", "checkpoint",
            step=completed_steps,
            epoch=eval_epoch,
            path=ckpt_path,
            category="last",
        )
    try:
        from pathlib import Path as _Path
        from src.scripts.plot_training import plot_run_dir as _plot_run_dir
        _plot_run_dir(_Path(run_dir))
    except Exception as _exc:
        log.debug("Auto-plotting skipped: %s", _exc)
```

- [ ] **Step 2: Add auto-plot call after `maybe_save_best` checkpoint**

Find the block around line 1265 that handles the best checkpoint (inside `if is_master:`):

```python
if best_candidate_path is not None:
    log.info(f"  Saved best checkpoint: {best_candidate_path}")
    if jlog is not None:
        jlog.log(
            "pretraining",
            "best_checkpoint",
            step=completed_steps,
            epoch=eval_epoch,
            path=best_candidate_path,
            val_loss=val_loss,
        )
```

Extend it to:

```python
if best_candidate_path is not None:
    log.info(f"  Saved best checkpoint: {best_candidate_path}")
    if jlog is not None:
        jlog.log(
            "pretraining",
            "best_checkpoint",
            step=completed_steps,
            epoch=eval_epoch,
            path=best_candidate_path,
            val_loss=val_loss,
        )
    try:
        from pathlib import Path as _Path
        from src.scripts.plot_training import plot_run_dir as _plot_run_dir
        _plot_run_dir(_Path(run_dir))
    except Exception as _exc:
        log.debug("Auto-plotting skipped: %s", _exc)
```

- [ ] **Step 3: Run trainer tests**

```bash
uv run pytest tests/training/ -v
```

Expected: all tests PASS. The try/except ensures plotting failures are silent.

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: auto-generate training_plots.pdf at each checkpoint save"
```

---

## Self-Review

**Spec coverage:**
- ✅ Automatic plots at checkpoints — Task 5
- ✅ Tokens/second throughput metric — Task 2 (logged) + Task 3 subplot [2,0]
- ✅ Training perplexity on plot — Task 3 subplot [0,1], alongside val perplexity
- ✅ Blue/orange train-vs-val colour split — Task 3 (`TRAIN_COLOR`, `VAL_COLOR`)
- ✅ Backward compatibility with old `train.log` runs — `plot_run_dir` fallback

**Suggested metrics excluded:** z-loss and smoothed loss pending approval.

**No placeholders** — all steps contain complete code.

**Type/key consistency:**
- Both `parse_log` and `parse_metrics` return the same dict keys including `tokens_per_sec`.
- `build_figure` accesses `data.get("tokens_per_sec", [])` safely for both sources.
- `grad_norm` in `build_figure` filters `math.isnan` to handle `[nan, nan, ...]` from `parse_log`.
