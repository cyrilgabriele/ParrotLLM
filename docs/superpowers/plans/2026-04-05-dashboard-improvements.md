# Dashboard Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Gradio Live Monitor with a top-level status banner, richer progress section with historical deltas, reordered/larger plots, ETA in h/m/s, and a better Architecture tab showing model params and training config.

**Architecture:** All changes are confined to `src/dashboard/app.py` (helper functions + UI layout) and `src/dashboard/plots.py` (panel order and figure size). No new files, no new dependencies.

**Tech Stack:** Gradio 6, matplotlib (already installed).

**Spec:** `docs/superpowers/specs/2026-04-05-dashboard-improvements.md`

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/dashboard/app.py` | Modify | New `_fmt_status_banner()`, updated `_compute_eta()`, updated `_fmt_progress_detail()`, updated `_arch_and_config_text()`, rebuilt Live Monitor layout, updated Architecture tab |
| `src/dashboard/plots.py` | Modify | Reorder panels (Loss · Tok/s · LR+Grad on top), increase figure size |
| `tests/dashboard/test_app_helpers.py` | Create | Unit tests for the four helper functions |

---

## Task 1: ETA helper — add seconds

`_compute_eta` currently returns `~2h 14m` or `~14m`. Change to always include seconds: `~2h 14m 32s` or `~14m 32s` or `~45s`.

**Files:**
- Modify: `src/dashboard/app.py`
- Create: `tests/dashboard/test_app_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_app_helpers.py`:

```python
# tests/dashboard/test_app_helpers.py
import pytest
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.app import _compute_eta


def _metrics_with_tps(steps, tps, max_steps=1000, batch=64, ctx=1024, grad_acc=1):
    m = TrainingMetrics()
    m.steps = steps
    m.tokens_per_sec = tps
    m.train_losses = [4.0] * len(steps)
    m.lrs = [1e-4] * len(steps)
    m.grad_norms = [0.5] * len(steps)
    m.config = {
        "max_steps": max_steps,
        "batch_size": batch,
        "context_length": ctx,
        "gradient_accumulation_steps": grad_acc,
    }
    return m


def test_eta_includes_seconds_hours():
    # 900 steps remaining × 1024 tokens/step ÷ 1024 tok/s = 900s = 0h 15m 0s
    m = _metrics_with_tps([100], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    assert "m" in eta and "s" in eta


def test_eta_large_remaining_shows_hours():
    # 7200s remaining → ~2h 0m 0s
    m = _metrics_with_tps([100], [1024.0], max_steps=7300, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    assert "h" in eta


def test_eta_seconds_only_when_under_a_minute():
    # 30s remaining
    m = _metrics_with_tps([970], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    eta = _compute_eta(m)
    # Should not show hours or minutes prefix (just seconds)
    assert "s" in eta


def test_eta_done_when_no_remaining():
    m = _metrics_with_tps([1000], [1024.0], max_steps=1000, batch=1, ctx=1024, grad_acc=1)
    assert _compute_eta(m) == "Done"


def test_eta_dash_when_no_tps():
    m = TrainingMetrics()
    m.steps = [100]
    m.train_losses = [4.0]
    m.lrs = [1e-4]
    m.grad_norms = [0.5]
    m.config = {"max_steps": 1000}
    assert _compute_eta(m) == "—"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_app_helpers.py -v
```

Expected: `ImportError` or test failures — `_compute_eta` currently doesn't include seconds.

- [ ] **Step 3: Update `_compute_eta` in `src/dashboard/app.py`**

Replace the existing `_compute_eta` function (lines ~65-84) with:

```python
def _compute_eta(metrics: TrainingMetrics) -> str:
    if not metrics.steps or not metrics.tokens_per_sec:
        return "—"
    max_steps = metrics.config.get("max_steps")
    if not max_steps:
        return "—"
    remaining = max_steps - metrics.steps[-1]
    if remaining <= 0:
        return "Done"
    tokens_per_step = (
        metrics.config.get("batch_size", 64)
        * metrics.config.get("context_length", 1024)
        * metrics.config.get("gradient_accumulation_steps", 4)
    )
    avg_tps = sum(metrics.tokens_per_sec[-10:]) / len(metrics.tokens_per_sec[-10:])
    if avg_tps <= 0:
        return "—"
    eta_sec = int(remaining * tokens_per_step / avg_tps)
    h = eta_sec // 3600
    m = (eta_sec % 3600) // 60
    s = eta_sec % 60
    if h > 0:
        return f"~{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"~{m}m {s:02d}s"
    return f"~{s}s"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_eta_includes_seconds_hours tests/dashboard/test_app_helpers.py::test_eta_large_remaining_shows_hours tests/dashboard/test_app_helpers.py::test_eta_seconds_only_when_under_a_minute tests/dashboard/test_app_helpers.py::test_eta_done_when_no_remaining tests/dashboard/test_app_helpers.py::test_eta_dash_when_no_tps -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py tests/dashboard/test_app_helpers.py
git commit -m "feat(dashboard): ETA now shows h/m/s (e.g. ~2h 14m 32s)"
```

---

## Task 2: Status banner helper

Add `_fmt_status_banner(metrics)` that returns an HTML string: green when no alerts, red/yellow when problems.

**Files:**
- Modify: `src/dashboard/app.py`
- Modify: `tests/dashboard/test_app_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app_helpers.py`:

```python
from src.dashboard.app import _fmt_status_banner


def _metrics_with_grad_explosion():
    m = TrainingMetrics()
    m.steps = [100, 200, 300]
    m.train_losses = [4.5, 4.1, 3.8]
    m.lrs = [3e-4] * 3
    m.grad_norms = [0.5, 15.0, 12.0]
    return m


def test_status_banner_green_when_no_alerts():
    m = TrainingMetrics()
    html = _fmt_status_banner(m)
    assert "No errors" in html or "No problems" in html or "✅" in html
    assert "green" in html or "#" in html  # has a colour style


def test_status_banner_shows_error_code():
    m = _metrics_with_grad_explosion()
    html = _fmt_status_banner(m)
    assert "GRAD_EXPLOSION" in html


def test_status_banner_red_on_error():
    m = _metrics_with_grad_explosion()
    html = _fmt_status_banner(m)
    assert "🔴" in html or "red" in html.lower()
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_status_banner_green_when_no_alerts tests/dashboard/test_app_helpers.py::test_status_banner_shows_error_code tests/dashboard/test_app_helpers.py::test_status_banner_red_on_error -v
```

Expected: `ImportError` — `_fmt_status_banner` doesn't exist yet.

- [ ] **Step 3: Add `_fmt_status_banner` to `src/dashboard/app.py`**

Add this function after `_alert_rows`:

```python
def _fmt_status_banner(metrics: TrainingMetrics) -> str:
    """Return an HTML status banner: green when clean, red/yellow when alerts exist."""
    alerts = detect_problems(metrics)
    if not alerts:
        return (
            "<div style='background:#d1fae5;border:1px solid #6ee7b7;border-radius:6px;"
            "padding:10px 16px;font-size:15px;font-weight:600;color:#065f46'>"
            "✅  No errors or malfunctions detected"
            "</div>"
        )
    lines = []
    for a in alerts:
        emoji = _SEVERITY_EMOJI[a.severity]
        bg = "#fee2e2" if a.severity == Severity.ERROR else "#fef3c7"
        border = "#fca5a5" if a.severity == Severity.ERROR else "#fcd34d"
        color = "#7f1d1d" if a.severity == Severity.ERROR else "#78350f"
        lines.append(
            f"<div style='background:{bg};border:1px solid {border};border-radius:6px;"
            f"padding:8px 16px;font-size:14px;font-weight:600;color:{color};margin-bottom:4px'>"
            f"{emoji}  {a.code} — {a.message}: {a.detail}"
            "</div>"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_status_banner_green_when_no_alerts tests/dashboard/test_app_helpers.py::test_status_banner_shows_error_code tests/dashboard/test_app_helpers.py::test_status_banner_red_on_error -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py tests/dashboard/test_app_helpers.py
git commit -m "feat(dashboard): add _fmt_status_banner — green/red HTML banner for alert state"
```

---

## Task 3: Progress helper — deltas and larger display

Replace `_fmt_progress` with `_fmt_progress_detail` that returns a multi-line HTML string including historical deltas (loss change over last N steps) and a text progress bar.

**Files:**
- Modify: `src/dashboard/app.py`
- Modify: `tests/dashboard/test_app_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app_helpers.py`:

```python
from src.dashboard.app import _fmt_progress_detail


def _metrics_with_history():
    m = TrainingMetrics()
    m.steps = list(range(100, 600, 100))      # [100,200,300,400,500]
    m.train_losses = [4.5, 4.3, 4.1, 3.9, 3.7]
    m.lrs = [3e-4] * 5
    m.grad_norms = [0.8, 0.7, 0.65, 0.6, 0.55]
    m.tokens_per_sec = [12000.0] * 5
    m.eval_steps = [200, 400]
    m.val_losses = [4.4, 3.95]
    m.best_step = 400
    m.config = {"max_steps": 1000, "batch_size": 64, "context_length": 1024,
                "gradient_accumulation_steps": 4}
    return m


def test_progress_detail_shows_step():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "500" in html


def test_progress_detail_shows_loss_delta():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    # Loss went from 4.5 to 3.7, delta = -0.8 over 5 steps
    assert "Δ" in html or "delta" in html.lower() or "-0." in html


def test_progress_detail_shows_progress_bar():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    # Should have a visual progress bar (50% complete)
    assert "50" in html or "50.0" in html


def test_progress_detail_shows_val_loss():
    m = _metrics_with_history()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "3.95" in html or "3.9500" in html


def test_progress_detail_empty_state():
    m = TrainingMetrics()
    html = _fmt_progress_detail(m, m_run_dir=None)
    assert "No runs found" in html
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_progress_detail_shows_step tests/dashboard/test_app_helpers.py::test_progress_detail_shows_loss_delta tests/dashboard/test_app_helpers.py::test_progress_detail_shows_progress_bar tests/dashboard/test_app_helpers.py::test_progress_detail_shows_val_loss tests/dashboard/test_app_helpers.py::test_progress_detail_empty_state -v
```

Expected: `ImportError` — function doesn't exist yet.

- [ ] **Step 3: Add `_fmt_progress_detail` to `src/dashboard/app.py`**

Add after `_fmt_status_banner`:

```python
def _fmt_progress_detail(metrics: TrainingMetrics, m_run_dir) -> str:
    """Return a rich HTML progress section with current values, deltas, and a progress bar."""
    if not metrics.steps:
        return "<p style='color:#6b7280'>No runs found in runs/. Start training first.</p>"

    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    max_steps = metrics.config.get("max_steps")

    # ── Progress bar ──────────────────────────────────────────────────
    if max_steps:
        pct = 100.0 * step / max_steps
        bar = (
            f"<div style='background:#e5e7eb;border-radius:4px;height:12px;margin:6px 0'>"
            f"<div style='background:#2563EB;width:{pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div>"
            f"<div style='font-size:13px;color:#6b7280'>Step {step:,} / {max_steps:,} &nbsp;·&nbsp; {pct:.1f}%</div>"
        )
    else:
        bar = f"<div style='font-size:13px;color:#6b7280'>Step {step:,}</div>"

    # ── Loss delta over all recorded steps ───────────────────────────
    if len(metrics.train_losses) >= 2:
        delta = metrics.train_losses[-1] - metrics.train_losses[0]
        delta_str = f"{delta:+.4f}"
        delta_color = "#16a34a" if delta < 0 else "#dc2626"
        loss_html = (
            f"<b>Train Loss</b> {loss:.4f} "
            f"<span style='color:{delta_color};font-size:12px'>(Δ {delta_str} total)</span>"
        )
    else:
        loss_html = f"<b>Train Loss</b> {loss:.4f}"

    # ── Val loss ──────────────────────────────────────────────────────
    val_html = ""
    if metrics.val_losses:
        val = metrics.val_losses[-1]
        if len(metrics.val_losses) >= 2:
            val_delta = metrics.val_losses[-1] - metrics.val_losses[0]
            vd_color = "#16a34a" if val_delta < 0 else "#dc2626"
            val_html = (
                f"&nbsp;·&nbsp; <b>Val Loss</b> {val:.4f} "
                f"<span style='color:{vd_color};font-size:12px'>(Δ {val_delta:+.4f} total)</span>"
            )
        else:
            val_html = f"&nbsp;·&nbsp; <b>Val Loss</b> {val:.4f}"

    # ── Secondary metrics ─────────────────────���───────────────────────
    secondary = [f"<b>LR</b> {lr:.2e}"]
    if metrics.grad_norms:
        secondary.append(f"<b>Grad Norm</b> {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        secondary.append(f"<b>Tok/s</b> {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        secondary.append(f"<b>Best Step</b> {metrics.best_step:,}")

    # ── Stale warning ─────────────────────────────────────────────────
    stale_html = ""
    if m_run_dir is not None:
        stale, age = is_metrics_stale(m_run_dir)
        if stale:
            stale_html = (
                f"<div style='color:#d97706;margin-top:6px;font-size:13px'>"
                f"⚠ Metrics not updated for {age}s — training may have stalled or crashed."
                f"</div>"
            )

    return (
        f"{bar}"
        f"<div style='margin-top:8px;font-size:15px'>{loss_html}{val_html}</div>"
        f"<div style='margin-top:4px;font-size:13px;color:#4b5563'>"
        + "&nbsp;·&nbsp;".join(secondary) +
        f"</div>{stale_html}"
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_progress_detail_shows_step tests/dashboard/test_app_helpers.py::test_progress_detail_shows_loss_delta tests/dashboard/test_app_helpers.py::test_progress_detail_shows_progress_bar tests/dashboard/test_app_helpers.py::test_progress_detail_shows_val_loss tests/dashboard/test_app_helpers.py::test_progress_detail_empty_state -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py tests/dashboard/test_app_helpers.py
git commit -m "feat(dashboard): add _fmt_progress_detail — progress bar, loss deltas, richer HTML"
```

---

## Task 4: Architecture + config helper

Replace `_arch_text` with `_arch_and_config_text` that renders model params first (matching training startup output) followed by a training config section.

**Files:**
- Modify: `src/dashboard/app.py`
- Modify: `tests/dashboard/test_app_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_app_helpers.py`:

```python
from src.dashboard.app import _arch_and_config_text


def _metrics_with_arch_and_config():
    m = TrainingMetrics()
    m.architecture = {
        "vocab_size": 50257, "n_layers": 16, "n_heads": 8,
        "d_model": 320, "d_ff": 854,
        "total_params": 35763840, "trainable_params": 35763840,
    }
    m.config = {
        "max_steps": 10000, "batch_size": 64, "context_length": 1024,
        "gradient_accumulation_steps": 4, "learning_rate": 3e-4,
    }
    return m


def test_arch_text_params_first():
    m = _metrics_with_arch_and_config()
    text = _arch_and_config_text(m)
    params_pos = text.index("35,763,840")
    layers_pos = text.index("16")
    assert params_pos < layers_pos  # total params appears before n_layers


def test_arch_text_includes_config():
    m = _metrics_with_arch_and_config()
    text = _arch_and_config_text(m)
    assert "10,000" in text or "10000" in text  # max_steps
    assert "64" in text   # batch_size
    assert "3e-04" in text or "3.00e-04" in text or "0.0003" in text  # lr


def test_arch_text_no_data():
    m = TrainingMetrics()
    text = _arch_and_config_text(m)
    assert "No architecture" in text or "No data" in text
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_arch_text_params_first tests/dashboard/test_app_helpers.py::test_arch_text_includes_config tests/dashboard/test_app_helpers.py::test_arch_text_no_data -v
```

Expected: `ImportError` — function doesn't exist yet.

- [ ] **Step 3: Add `_arch_and_config_text` to `src/dashboard/app.py`**

Replace the existing `_arch_text` function with:

```python
def _arch_and_config_text(metrics: TrainingMetrics) -> str:
    """Return a formatted string matching training startup output, params-first."""
    arch = metrics.architecture
    cfg = metrics.config

    if not arch and not cfg:
        return "No architecture or config data available. Load a run first."

    lines = []

    if arch:
        lines += [
            "── Model Architecture ──────────────────────────────",
            f"  Total params:     {arch.get('total_params', '?'):,}" if isinstance(arch.get('total_params'), int) else f"  Total params:     {arch.get('total_params', '?')}",
            f"  Trainable params: {arch.get('trainable_params', '?'):,}" if isinstance(arch.get('trainable_params'), int) else f"  Trainable params: {arch.get('trainable_params', '?')}",
            f"  Vocab size:       {arch.get('vocab_size', '?'):,}" if isinstance(arch.get('vocab_size'), int) else f"  Vocab size:       {arch.get('vocab_size', '?')}",
            f"  Layers:           {arch.get('n_layers', '?')}",
            f"  Attention heads:  {arch.get('n_heads', '?')}",
            f"  d_model:          {arch.get('d_model', '?')}",
            f"  FFN dim (d_ff):   {arch.get('d_ff', '?')}",
            f"  Context length:   {arch.get('context_length', cfg.get('context_length', '?'))}",
        ]

    if cfg:
        if lines:
            lines.append("")
        lines += [
            "── Training Config ─────────────────────────────────",
            f"  Max steps:        {cfg.get('max_steps', '?'):,}" if isinstance(cfg.get('max_steps'), int) else f"  Max steps:        {cfg.get('max_steps', '?')}",
            f"  Batch size:       {cfg.get('batch_size', '?')}",
            f"  Context length:   {cfg.get('context_length', '?')}",
            f"  Grad accumulation:{cfg.get('gradient_accumulation_steps', '?')}",
        ]
        if "learning_rate" in cfg:
            lines.append(f"  Learning rate:    {cfg['learning_rate']:.2e}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/dashboard/test_app_helpers.py::test_arch_text_params_first tests/dashboard/test_app_helpers.py::test_arch_text_includes_config tests/dashboard/test_app_helpers.py::test_arch_text_no_data -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py tests/dashboard/test_app_helpers.py
git commit -m "feat(dashboard): add _arch_and_config_text — params-first layout with training config"
```

---

## Task 5: Reorder and resize plots

Change the 2×3 figure in `src/dashboard/plots.py`:
- **Top row** (most active): Train & Val Loss · Tokens/sec · LR & Grad Norm
- **Bottom row** (contextual): Validation Perplexity · Generalization Gap · (hidden)
- Increase figure size from `(18, 9)` to `(20, 10)`.

**Files:**
- Modify: `src/dashboard/plots.py`

No new unit test — existing smoke test validates render. Run it after.

- [ ] **Step 1: Reorder panels in `src/dashboard/plots.py`**

Replace the full `build_training_figure` function:

```python
def build_training_figure(metrics: TrainingMetrics) -> Optional[plt.Figure]:
    """Return a 2×3 Figure from TrainingMetrics, or None if no data.

    Top row (most active): Train & Val Loss | Tokens/sec | LR & Grad Norm
    Bottom row (contextual): Val Perplexity | Generalization Gap | (hidden)
    """
    if not metrics.steps:
        return None

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    steps = metrics.steps
    eval_steps = metrics.eval_steps
    n_eval = len(eval_steps)

    # ── [0,0] Train & Val Loss (most important) ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, metrics.train_losses, color=TRAIN_COLOR, linewidth=1.8, label="Train")
    if eval_steps and metrics.val_losses:
        ax1.plot(eval_steps, metrics.val_losses[:n_eval], color=VAL_COLOR, linewidth=1.8,
                 linestyle="--", label="Val")
    if metrics.best_step:
        ax1.axvline(metrics.best_step, color="#999", linewidth=0.8, linestyle=":",
                    label=f"Best @ {metrics.best_step}")
    _style(ax1)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train & Validation Loss", fontweight="bold")
    ax1.legend(fontsize=8, frameon=False)

    # ── [0,1] Tokens per second ───────────────────��───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics.tokens_per_sec:
        ax2.plot(steps[:len(metrics.tokens_per_sec)], metrics.tokens_per_sec,
                 color=TRAIN_COLOR, linewidth=1.8)
    _style(ax2)
    ax2.set_ylabel("Tok/s")
    ax2.set_title("Tokens per Second", fontweight="bold")

    # ── [0,2] LR + Grad Norm (twin axis) ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, metrics.lrs, color=LR_COLOR, linewidth=1.8, label="LR")
    ax3.set_ylabel("Learning Rate", color=LR_COLOR)
    ax3.tick_params(axis="y", labelcolor=LR_COLOR)
    ax3.set_xlabel("Step")
    ax3.set_title("LR & Grad Norm", fontweight="bold")
    _style(ax3)
    if metrics.grad_norms:
        ax3b = ax3.twinx()
        ax3b.plot(steps, metrics.grad_norms, color=GRAD_COLOR, linewidth=1.2,
                  alpha=0.7, label="Grad Norm")
        ax3b.set_ylabel("Grad Norm", color=GRAD_COLOR)
        ax3b.tick_params(axis="y", labelcolor=GRAD_COLOR)
        ax3b.spines["top"].set_visible(False)

    # ── [1,0] Validation Perplexity ───────────────────────��───────────
    ax4 = fig.add_subplot(gs[1, 0])
    if eval_steps and metrics.val_ppls:
        ax4.plot(eval_steps, metrics.val_ppls[:n_eval], color=VAL_COLOR, linewidth=1.8)
        ax4.set_yscale("log")
    _style(ax4)
    ax4.set_ylabel("Perplexity (log)")
    ax4.set_xlabel("Step")
    ax4.set_title("Validation Perplexity")

    # ── [1,1] Generalization Gap ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if eval_steps and metrics.val_losses and metrics.eval_train_losses:
        gap = [v - t for v, t in zip(metrics.val_losses[:n_eval],
                                      metrics.eval_train_losses[:n_eval])]
        ax5.plot(eval_steps[:len(gap)], gap, color=VAL_COLOR, linewidth=1.8)
        ax5.axhline(0, color="#bbb", linewidth=0.8, linestyle="--")
    _style(ax5)
    ax5.set_ylabel("Val − Train Loss")
    ax5.set_xlabel("Step")
    ax5.set_title("Generalization Gap")

    # ── [1,2] hidden ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_visible(False)

    fig.tight_layout(pad=1.5)
    return fig
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -c "
from src.dashboard.metrics_reader import TrainingMetrics
from src.dashboard.plots import build_training_figure
m = TrainingMetrics()
m.steps = [100, 200, 300]
m.train_losses = [4.5, 4.1, 3.8]
m.lrs = [1e-4, 1e-4, 1e-4]
m.grad_norms = [0.8, 0.7, 0.6]
m.tokens_per_sec = [12000.0, 12500.0, 13000.0]
m.eval_steps = [200]
m.val_losses = [4.3]
m.val_ppls = [73.7]
m.eval_train_losses = [4.1]
fig = build_training_figure(m)
assert fig is not None
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard/plots.py
git commit -m "feat(dashboard): reorder plots — Loss/Tok/LR on top row; larger figure (20x10)"
```

---

## Task 6: Wire helpers into Live Monitor UI

Rebuild the Live Monitor tab in `build_app` to use the new helpers and a cleaner layout.

**Files:**
- Modify: `src/dashboard/app.py`

Changes:
1. `refresh_monitor` returns the status banner HTML and the progress detail HTML instead of separate boxes
2. Remove `_fmt_progress` (replaced by `_fmt_progress_detail`) and rename references
3. Status banner as `gr.HTML` at very top of Live Monitor (before run selector)
4. Progress section as `gr.HTML` (richer, larger)
5. ETA removed as separate box (it's now inside the progress HTML or shown alongside the run selector)

- [ ] **Step 1: Update `refresh_monitor` inside `build_app`**

Replace the `refresh_monitor` function inside `build_app` with:

```python
    def refresh_monitor(run_name):
        metrics, run_dir = _selected(runs_dir, run_name)
        stats = get_system_stats()
        log_lines = get_log_lines(run_dir) if run_dir else []
        eta = _compute_eta(metrics)
        return (
            _fmt_status_banner(metrics),
            _fmt_progress_detail(metrics, run_dir),
            eta,
            f"CPU {stats.cpu_percent:.1f}%  |  RAM {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB",
            _gpu_rows(stats),
            build_training_figure(metrics),
            "\n".join(log_lines) if log_lines else "(no log yet)",
        )
```

- [ ] **Step 2: Rebuild the Live Monitor tab layout**

Replace the entire `# ── TAB 1: Live Monitor ───` block (everything inside `with gr.Tab("Live Monitor"):`) with:

```python
            with gr.Tab("Live Monitor"):
                # Status banner — always visible at top
                status_banner = gr.HTML(
                    value="<div style='background:#f3f4f6;border-radius:6px;padding:10px 16px;"
                          "font-size:14px;color:#6b7280'>Waiting for first refresh…</div>"
                )

                with gr.Row():
                    run_selector = gr.Dropdown(label="Run", choices=choices, scale=3)
                    refresh_slider = gr.Slider(1, 30, value=5, step=1,
                                               label="Refresh every (s)", scale=2)
                    eta_box = gr.Textbox(label="ETA", scale=1, interactive=False)

                # Progress section — HTML for richer display
                progress_html = gr.HTML()

                plot_out = gr.Plot(label="Training Metrics")

                sys_header_box = gr.Textbox(label="System", interactive=False)
                gpu_table = gr.Dataframe(
                    headers=["GPU", "Name", "Mem Used (GB)", "Mem Total (GB)",
                             "Util (%)", "Temp (°C)"],
                    label="GPU Stats",
                )

                with gr.Row():
                    pdf_btn = gr.Button("⬇ Generate PDF", scale=1)
                    pdf_file = gr.File(label="Download", scale=2, interactive=False)

                with gr.Accordion("Training log (last 20 lines)", open=False):
                    log_box = gr.Textbox(lines=10, show_label=False, interactive=False)

                timer = gr.Timer(value=5)
                timer.tick(
                    fn=refresh_monitor, inputs=[run_selector],
                    outputs=[status_banner, progress_html, eta_box,
                             sys_header_box, gpu_table, plot_out, log_box],
                )
                refresh_slider.change(
                    fn=lambda v: gr.Timer(value=v),
                    inputs=[refresh_slider], outputs=[timer],
                )
                pdf_btn.click(fn=generate_pdf_file, inputs=[run_selector], outputs=[pdf_file])
```

- [ ] **Step 3: Smoke test that the app builds without error**

```bash
uv run python -c "
from pathlib import Path
from src.dashboard.app import build_app
demo = build_app(Path('runs'), Path('configs/default.yaml'))
print('build_app ok')
"
```

Expected: `build_app ok`

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/dashboard/ -v
```

Expected: all tests PASS (the new helper tests from Tasks 1–4 plus the original 33).

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat(dashboard): rebuild Live Monitor — status banner at top, rich progress HTML, cleaner layout"
```

---

## Task 7: Improve Architecture tab

Update the Architecture tab to use `_arch_and_config_text` and auto-load on run selection.

**Files:**
- Modify: `src/dashboard/app.py`

- [ ] **Step 1: Update `refresh_arch` inside `build_app`**

Replace the `refresh_arch` function inside `build_app` with:

```python
    def refresh_arch(run_name):
        metrics, _ = _selected(runs_dir, run_name)
        return _arch_and_config_text(metrics), metrics.architecture or {}, metrics.config or {}
```

- [ ] **Step 2: Rebuild the Architecture tab**

Replace the entire `# ── TAB 2: Architecture ───` block with:

```python
            with gr.Tab("Architecture"):
                arch_run_selector = gr.Dropdown(label="Run", choices=choices)
                arch_load_btn = gr.Button("Load")
                arch_box = gr.Textbox(
                    label="Model Architecture & Training Config",
                    lines=20,
                    interactive=False,
                )
                with gr.Accordion("Raw JSON — Architecture", open=False):
                    arch_json = gr.JSON()
                with gr.Accordion("Raw JSON — Config", open=False):
                    config_json = gr.JSON()
                arch_load_btn.click(
                    fn=refresh_arch,
                    inputs=[arch_run_selector],
                    outputs=[arch_box, arch_json, config_json],
                )
```

- [ ] **Step 3: Smoke test**

```bash
uv run python -c "
from pathlib import Path
from src.dashboard.app import build_app
demo = build_app(Path('runs'), Path('configs/default.yaml'))
print('build_app ok')
"
```

Expected: `build_app ok`

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/dashboard/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat(dashboard): Architecture tab shows params-first + training config, two JSON accordions"
```

---

## Self-Review

**Spec coverage:**
- ✅ Status banner at very top — Task 6 (wired), Task 2 (helper)
- ✅ Green when no errors, red/yellow when alerts — Task 2
- ✅ Progress section larger and clearer — Task 3 + Task 6
- ✅ Historical deltas shown (Δ total for train and val loss) — Task 3
- ✅ Progress bar (visual %) — Task 3
- ✅ ETA in h/m/s — Task 1
- ✅ Graphs reordered: Loss · Tok/s · LR+Grad on top row — Task 5
- ✅ Graphs larger (20×10) — Task 5
- ✅ Architecture shows params first — Task 4 + Task 7
- ✅ Training config shown in Architecture tab — Task 4 + Task 7

**Type consistency check:**
- `_fmt_status_banner(metrics: TrainingMetrics) -> str` used in Task 6 ✅
- `_fmt_progress_detail(metrics: TrainingMetrics, m_run_dir: Optional[Path]) -> str` used in Task 6 ✅
- `_compute_eta(metrics: TrainingMetrics) -> str` unchanged signature, used in Task 6 ✅
- `_arch_and_config_text(metrics: TrainingMetrics) -> str` used in Task 7 ✅
- `refresh_monitor` returns 7-tuple: `(status_banner_html, progress_html, eta, sys_header, gpu_rows, fig, log)` wired to 7 outputs in Task 6 ✅
- `refresh_arch` returns 3-tuple: `(text, arch_dict, config_dict)` wired to 3 outputs in Task 7 ✅
