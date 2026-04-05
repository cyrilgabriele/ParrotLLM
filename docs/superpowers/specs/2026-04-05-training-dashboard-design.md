# Training Dashboard — Design Spec

**Date:** 2026-04-05
**Status:** Approved

---

## Goal

Build a multi-frontend training dashboard for ParrotLLM that works in every GPU cluster
scenario: browser on the server, SSH tunnel, public URL, Jupyter notebook, and raw terminal.
All frontends share a single set of backend modules and show consistent information.

---

## Folder Structure

```
src/dashboard/              ← self-contained package; imports freely from src/*
├── __init__.py
├── metrics_reader.py       ← parse metrics.jsonl → TrainingMetrics dataclass
├── system_monitor.py       ← CPU, RAM, per-GPU stats → SystemStats dataclass
├── problem_detector.py     ← analyze TrainingMetrics → list[Alert]
├── run_manager.py          ← list runs, launch/resume/kill training subprocess
├── plots.py                ← matplotlib figures for Gradio + Jupyter
├── app.py                  ← Frontend 1: Gradio web UI
├── notebook.py             ← Frontend 2: Jupyter ipywidgets UI
└── tui.py                  ← Frontend 3: Rich terminal UI
```

`src/dashboard/` is a peer of `src/training/`, `src/model/`, `src/eval/`. Nothing outside
`src/dashboard/` imports from it. It is not a script and not at the project root.

Tests live in `tests/dashboard/` mirroring this structure.

---

## Access Modes

All modes are launched via `uv run main.py --stage dashboard [flags]`.

| Mode | Command | When to use |
|------|---------|-------------|
| **Browser on server** | `--open` | Server has a browser (GUI, VNC, X11) — auto-opens `localhost:7861` |
| **SSH tunnel → Gradio** | *(no extra flag)* | SSH access, browser on your laptop: `ssh -L 7861:localhost:7861 user@server` |
| **Gradio public URL** | `--share` | No tunnel possible — creates a temporary `gradio.live` URL |
| **Jupyter widget** | `from src.dashboard.notebook import monitor; monitor()` | Jupyter already running and tunneled on port 8888 |
| **Terminal TUI** | `--tui` | Bare SSH, no ports, no browser anywhere |

`--open` calls `webbrowser.open("http://localhost:7861")` after the server starts (one line).
`--share` passes `share=True` to `demo.launch()` (one line).
Both flags are additive to the Gradio app — no separate frontend needed.

---

## Backend Modules

### metrics_reader.py — unchanged from existing plan

Parses `metrics.jsonl` into a `TrainingMetrics` dataclass. Returns an empty dataclass if the
file does not exist. All frontends call `read_metrics(run_dir)`.

### system_monitor.py — extended for multi-GPU

New `GPUStats` dataclass per device. `SystemStats` holds a list of them plus combined totals.

```python
@dataclass
class GPUStats:
    index: int
    name: str
    mem_used_gb: float
    mem_total_gb: float
    utilization_pct: float   # from pynvml; nan if unavailable
    temperature_c: float     # from pynvml; nan if unavailable

@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpus: list[GPUStats]          # one entry per device
    gpu_total_used_gb: float      # sum across all GPUs
    gpu_total_mem_gb: float       # sum across all GPUs
    gpu_avg_utilization: float    # mean across all GPUs; nan if unavailable
    gpu_available: bool
```

`get_system_stats()` iterates `range(torch.cuda.device_count())` and queries `pynvml` for
utilization and temperature per device. If `pynvml` is unavailable (no NVIDIA driver, CI),
`utilization_pct` and `temperature_c` fall back to `float("nan")` — the function never raises.

**New dependency:** `pynvml` added to `pyproject.toml`.

### problem_detector.py — unchanged from existing plan

Detects: `GRAD_EXPLOSION`, `OVERFITTING`, `STAGNATION`, `HIGH_LOSS`, `LR_ZERO`.
Returns `list[Alert]` with `Severity.ERROR / WARNING / INFO`.

### run_manager.py — one addition

Add `kill_training()` which sends `SIGTERM` to the active subprocess. Used by Gradio Run
Manager (Stop button) and Jupyter Phase 2. TUI is read-only and does not call it.

### plots.py — style change only

Switch from dark background (`#1a1a2e`) to light/neutral style (`facecolor="white"`) so
figures blend with the Gradio light theme. The `plot_training.py` PDF plotter is unaffected.

---

## Frontend 1: Gradio (Browser on Server / SSH Tunnel / Public URL)

**File:** `src/dashboard/app.py`
**Tech:** Gradio 5 (already installed), `gr.Timer` for auto-refresh.

### Tab 1 — Live Monitor

```
┌─ ParrotLLM Training Dashboard ─────────────────────────────────────────────────┐
│  [Live Monitor]  [Architecture]  [Run Manager]                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Run: [dropdown ▾ 20260405_143022    ]   Refresh every [====5====] s            │
│                                                                                  │
│  ┌─ Alerts ───────────────────────────────────────────────────────────────────┐ │
│  │  🔴 GRAD_EXPLOSION — Grad norm 14.2 in last 3 steps. Reduce LR.           │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│  (hidden entirely when no alerts)                                                │
│                                                                                  │
│  Step 1,200 / 10,000 (12.0%)  │  Loss 3.2134  │  Val 3.4501  │  ETA ~2h 14m   │
│                                                                                  │
│  ┌─ Training Metrics ─────────────────────────────────────────────────────────┐ │
│  │         [Train & Val Loss] [Perplexity] [LR] [Grad Norm] [Tok/s] [Gap]    │ │
│  │                      (3×2 matplotlib figure, light theme)                  │ │
│  └──────────────────────────────────────────── [⬇ Download PDF] ─────────────┘ │
│                                                                                  │
│  ┌─ System ───────────────────────────────────────────────────────────────────┐ │
│  │  CPU 42.1%  │  RAM 8.1 / 16.0 GB                                           │ │
│  │                                                                             │ │
│  │  GPU   Name             Mem Used   Mem Total   Util    Temp                 │ │
│  │   0    A100-SXM-80G     52.1 GB    80.0 GB     87%     72°C                │ │
│  │   1    A100-SXM-80G     51.8 GB    80.0 GB     85%     70°C                │ │
│  │   2    A100-SXM-80G     52.0 GB    80.0 GB     86%     71°C                │ │
│  │   3    A100-SXM-80G     51.9 GB    80.0 GB     84%     69°C                │ │
│  │  ALL                   207.8 GB   320.0 GB     85.5%                        │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ▸ Training log (last 20 lines)  [collapsed accordion]                          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Behaviour:**
- Alerts row: `gr.Dataframe` with colour-coded severity. Hidden (`visible=False`) when no alerts.
- Progress row: single `gr.Textbox` updated each timer tick.
- ETA: computed from `tokens_per_sec × remaining_steps`. Shows `—` if tokens/sec unavailable.
- Refresh slider: `gr.Slider(1, 30, value=5)` updates `gr.Timer` interval live.
- Download PDF: `gr.DownloadButton` wired to `plot_run_dir(run_dir)` — regenerates on click.
- GPU table: `gr.Dataframe` with columns `GPU | Name | Mem Used | Mem Total | Util | Temp`.
  ALL row pinned at bottom. Util/Temp show `—` when pynvml unavailable.
- Log tail: `gr.Accordion` (collapsed by default) containing a `gr.Textbox`. The `RunManager`
  captures subprocess stdout into a 20-line rotating buffer; this reads from it.
- Empty state: when no runs exist, shows `"No runs found in runs/. Start training first."`
- Stale data warning: if `metrics.jsonl` not modified in >60s but process appears alive,
  shows `"⚠ Metrics not updated for Xs — training may have stalled."` in the progress row.

### Tab 2 — Architecture

```
│  Run: [dropdown ▾]   [Load]                                                      │
│                                                                                   │
│  ┌─ Summary ──────────────────────────────────────────────────────────────────┐  │
│  │  Vocab size: 50,257    d_model: 320    Layers: 16    Heads: 8              │  │
│  │  FFN dim:    854       Context: 1024                                        │  │
│  │  Total params: 35,763,840    Trainable: 35,763,840                          │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ▸ Raw JSON  [collapsed accordion]                                                │
```

### Tab 3 — Run Manager

```
│  Status: ● Running — PID 18432   (refreshes every 5s)                            │
│                                                                                   │
│  ┌─ Controls ─────────────────────────────────────────────────────────────────┐  │
│  │  [▶ Start New Run]   [⏩ Resume: dropdown ▾ 20260405…]   [⏹ Stop]          │  │
│  │   Output: "Started training. PID: 18432"                                    │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ┌─ All Runs ──────────────────────────────────────────────── [Refresh] ───────┐ │
│  │  Run                    Last Step   Best Val Loss   Status                   │ │
│  │  20260405_143022        1,200       3.4501          running                  │ │
│  │  20260402_091500        10,000      3.1204          done                     │ │
│  └────────────────────────────────────────────────────────────────────────────┘  │
```

**Behaviour:**
- Start button: disabled when a process is alive.
- Stop button: calls `kill_training()`. Disabled when no process is alive.
- Resume dropdown: populated from `list_runs()`.
- Status textbox: auto-refreshes alongside Live Monitor timer.

---

## Frontend 2: Jupyter Widget

**File:** `src/dashboard/notebook.py`
**New dependency:** `ipywidgets>=8.0` added to `pyproject.toml`.
**No `ipympl` required** — matplotlib renders to PNG bytes via `Agg` backend and displays
with `IPython.display.Image`. Works in Jupyter Notebook and JupyterLab without extensions.

### Usage

```python
from src.dashboard.notebook import monitor
monitor()                            # auto-detects latest run
monitor(run_dir="runs/20260405_…")   # specific run
monitor(refresh=10)                  # custom refresh interval in seconds
```

### Phase 1 — Read-only (always shippable, permanent backup)

Layout (built with `ipywidgets.VBox/HBox`):

```
┌─ ParrotLLM Monitor ──────────────────── [Run ▾] [■ Stop refresh] ──────────────┐
│                                                                                  │
│  Step 1,200  │  Train Loss 3.2134  │  Val 3.4501  │  LR 1.20e-4  │  ETA ~2h14m │
│  CPU 42%  │  RAM 8.1/16 GB                                                      │
│                                                                                  │
│  GPU   Name             Mem Used   Mem Total   Util    Temp                      │
│   0    A100-SXM-80G     52.1 GB    80.0 GB     87%     72°C                     │
│   1    A100-SXM-80G     51.8 GB    80.0 GB     85%     70°C                     │
│   2    A100-SXM-80G     52.0 GB    80.0 GB     86%     71°C                     │
│   3    A100-SXM-80G     51.9 GB    80.0 GB     84%     69°C                     │
│  ALL                   207.8 GB   320.0 GB     85.5%                             │
│                                                                                  │
│  Alerts: 🔴 GRAD_EXPLOSION — Grad norm 14.2 in last 3 steps.                    │
│  (hidden when no alerts)                                                         │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │               [matplotlib figure — reuses plots.py]                         │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Behaviour:**
- Metrics row: `ipywidgets.HTML`, updated each tick.
- GPU table: `ipywidgets.HTML` table, one row per GPU + ALL row.
- Alerts row: `ipywidgets.HTML`. Hidden when no alerts.
- Plot: `fig.savefig(buf, format="png")` → `IPython.display.Image(data=buf.getvalue())`.
  Rendered inside an `ipywidgets.Output` widget via `clear_output(wait=True)`.
- Auto-refresh: `threading.Timer` fires every N seconds, updates all widgets in-place.
- Run dropdown: `ipywidgets.Dropdown` from `list_runs()`. Switching run restarts the timer.
- Stop refresh button: cancels the timer so the user can inspect a frozen view.
- Empty state: shows `"No runs found. Start training first."` in the metrics row.
- Stale data warning: same 60s threshold as Gradio, shown in metrics row.

### Phase 2 — Run management (additive, does not break Phase 1)

Three buttons added to the header row:

| Button | Widget | Action |
|--------|--------|--------|
| `▶ Start` | `ipywidgets.Button` | `launch_training(config_path)` — disabled if process alive |
| `⏩ Resume` | `ipywidgets.Button` | `launch_training(resume_run_dir=selected)` |
| `⏹ Stop` | `ipywidgets.Button` | `kill_training()` — disabled if no process alive |

A small `ipywidgets.Output` below the buttons shows the last 10 lines from the subprocess
stdout buffer (same buffer Gradio's log tail reads from).

Phase 2 adds no new dependencies.

---

## Frontend 3: Rich TUI

**File:** `src/dashboard/tui.py`
**New dependencies:** `rich`, `plotext` — both added to `pyproject.toml`.
**Launched via:** `uv run main.py --stage dashboard --tui`
**Exit:** `Ctrl+C`. No other keyboard interaction.
**Read-only** — no run management. Designed for passive monitoring over plain SSH.

### Layout

```
┌─ ParrotLLM Training Monitor ──────────────── 5s refresh ── Ctrl+C to exit ─────┐
│                                                                                  │
│ ┌─ Progress ───────────────────────────────────────────────────────────────────┐ │
│ │ Run: 20260405_143022   Step 1,200 / 10,000 (12.0%)                           │ │
│ │ Train 3.2134  │  Val 3.4501  │  Val PPL 31.5  │  LR 1.20e-4  │  ETA ~2h 14m │ │
│ │ Grad Norm 0.82  │  Tok/s 12,450  │  Best Step 800                            │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│ ┌─ Alerts ─────────────────────────────────────────────────────────────────────┐ │
│ │  ✅  No problems detected                                                    │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│ ┌─ System ─────────────────────────────────────────────────────────────────────┐ │
│ │ CPU 42.1%  │  RAM 8.1 / 16.0 GB                                              │ │
│ │                                                                               │ │
│ │ GPU   Name             Mem Used   Mem Total   Util    Temp                    │ │
│ │  0    A100-SXM-80G     52.1 GB    80.0 GB     87%     72°C                   │ │
│ │  1    A100-SXM-80G     51.8 GB    80.0 GB     85%     70°C                   │ │
│ │  2    A100-SXM-80G     52.0 GB    80.0 GB     86%     71°C                   │ │
│ │  3    A100-SXM-80G     51.9 GB    80.0 GB     84%     69°C                   │ │
│ │ ALL                   207.8 GB   320.0 GB     85.5%                           │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Behaviour:**
- `rich.Live` wraps a `rich.Layout` and re-renders every N seconds.
  Refresh interval configurable via `--tui-refresh N` (default 5).
- Progress panel: `rich.Panel` containing a `rich.Table` (one row, labelled columns).
- Alerts panel: `rich.Panel` with a `rich.Table`. Severity colours: red / yellow / green.
  Shows `✅ No problems detected` when alert list is empty.
- System panel: `rich.Panel` with a `rich.Table` for GPU rows. ALL row at bottom.
  Util/Temp columns show `—` when pynvml unavailable.
- Always shows the latest run (from `get_latest_run_dir()`). No run switching.
- Empty state: Progress panel shows `"No runs found in runs/. Start training first."`.
- Stale data warning: shown in Progress panel using the same 60s threshold.
- No ASCII charts — removed as not useful in terminal.

**What TUI intentionally omits vs Gradio/Jupyter:**
- No run switching (always latest)
- No run management (start/stop/resume)
- No training log tail
- No plots

These are deliberate — TUI is the "just tell me what's happening" option.

---

## Cross-Cutting UX Rules

These apply identically across all three frontends:

**Information hierarchy (top to bottom):**
1. Alerts — most urgent; hidden entirely when none
2. Progress (step, loss, ETA)
3. System stats (GPU table)
4. Everything else (plots in Gradio/Jupyter; not in TUI)

**Consistent metric labels across all frontends:**
`Train Loss`, `Val Loss`, `Val PPL`, `Grad Norm`, `Tok/s`, `LR`, `ETA`, `Best Step`

**Empty state:**
All frontends show `"No runs found in runs/. Start training first."` before any training has run.

**Stale data warning:**
If `metrics.jsonl` has not been modified in >60s but a training process appears alive, show:
`"⚠ Metrics not updated for Xs — training may have stalled or crashed."`
Detected via `Path.stat().st_mtime` vs `time.time()`.

---

## New Dependencies

| Package | Reason | Added to |
|---------|--------|----------|
| `psutil>=6.0.0` | CPU + RAM stats | `pyproject.toml` (already in existing plan) |
| `pynvml` | Per-GPU utilization %, temperature | `pyproject.toml` |
| `ipywidgets>=8.0` | Jupyter widget UI | `pyproject.toml` |
| `rich` | TUI rendering | `pyproject.toml` |

---

## CLI Changes

`main.py` `--stage` choices extended to include `dashboard`. Handler reads `runs_dir` from
config and dispatches based on flags:

```
--tui              → tui.run_tui(runs_dir, refresh=N)
--tui-refresh N    → set TUI refresh interval in seconds (default 5); only valid with --tui
--open             → app.run_dashboard(..., open_browser=True)
--share            → app.run_dashboard(..., share=True)
(no flag)          → app.run_dashboard(runs_dir, config_path)
```

---

## Out of Scope

- Run comparison side-by-side (already available via `plot_training.py` CLI)
- Config editor (read-only display only)
- Checkpoint deletion (too destructive for a monitoring tool)
- Notebook `.ipynb` file committed to repo (use `monitor()` from any existing notebook)
