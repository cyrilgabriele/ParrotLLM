# src/dashboard/notebook.py
"""ParrotLLM Training Dashboard — Jupyter ipywidgets UI.

Phase 1: read-only monitoring (auto-refresh, run selector, GPU table, alerts, plot).
Phase 2: run management buttons (see bottom of file).

Usage:
    from src.dashboard.notebook import monitor
    monitor()                          # auto-detects latest run
    monitor(run_dir="runs/20260405_…") # specific run
    monitor(refresh=10)                # custom refresh interval in seconds
"""
from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, Image, clear_output

from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import list_runs, get_latest_run_dir, launch_training, kill_training
from src.dashboard.plots import build_training_figure

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _metrics_html(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "<i>No training data yet. Start training first.</i>"
    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    parts = [f"<b>Step</b> {step:,}", f"<b>Train Loss</b> {loss:.4f}", f"<b>LR</b> {lr:.2e}"]
    if metrics.val_losses:
        parts.append(f"<b>Val</b> {metrics.val_losses[-1]:.4f}")
    if metrics.grad_norms:
        parts.append(f"<b>Grad Norm</b> {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        parts.append(f"<b>Tok/s</b> {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        parts.append(f"<b>Best Step</b> {metrics.best_step:,}")
    html = "  │  ".join(parts)
    if run_dir is not None:
        stale, age = is_metrics_stale(run_dir)
        if stale:
            html += f"<br><span style='color:orange'>⚠ Metrics not updated for {age}s</span>"
    return html


def _gpu_html(stats) -> str:
    header = (f"<b>CPU</b> {stats.cpu_percent:.1f}%  │  "
              f"<b>RAM</b> {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB")
    if not stats.gpu_available:
        return header + "  │  GPU: N/A"
    rows = []
    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}%" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}°C" if not math.isnan(g.temperature_c) else "—"
        rows.append(f"<tr><td>{g.index}</td><td>{g.name}</td>"
                    f"<td>{g.mem_used_gb:.1f} GB</td><td>{g.mem_total_gb:.1f} GB</td>"
                    f"<td>{util}</td><td>{temp}</td></tr>")
    avg_util = (f"{stats.gpu_avg_utilization:.0f}%"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    rows.append(f"<tr><td><b>ALL</b></td><td></td>"
                f"<td><b>{stats.gpu_total_used_gb:.1f} GB</b></td>"
                f"<td><b>{stats.gpu_total_mem_gb:.1f} GB</b></td>"
                f"<td><b>{avg_util}</b></td><td>—</td></tr>")
    table = (
        "<table border='1' style='border-collapse:collapse;font-size:12px;margin-top:4px'>"
        "<tr><th>GPU</th><th>Name</th><th>Mem Used</th><th>Mem Total</th>"
        "<th>Util</th><th>Temp</th></tr>"
        + "".join(rows) + "</table>"
    )
    return header + "<br>" + table


def _alerts_html(metrics: TrainingMetrics) -> str:
    alerts = detect_problems(metrics)
    if not alerts:
        return ""
    lines = [f"{_SEVERITY_EMOJI[a.severity]} <b>{a.code}</b> — {a.message}" for a in alerts]
    return (
        "<div style='background:#fff3cd;padding:6px;border-radius:4px;margin:4px 0'>"
        + "<br>".join(lines) + "</div>"
    )


class _Monitor:
    """Phase 1 read-only monitor widget."""

    def __init__(self, runs_dir: Path, run_dir: Optional[Path], refresh: int):
        self._runs_dir = runs_dir
        self._refresh = refresh
        self._timer: Optional[threading.Timer] = None
        self._stopped = False

        runs = list_runs(runs_dir)
        run_names = [r.name for r in runs]
        initial = run_dir.name if run_dir and run_dir.name in run_names else (
            run_names[0] if run_names else ""
        )

        self._dropdown = widgets.Dropdown(options=run_names, value=initial,
                                          description="Run:", layout=widgets.Layout(width="300px"))
        self._stop_btn = widgets.Button(description="■ Stop refresh",
                                        button_style="warning",
                                        layout=widgets.Layout(width="140px"))
        self._metrics_w = widgets.HTML()
        self._gpu_w = widgets.HTML()
        self._alerts_w = widgets.HTML()
        self._plot_w = widgets.Output()

        # Phase 2: run management
        self._config_path = Path("configs/default.yaml")
        self._proc = None
        self._proc_lock = threading.Lock()

        self._start_btn = widgets.Button(description="▶ Start", button_style="success",
                                         layout=widgets.Layout(width="100px"))
        self._resume_btn = widgets.Button(description="⏩ Resume", button_style="info",
                                          layout=widgets.Layout(width="100px"))
        self._stop_btn2 = widgets.Button(description="⏹ Stop", button_style="danger",
                                         layout=widgets.Layout(width="100px"),
                                         disabled=True)
        self._action_out = widgets.HTML()

        self._start_btn.on_click(self._on_start)
        self._resume_btn.on_click(self._on_resume)
        self._stop_btn2.on_click(self._on_stop)

        self._stop_btn.on_click(lambda _: self._stop())
        self._dropdown.observe(
            lambda change: self._refresh_data() if change["name"] == "value" else None
        )

        self._refresh_data()
        self._schedule()

    def _get_run_dir(self) -> Optional[Path]:
        val = self._dropdown.value
        if val:
            return self._runs_dir / val
        return get_latest_run_dir(self._runs_dir)

    def _refresh_data(self):
        run_dir = self._get_run_dir()
        if run_dir is None:
            self._metrics_w.value = "<i>No runs found. Start training first.</i>"
            return
        metrics = read_metrics(run_dir)
        stats = get_system_stats()
        self._metrics_w.value = _metrics_html(metrics, run_dir)
        self._gpu_w.value = _gpu_html(stats)
        alerts = _alerts_html(metrics)
        self._alerts_w.value = alerts
        self._alerts_w.layout.display = "" if alerts else "none"
        fig = build_training_figure(metrics)
        with self._plot_w:
            clear_output(wait=True)
            if fig:
                display(Image(data=_fig_to_png(fig)))

    def _schedule(self):
        if not self._stopped:
            self._timer = threading.Timer(self._refresh, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self):
        self._refresh_data()
        self._schedule()

    def _stop(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._stop_btn.description = "■ Stopped"
        self._stop_btn.disabled = True

    def _on_start(self, _):
        with self._proc_lock:
            self._proc = launch_training(config_path=self._config_path)
        self._action_out.value = f"<b>Started.</b> PID: {self._proc.pid}"
        self._start_btn.disabled = True
        self._resume_btn.disabled = True
        self._stop_btn2.disabled = False

    def _on_resume(self, _):
        run_dir = self._get_run_dir()
        if run_dir is None:
            self._action_out.value = "<span style='color:red'>No run selected.</span>"
            return
        with self._proc_lock:
            self._proc = launch_training(config_path=self._config_path,
                                         resume_run_dir=run_dir)
        self._action_out.value = f"<b>Resumed</b> {run_dir.name}. PID: {self._proc.pid}"
        self._start_btn.disabled = True
        self._resume_btn.disabled = True
        self._stop_btn2.disabled = False

    def _on_stop(self, _):
        with self._proc_lock:
            if self._proc is not None:
                kill_training(self._proc)
                self._proc = None
        self._action_out.value = "<b>Training stopped.</b>"
        self._start_btn.disabled = False
        self._resume_btn.disabled = False
        self._stop_btn2.disabled = True

    def widget(self) -> widgets.VBox:
        refresh_header = widgets.HBox([self._dropdown, self._stop_btn])
        mgmt_row = widgets.HBox([
            self._start_btn, self._resume_btn, self._stop_btn2, self._action_out
        ])
        return widgets.VBox([
            refresh_header,
            mgmt_row,
            self._metrics_w,
            self._gpu_w,
            self._alerts_w,
            self._plot_w,
        ])


def monitor(
    runs_dir: str | Path = "runs",
    run_dir: Optional[str | Path] = None,
    refresh: int = 5,
) -> None:
    """Display the ParrotLLM training monitor widget in a Jupyter notebook.

    Args:
        runs_dir: Directory containing run subdirectories. Default: "runs".
        run_dir:  Specific run directory to show. Default: auto-detects latest.
        refresh:  Auto-refresh interval in seconds. Default: 5.
    """
    runs_dir = Path(runs_dir)
    run_dir = Path(run_dir) if run_dir is not None else None
    m = _Monitor(runs_dir, run_dir, refresh)
    display(m.widget())
