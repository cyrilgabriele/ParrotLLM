# src/dashboard/notebook.py
"""ParrotLLM Training Dashboard — Jupyter ipywidgets UI (read-only monitor).

Usage:
    from src.dashboard.notebook import monitor
    monitor()                          # auto-detects latest run
    monitor(run_dir="runs/20260405_…") # specific run
    monitor(refresh=2)                 # custom refresh interval in seconds
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
from src.dashboard.run_manager import list_runs, get_latest_run_dir
from src.dashboard.plots import build_training_figure

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


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
    h, rem = divmod(eta_sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"~{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"~{m}m {s:02d}s"
    return f"~{s}s"


def _alerts_html(metrics: TrainingMetrics) -> str:
    alerts = detect_problems(metrics)
    if not alerts:
        return (
            "<div style='background:#d1fae5;border:1px solid #6ee7b7;border-radius:6px;"
            "padding:8px 16px;font-size:14px;font-weight:600;color:#065f46;margin-bottom:6px'>"
            "✅  No errors or malfunctions detected</div>"
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
            f"{emoji}  {a.code} — {a.message}: {a.detail}</div>"
        )
    return "\n".join(lines)


def _progress_html(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "<p style='color:#6b7280'>No training data yet.</p>"

    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    max_steps = metrics.config.get("max_steps")
    eta = _compute_eta(metrics)

    # Progress bar
    if max_steps:
        pct = 100.0 * step / max_steps
        bar = (
            f"<div style='background:#e5e7eb;border-radius:4px;height:12px;margin:6px 0'>"
            f"<div style='background:#2563EB;width:{pct:.1f}%;height:100%;border-radius:4px'></div>"
            f"</div>"
            f"<div style='font-size:13px;color:#6b7280'>"
            f"Step {step:,} / {max_steps:,} &nbsp;·&nbsp; {pct:.1f}% &nbsp;·&nbsp; ETA {eta}"
            f"</div>"
        )
    else:
        bar = f"<div style='font-size:13px;color:#6b7280'>Step {step:,} &nbsp;·&nbsp; ETA {eta}</div>"

    # Line 1: Train loss with delta
    if len(metrics.train_losses) >= 2:
        delta = metrics.train_losses[-1] - metrics.train_losses[0]
        delta_color = "#16a34a" if delta < 0 else "#dc2626"
        loss_html = (
            f"<b>Train Loss</b> {loss:.4f} "
            f"<span style='color:{delta_color};font-size:12px'>(Δ {delta:+.4f} total)</span>"
        )
    else:
        loss_html = f"<b>Train Loss</b> {loss:.4f}"

    # Val loss with delta (line 2 — only when available)
    val_html = ""
    if metrics.val_losses:
        val = metrics.val_losses[-1]
        if len(metrics.val_losses) >= 2:
            vd = metrics.val_losses[-1] - metrics.val_losses[0]
            vd_color = "#16a34a" if vd < 0 else "#dc2626"
            val_html = (
                f"&nbsp;·&nbsp; <b>Val Loss</b> {val:.4f} "
                f"<span style='color:{vd_color};font-size:12px'>(Δ {vd:+.4f} total)</span>"
            )
        else:
            val_html = f"&nbsp;·&nbsp; <b>Val Loss</b> {val:.4f}"

    # Secondary: LR, Grad Norm, Tok/s, Best Step
    secondary = [f"<b>LR</b> {lr:.2e}"]
    if metrics.grad_norms:
        secondary.append(f"<b>Grad Norm</b> {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        secondary.append(f"<b>Tok/s</b> {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        secondary.append(f"<b>Best Step</b> {metrics.best_step:,}")

    stale_html = ""
    if run_dir is not None:
        stale, age = is_metrics_stale(run_dir)
        if stale:
            stale_html = (
                f"<div style='color:#d97706;margin-top:6px;font-size:13px'>"
                f"⚠ Metrics not updated for {age}s — training may have stalled or crashed.</div>"
            )

    return (
        f"{bar}"
        f"<div style='margin-top:8px;font-size:15px'>{loss_html}{val_html}</div>"
        f"<div style='margin-top:4px;font-size:13px;color:#4b5563'>"
        + "&nbsp;·&nbsp;".join(secondary)
        + f"</div>{stale_html}"
    )


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


class _Monitor:
    """Read-only monitor widget."""

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

        self._dropdown = widgets.Dropdown(
            options=run_names, value=initial,
            description="Run:", layout=widgets.Layout(width="320px"),
        )
        self._stop_btn = widgets.Button(
            description="■ Stop refresh", button_style="warning",
            layout=widgets.Layout(width="140px"),
        )
        self._status_w = widgets.HTML()
        self._progress_w = widgets.HTML()
        self._gpu_w = widgets.HTML()
        self._plot_w = widgets.Output()

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
            self._status_w.value = "<p style='color:#6b7280'>No runs found in runs/.</p>"
            return
        metrics = read_metrics(run_dir)
        stats = get_system_stats()
        self._status_w.value = _alerts_html(metrics)
        self._progress_w.value = _progress_html(metrics, run_dir)
        self._gpu_w.value = _gpu_html(stats)
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

    def widget(self) -> widgets.VBox:
        header = widgets.HBox([self._dropdown, self._stop_btn])
        return widgets.VBox([
            header,
            self._status_w,
            self._progress_w,
            self._gpu_w,
            self._plot_w,
        ])


def monitor(
    runs_dir: str | Path = "runs",
    run_dir: Optional[str | Path] = None,
    refresh: int = 2,
    config_path: str | Path = "configs/default.yaml",
) -> None:
    """Display the ParrotLLM training monitor widget in a Jupyter notebook.

    Args:
        runs_dir:    Directory containing run subdirectories. Default: "runs".
        run_dir:     Specific run directory to show. Default: auto-detects latest.
        refresh:     Auto-refresh interval in seconds. Default: 2.
        config_path: Unused, kept for backwards compatibility.
    """
    runs_dir = Path(runs_dir)
    run_dir = Path(run_dir) if run_dir is not None else None
    m = _Monitor(runs_dir, run_dir, refresh)
    display(m.widget())
