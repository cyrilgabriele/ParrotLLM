# src/dashboard/tui.py
"""ParrotLLM Training Dashboard — Rich terminal UI.

Read-only. Always shows the latest run. Exit with Ctrl+C.

Launch: uv run main.py --stage dashboard --tui [--tui-refresh N]
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import rich.box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.dashboard.metrics_reader import read_metrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import get_latest_run_dir

_SEVERITY_STYLE = {Severity.ERROR: "red", Severity.WARNING: "yellow", Severity.INFO: "blue"}


def _progress_panel(run_dir: Path) -> Panel:
    metrics = read_metrics(run_dir)
    if not metrics.steps:
        return Panel(Text("No training data yet. Start training first.", style="dim"),
                     title="Progress")
    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    max_steps = metrics.config.get("max_steps")

    lines = [Text(f"Run: {run_dir.name}", style="bold")]

    step_line = f"Step {step:,}"
    if max_steps:
        step_line += f" / {max_steps:,}  ({100.0 * step / max_steps:.1f}%)"
    lines.append(Text(step_line))

    row = f"Train {loss:.4f}"
    if metrics.val_losses:
        row += f"  │  Val {metrics.val_losses[-1]:.4f}"
    if metrics.val_ppls:
        row += f"  │  Val PPL {metrics.val_ppls[-1]:.1f}"
    row += f"  │  LR {lr:.2e}"
    lines.append(Text(row))

    extra = []
    if metrics.grad_norms:
        extra.append(f"Grad Norm {metrics.grad_norms[-1]:.3f}")
    if metrics.tokens_per_sec:
        extra.append(f"Tok/s {metrics.tokens_per_sec[-1]:,.0f}")
    if metrics.best_step:
        extra.append(f"Best Step {metrics.best_step:,}")
    if extra:
        lines.append(Text("  │  ".join(extra)))

    stale, age = is_metrics_stale(run_dir)
    if stale:
        lines.append(Text(f"⚠ Metrics not updated for {age}s — training may have stalled or crashed.",
                          style="yellow"))

    return Panel(Group(*lines), title="Progress")


def _alerts_panel(run_dir: Path) -> Panel:
    metrics = read_metrics(run_dir)
    alerts = detect_problems(metrics)
    if not alerts:
        return Panel(Text("✅  No problems detected", style="green"), title="Alerts")
    table = Table.grid(padding=(0, 2))
    for a in alerts:
        style = _SEVERITY_STYLE[a.severity]
        table.add_row(
            Text("●", style=style),
            Text(a.code, style=f"bold {style}"),
            Text(f"— {a.message}"),
        )
    return Panel(table, title="Alerts")


def _system_panel() -> Panel:
    stats = get_system_stats()
    header = Text(
        f"CPU {stats.cpu_percent:.1f}%  │  "
        f"RAM {stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB"
    )
    if not stats.gpu_available:
        return Panel(Group(header, Text("GPU: N/A", style="dim")), title="System")

    table = Table(box=rich.box.SIMPLE, show_header=True, header_style="bold",
                  show_edge=False, pad_edge=False)
    table.add_column("GPU", justify="right", style="dim")
    table.add_column("Name")
    table.add_column("Mem Used", justify="right")
    table.add_column("Mem Total", justify="right")
    table.add_column("Util", justify="right")
    table.add_column("Temp", justify="right")

    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}%" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}°C" if not math.isnan(g.temperature_c) else "—"
        table.add_row(str(g.index), g.name,
                      f"{g.mem_used_gb:.1f} GB", f"{g.mem_total_gb:.1f} GB",
                      util, temp)

    avg_util = (f"{stats.gpu_avg_utilization:.0f}%"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    table.add_row(
        "[bold]ALL[/bold]", "",
        f"[bold]{stats.gpu_total_used_gb:.1f} GB[/bold]",
        f"[bold]{stats.gpu_total_mem_gb:.1f} GB[/bold]",
        f"[bold]{avg_util}[/bold]", "—",
    )

    return Panel(Group(header, table), title="System")


def _build_layout(run_dir: Path):
    from rich.layout import Layout
    layout = Layout()
    layout.split_column(
        Layout(_progress_panel(run_dir), name="progress", size=7),
        Layout(_alerts_panel(run_dir), name="alerts", size=5),
        Layout(_system_panel(), name="system"),
    )
    return layout


def run_tui(runs_dir: Path, refresh: int = 2, run_name: str | None = None) -> None:
    """Run the terminal dashboard. Blocks until Ctrl+C."""
    console = Console()

    if run_name:
        run_dir = runs_dir / run_name
        if not run_dir.is_dir():
            console.print(f"[red]Run not found: {run_dir}[/red]")
            return
    else:
        run_dir = get_latest_run_dir(runs_dir)
        if run_dir is None:
            console.print("[yellow]No runs found in runs/. Start training first.[/yellow]")
            return

    console.print(
        f"[dim]ParrotLLM TUI — {refresh}s refresh — Ctrl+C to exit — "
        f"showing: {run_dir.name}[/dim]"
    )

    with Live(_build_layout(run_dir), refresh_per_second=0.5, console=console) as live:
        try:
            while True:
                time.sleep(refresh)
                if not run_name:
                    # Auto-follow the latest run unless user pinned one
                    latest = get_latest_run_dir(runs_dir)
                    if latest:
                        run_dir = latest
                live.update(_build_layout(run_dir))
        except KeyboardInterrupt:
            pass
