# src/dashboard/app.py
"""ParrotLLM Training Dashboard — Gradio UI."""
from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")

from src.dashboard.metrics_reader import read_metrics, TrainingMetrics, is_metrics_stale
from src.dashboard.system_monitor import get_system_stats
from src.dashboard.problem_detector import detect_problems, Severity
from src.dashboard.run_manager import (
    list_runs, launch_training, get_latest_run_dir,
    kill_training, get_log_lines,
)
from src.dashboard.plots import build_training_figure

_active_proc: Optional[object] = None
_proc_lock = threading.Lock()

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_choices(runs_dir: Path) -> list[str]:
    return [r.name for r in list_runs(runs_dir)]


def _selected(runs_dir: Path, run_name: str) -> tuple[TrainingMetrics, Optional[Path]]:
    run_dir = (runs_dir / run_name) if run_name else get_latest_run_dir(runs_dir)
    if run_dir is None:
        return TrainingMetrics(), None
    return read_metrics(run_dir), run_dir


def _fmt_progress(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "No runs found in runs/. Start training first."
    current = metrics.steps[-1]
    max_steps = metrics.config.get("max_steps")
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    parts = [f"Step {current:,}"]
    if max_steps:
        parts.append(f"/ {max_steps:,} ({100.0 * current / max_steps:.1f}%)")
    parts.append(f"| Loss {loss:.4f} | LR {lr:.2e}")
    if metrics.val_losses:
        parts.append(f"| Val {metrics.val_losses[-1]:.4f}")
    if metrics.best_step:
        parts.append(f"| Best @ {metrics.best_step:,}")
    if run_dir is not None:
        stale, age = is_metrics_stale(run_dir)
        if stale:
            parts.append(f"\n⚠ Metrics not updated for {age}s — training may have stalled or crashed.")
    return " ".join(parts)


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


def _alert_rows(metrics: TrainingMetrics) -> list[list[str]]:
    alerts = detect_problems(metrics)
    return [[_SEVERITY_EMOJI[a.severity], a.code, a.message] for a in alerts]


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

    # ── Secondary metrics ─────────────────────────────────────────────
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
        + "&nbsp;·&nbsp;".join(secondary)
        + f"</div>{stale_html}"
    )


def _gpu_rows(stats) -> list[list[str]]:
    if not stats.gpu_available:
        return [["—", "No GPU detected", "—", "—", "—", "—"]]
    rows = []
    for g in stats.gpus:
        util = f"{g.utilization_pct:.0f}" if not math.isnan(g.utilization_pct) else "—"
        temp = f"{g.temperature_c:.0f}" if not math.isnan(g.temperature_c) else "—"
        rows.append([str(g.index), g.name,
                     f"{g.mem_used_gb:.1f}", f"{g.mem_total_gb:.1f}", util, temp])
    avg_util = (f"{stats.gpu_avg_utilization:.0f}"
                if not math.isnan(stats.gpu_avg_utilization) else "—")
    rows.append(["ALL", "", f"{stats.gpu_total_used_gb:.1f}",
                 f"{stats.gpu_total_mem_gb:.1f}", avg_util, "—"])
    return rows


def _is_alive() -> bool:
    with _proc_lock:
        return _active_proc is not None and _active_proc.poll() is None


def _arch_and_config_text(metrics: TrainingMetrics) -> str:
    """Return a formatted string matching training startup output, params-first."""
    arch = metrics.architecture
    cfg = metrics.config

    if not arch and not cfg:
        return "No architecture or config data available. Load a run first."

    lines = []

    if arch:
        total = arch.get('total_params', '?')
        trainable = arch.get('trainable_params', '?')
        vocab = arch.get('vocab_size', '?')
        lines += [
            "── Model Architecture ──────────────────────────────",
            f"  Total params:     {total:,}" if isinstance(total, int) else f"  Total params:     {total}",
            f"  Trainable params: {trainable:,}" if isinstance(trainable, int) else f"  Trainable params: {trainable}",
            f"  Vocab size:       {vocab:,}" if isinstance(vocab, int) else f"  Vocab size:       {vocab}",
            f"  Layers:           {arch.get('n_layers', '?')}",
            f"  Attention heads:  {arch.get('n_heads', '?')}",
            f"  d_model:          {arch.get('d_model', '?')}",
            f"  FFN dim (d_ff):   {arch.get('d_ff', '?')}",
            f"  Context length:   {arch.get('context_length', cfg.get('context_length', '?'))}",
        ]

    if cfg:
        if lines:
            lines.append("")
        max_s = cfg.get('max_steps', '?')
        lines += [
            "── Training Config ─────────────────────────────────",
            f"  Max steps:        {max_s:,}" if isinstance(max_s, int) else f"  Max steps:        {max_s}",
            f"  Batch size:       {cfg.get('batch_size', '?')}",
            f"  Context length:   {cfg.get('context_length', '?')}",
            f"  Grad accumulation:{cfg.get('gradient_accumulation_steps', '?')}",
        ]
        if "learning_rate" in cfg:
            lines.append(f"  Learning rate:    {cfg['learning_rate']:.2e}")

    return "\n".join(lines)


# ── App builder ───────────────────────────────────────────────────────────────

def build_app(runs_dir: Path, config_path: Path) -> gr.Blocks:
    global _active_proc

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

    def generate_pdf_file(run_name):
        from src.scripts.plot_training import plot_run_dir
        _, run_dir = _selected(runs_dir, run_name)
        if run_dir is None:
            return None
        out = plot_run_dir(run_dir)
        return str(out)

    def refresh_arch(run_name):
        metrics, _ = _selected(runs_dir, run_name)
        return _arch_and_config_text(metrics), metrics.architecture or {}

    def action_start(_):
        global _active_proc
        with _proc_lock:
            _active_proc = launch_training(config_path=config_path)
        return (f"Started. PID: {_active_proc.pid}",
                gr.update(interactive=False), gr.update(interactive=True))

    def action_resume(run_name):
        global _active_proc
        if not run_name:
            return "Select a run to resume.", gr.update(), gr.update(), gr.update()
        with _proc_lock:
            _active_proc = launch_training(config_path=config_path,
                                           resume_run_dir=runs_dir / run_name)
        return (f"Resumed {run_name}. PID: {_active_proc.pid}",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=True))

    def action_stop(_):
        global _active_proc
        with _proc_lock:
            if _active_proc is not None:
                kill_training(_active_proc)
                _active_proc = None
        return "Stopped.", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)

    def refresh_status():
        alive = _is_alive()
        pid = _active_proc.pid if alive and _active_proc else None
        status = f"● Running — PID {pid}" if alive else "○ Idle"
        return status, gr.update(interactive=not alive), gr.update(interactive=alive)

    choices = _run_choices(runs_dir)

    _theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate")
    _css = ".gradio-container { max-width: 1400px; margin: auto; }"

    with gr.Blocks(title="ParrotLLM Training Dashboard") as demo:

        gr.Markdown("# ParrotLLM Training Dashboard")

        with gr.Tabs():

            # ── TAB 1: Live Monitor ───────────────────────────────────
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

            # ── TAB 2: Architecture ───────────────────────────────────
            with gr.Tab("Architecture"):
                arch_run_selector = gr.Dropdown(label="Run", choices=choices)
                arch_load_btn = gr.Button("Load")
                arch_box = gr.Textbox(label="Architecture Summary",
                                      lines=10, elem_id="arch_box")
                with gr.Accordion("Raw JSON", open=False):
                    arch_json = gr.JSON(elem_id="arch_json")
                arch_load_btn.click(
                    fn=refresh_arch, inputs=[arch_run_selector],
                    outputs=[arch_box, arch_json],
                )

            # ── TAB 3: Run Manager ────────────────────────────────────
            with gr.Tab("Run Manager"):
                status_box = gr.Textbox(label="Status", value="○ Idle", interactive=False)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Start new run**")
                        start_btn = gr.Button("▶ Start Training", variant="primary")
                        start_out = gr.Textbox(label="Output", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Resume existing run**")
                        resume_selector = gr.Dropdown(label="Run to Resume", choices=choices)
                        resume_btn = gr.Button("⏩ Resume Training")
                        resume_out = gr.Textbox(label="Output", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("**Stop training**")
                        stop_btn = gr.Button("⏹ Stop Training", variant="stop",
                                             interactive=False)
                        stop_out = gr.Textbox(label="Output", lines=2, interactive=False)

                start_btn.click(fn=action_start, inputs=[start_btn],
                                outputs=[start_out, start_btn, stop_btn])
                resume_btn.click(fn=action_resume, inputs=[resume_selector],
                                 outputs=[resume_out, resume_btn, start_btn, stop_btn])
                stop_btn.click(fn=action_stop, inputs=[stop_btn],
                               outputs=[stop_out, start_btn, resume_btn, stop_btn])

                gr.Markdown("### All Runs")
                runs_table = gr.Dataframe(
                    headers=["Run", "Last Step", "Best Val Loss", "Status"],
                    value=[[r.name, str(r.last_step or "—"),
                            f"{r.best_val_loss:.4f}" if r.best_val_loss else "—",
                            "unknown"]
                           for r in list_runs(runs_dir)],
                    label="Runs",
                )

                status_timer = gr.Timer(value=5)
                status_timer.tick(fn=refresh_status, inputs=[],
                                  outputs=[status_box, start_btn, stop_btn])

    return demo


def run_dashboard(
    runs_dir: Path,
    config_path: Path,
    port: int = 7861,
    share: bool = False,
    open_browser: bool = False,
) -> None:
    demo = build_app(runs_dir, config_path)
    if open_browser:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
