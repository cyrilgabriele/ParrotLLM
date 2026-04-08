# src/dashboard/app.py
"""ParrotLLM Training Dashboard — Gradio UI."""
from __future__ import annotations

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
    list_runs, list_subdirs, find_all_live_runs, get_latest_run_dir,
    get_log_lines, list_checkpoints, get_config_diff, RunInfo,
)
from src.dashboard.plots import build_training_figure, build_compare_figure

_SEVERITY_EMOJI = {Severity.ERROR: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}

_STATUS_BADGE = {
    "live":      ("🟢 LIVE", "#dcfce7", "#166534"),
    "completed": ("✅ Done", "#dbeafe", "#1e40af"),
    "crashed":   ("💀 Crashed", "#fee2e2", "#991b1b"),
    "stopped":   ("⏹ Stopped", "#fef3c7", "#92400e"),
    "empty":     ("📭 Empty", "#f3f4f6", "#6b7280"),
}

_CSS = """
.gradio-container { max-width: 1400px; margin: auto; }
.dark .gradio-container { background: #1e1e2e; color: #cdd6f4; }
"""

_DARK_MODE_JS = """
() => {
    const body = document.body;
    body.classList.toggle('dark');
    const isDark = body.classList.contains('dark');
    document.cookie = 'parrot_dark=' + (isDark ? '1' : '0') + ';path=/;max-age=31536000';
    return isDark ? '☀️ Light' : '🌙 Dark';
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _selected(runs_dir: Path, run_rel_path: str) -> tuple[TrainingMetrics, Optional[Path]]:
    if run_rel_path:
        run_dir = runs_dir / run_rel_path
        if run_dir.exists():
            return read_metrics(run_dir), run_dir
    latest = get_latest_run_dir(runs_dir)
    if latest is None:
        return TrainingMetrics(), None
    return read_metrics(latest), latest


def _fmt_status_banner(metrics: TrainingMetrics) -> str:
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


def _fmt_progress_detail(metrics: TrainingMetrics, run_dir: Optional[Path]) -> str:
    if not metrics.steps:
        return "<p style='color:#6b7280'>No runs found. Start training first.</p>"

    step = metrics.steps[-1]
    loss = metrics.train_losses[-1]
    lr = metrics.lrs[-1]
    max_steps = metrics.config.get("max_steps")

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

    if len(metrics.train_losses) >= 2:
        delta = metrics.train_losses[-1] - metrics.train_losses[0]
        delta_color = "#16a34a" if delta < 0 else "#dc2626"
        loss_html = (
            f"<b>Train Loss</b> {loss:.4f} "
            f"<span style='color:{delta_color};font-size:12px'>(Δ {delta:+.4f} total)</span>"
        )
    else:
        loss_html = f"<b>Train Loss</b> {loss:.4f}"

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


def _arch_and_config_text(metrics: TrainingMetrics) -> str:
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


# ── Run browser HTML builders ─────────────────────────────────────────────────

def _breadcrumb_html(current_path: str) -> str:
    parts = [p for p in current_path.split("/") if p]
    crumbs = ["📂 <b>runs</b>"]
    for p in parts:
        crumbs.append(f" / {p}")
    return (
        f"<div style='font-size:14px;padding:8px 12px;background:#f8fafc;"
        f"border:1px solid #e2e8f0;border-radius:6px;font-family:monospace'>"
        + "".join(crumbs)
        + "</div>"
    )


def _live_runs_banner(runs_dir: Path) -> str:
    live = find_all_live_runs(runs_dir)
    if not live:
        return ""
    cards = []
    for r in live:
        step_text = f"Step {r.last_step:,}" if r.last_step else "Starting..."
        max_text = f" / {r.max_steps:,}" if r.max_steps else ""
        val_text = f" · Val: {r.best_val_loss:.4f}" if r.best_val_loss else ""
        cards.append(
            f"<div style='display:inline-block;background:#dcfce7;border:1px solid #86efac;"
            f"border-radius:8px;padding:8px 14px;margin:4px;font-size:13px'>"
            f"<span style='color:#166534;font-weight:700'>🟢 LIVE</span> "
            f"<b>{r.rel_path}</b> — {step_text}{max_text}{val_text}"
            f"</div>"
        )
    return (
        "<div style='margin-bottom:8px'>"
        + "".join(cards)
        + "</div>"
    )


def _run_card_html(run: RunInfo) -> str:
    badge_text, badge_bg, badge_fg = _STATUS_BADGE.get(
        run.status, ("?", "#f3f4f6", "#6b7280"))
    step_text = f"Step {run.last_step:,}" if run.last_step is not None else "No data"
    max_text = f" / {run.max_steps:,}" if run.max_steps else ""
    val_text = f"Best val: {run.best_val_loss:.4f}" if run.best_val_loss else ""

    return (
        f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;"
        f"margin:4px 0;background:#fff'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div>"
        f"<span style='background:{badge_bg};color:{badge_fg};padding:2px 8px;"
        f"border-radius:4px;font-size:11px;font-weight:700'>{badge_text}</span>"
        f"&nbsp; <b style='font-size:14px'>{run.name}</b>"
        f"</div>"
        f"<div style='font-size:12px;color:#6b7280'>{step_text}{max_text}"
        f"{'  ·  ' + val_text if val_text else ''}</div>"
        f"</div>"
        f"</div>"
    )


def _folder_listing_html(runs_dir: Path, current_path: str) -> str:
    subdirs = list_subdirs(runs_dir, current_path)
    runs = list_runs(runs_dir, current_path)

    parts = []
    if subdirs:
        parts.append("<div style='margin:8px 0'><b style='color:#6b7280;font-size:12px'>FOLDERS</b></div>")
        for sd in subdirs:
            parts.append(
                f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;"
                f"margin:4px 0;background:#f8fafc;font-size:14px'>"
                f"📁 <b>{sd}</b></div>"
            )
    if runs:
        parts.append("<div style='margin:8px 0'><b style='color:#6b7280;font-size:12px'>RUNS</b></div>")
        for r in runs:
            parts.append(_run_card_html(r))

    if not parts:
        parts.append("<p style='color:#9ca3af'>No runs or subfolders here.</p>")

    return "\n".join(parts)


def _checkpoint_rows(run_dir: Optional[Path]) -> list[list[str]]:
    if not run_dir:
        return []
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        return [["—", "No checkpoints found", "—", "—"]]
    rows = []
    for c in ckpts:
        star = "⭐" if c.is_best else ""
        step = str(c.step) if c.step is not None else "—"
        rows.append([star, c.name, step, f"{c.size_mb:.1f}"])
    return rows


# ── App builder ───────────────────────────────────────────────────────────────

def build_app(runs_dir: Path, config_path: Path) -> gr.Blocks:

    def _nav_choices(current_path: str) -> list[str]:
        subdirs = list_subdirs(runs_dir, current_path)
        runs = list_runs(runs_dir, current_path)
        choices = []
        for sd in subdirs:
            choices.append(f"📁 {sd}")
        for r in runs:
            badge = _STATUS_BADGE.get(r.status, ("?",))[0]
            choices.append(f"{badge}  {r.name}")
        return choices

    def _run_dropdown_choices(current_path: str) -> list[str]:
        runs = list_runs(runs_dir, current_path)
        return [r.name for r in runs]

    def navigate_to(selection: str, current_path: str):
        """Handle selection: if folder, navigate into it; if run, select it."""
        if not selection:
            return current_path, "", gr.update(), gr.update(), gr.update()
        if selection.startswith("📁 "):
            folder_name = selection[2:].strip()
            new_path = f"{current_path}/{folder_name}".strip("/")
            choices = _nav_choices(new_path)
            run_choices = _run_dropdown_choices(new_path)
            return (
                new_path,
                "",  # clear selected run
                gr.update(choices=choices, value=None),
                gr.update(value=_breadcrumb_html(new_path)),
                gr.update(choices=run_choices, value=None),
            )
        # It's a run — extract name
        run_name = selection.split("  ", 1)[-1].strip()
        rel = f"{current_path}/{run_name}".strip("/")
        return (
            current_path,
            rel,
            gr.update(),
            gr.update(),
            gr.update(value=run_name),
        )

    def go_up(current_path: str):
        parts = [p for p in current_path.split("/") if p]
        if not parts:
            return current_path, gr.update(), gr.update(), gr.update()
        new_path = "/".join(parts[:-1])
        choices = _nav_choices(new_path)
        run_choices = _run_dropdown_choices(new_path)
        return (
            new_path,
            gr.update(choices=choices, value=None),
            gr.update(value=_breadcrumb_html(new_path)),
            gr.update(choices=run_choices, value=None),
        )

    def refresh_browser(current_path: str):
        choices = _nav_choices(current_path)
        run_choices = _run_dropdown_choices(current_path)
        return (
            gr.update(choices=choices, value=None),
            gr.update(value=_live_runs_banner(runs_dir)),
            gr.update(value=_folder_listing_html(runs_dir, current_path)),
            gr.update(choices=run_choices, value=None),
        )

    def select_run_from_dropdown(run_name: str, current_path: str):
        if not run_name:
            return ""
        return f"{current_path}/{run_name}".strip("/")

    def select_live_run(current_path: str):
        """Auto-select the first live run found globally."""
        live = find_all_live_runs(runs_dir)
        if live:
            return live[0].rel_path
        return ""

    def refresh_monitor(selected_run: str):
        metrics, run_dir = _selected(runs_dir, selected_run)
        stats = get_system_stats()
        log_lines = get_log_lines(run_dir) if run_dir else []
        eta = _compute_eta(metrics)
        run_label = run_dir.name if run_dir else "—"
        return (
            _fmt_status_banner(metrics),
            _fmt_progress_detail(metrics, run_dir),
            eta,
            run_label,
            f"CPU {stats.cpu_percent:.1f}%  |  RAM {stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB",
            _gpu_rows(stats),
            build_training_figure(metrics),
            "\n".join(log_lines) if log_lines else "(no log yet)",
            _checkpoint_rows(run_dir),
            _live_runs_banner(runs_dir),
        )

    def generate_pdf_file(selected_run: str):
        from src.scripts.plot_training import plot_run_dir
        _, run_dir = _selected(runs_dir, selected_run)
        if run_dir is None:
            return None
        out = plot_run_dir(run_dir)
        return str(out)

    def refresh_arch(selected_run: str):
        metrics, _ = _selected(runs_dir, selected_run)
        return _arch_and_config_text(metrics), metrics.architecture or {}, metrics.config or {}

    # ── Compare tab helpers ───────────────────────────────────────────

    def get_all_runs_for_compare():
        """Recursively list all runs for the compare multi-select."""
        all_runs = []
        for mp in sorted(runs_dir.rglob("metrics.jsonl")):
            rel = str(mp.parent.relative_to(runs_dir))
            all_runs.append(rel)
        return all_runs

    def do_compare(selected_runs: list[str]):
        if not selected_runs or len(selected_runs) < 2:
            return None, []
        dirs = [runs_dir / r for r in selected_runs]
        fig = build_compare_figure(dirs)
        diff = get_config_diff(dirs)
        rows = []
        for key, vals in diff.items():
            row = [key] + [str(vals.get(Path(r).name, "—")) for r in selected_runs]
            rows.append(row)
        return fig, rows

    # ── Build UI ──────────────────────────────────────────────────────

    initial_choices = _nav_choices("")
    initial_run_choices = _run_dropdown_choices("")
    all_compare_runs = get_all_runs_for_compare()

    with gr.Blocks(
        title="ParrotLLM Training Dashboard",
    ) as demo:

        # State
        current_path = gr.State("")
        selected_run = gr.State("")

        # Header
        with gr.Row():
            gr.Markdown("# ParrotLLM Training Dashboard")
            dark_btn = gr.Button("🌙 Dark", size="sm")

        with gr.Tabs():

            # ── TAB 1: Live Monitor ───────────────────────────────
            with gr.Tab("Live Monitor"):
                # Live runs banner
                live_banner = gr.HTML(value=_live_runs_banner(runs_dir))

                # Navigation
                with gr.Row():
                    breadcrumb = gr.HTML(value=_breadcrumb_html(""))
                with gr.Row():
                    nav_dropdown = gr.Dropdown(
                        label="Browse", choices=initial_choices,
                        scale=4, interactive=True,
                    )
                    up_btn = gr.Button("⬆ Up", scale=1, size="sm")
                    refresh_list_btn = gr.Button("🔄 Refresh", scale=1, size="sm")

                # Run selector (for currently viewed runs at this level)
                with gr.Row():
                    run_dropdown = gr.Dropdown(
                        label="Select Run", choices=initial_run_choices,
                        scale=4, interactive=True,
                    )
                    live_btn = gr.Button("🟢 Go to Live", scale=1, size="sm")

                # Folder contents display
                with gr.Accordion("Folder Contents", open=False):
                    folder_html = gr.HTML(value=_folder_listing_html(runs_dir, ""))

                # Status banner
                status_banner = gr.HTML(
                    value="<div style='background:#f3f4f6;border-radius:6px;padding:10px 16px;"
                          "font-size:14px;color:#6b7280'>Waiting for first refresh…</div>"
                )

                with gr.Row():
                    run_label = gr.Textbox(label="Active Run", scale=2, interactive=False)
                    eta_box = gr.Textbox(label="ETA", scale=1, interactive=False)

                progress_html = gr.HTML()
                plot_out = gr.Plot(label="Training Metrics")

                with gr.Accordion("System Stats", open=False):
                    sys_header_box = gr.Textbox(label="System", interactive=False)
                    gpu_table = gr.Dataframe(
                        headers=["GPU", "Name", "Mem Used (GB)", "Mem Total (GB)",
                                 "Util (%)", "Temp (°C)"],
                        label="GPU Stats",
                    )

                with gr.Accordion("Checkpoints", open=False):
                    ckpt_table = gr.Dataframe(
                        headers=["", "Checkpoint", "Step", "Size (MB)"],
                        label="Checkpoints",
                    )

                with gr.Row():
                    pdf_btn = gr.Button("⬇ Generate PDF", scale=1)
                    pdf_file = gr.File(label="Download", scale=2, interactive=False)

                with gr.Accordion("Training log (last 20 lines)", open=False):
                    log_box = gr.Textbox(lines=10, show_label=False, interactive=False)

                # Timer — fixed 1s refresh
                timer = gr.Timer(value=1)
                timer.tick(
                    fn=refresh_monitor, inputs=[selected_run],
                    outputs=[status_banner, progress_html, eta_box, run_label,
                             sys_header_box, gpu_table, plot_out, log_box,
                             ckpt_table, live_banner],
                )

                # Navigation events
                nav_dropdown.select(
                    fn=navigate_to,
                    inputs=[nav_dropdown, current_path],
                    outputs=[current_path, selected_run, nav_dropdown,
                             breadcrumb, run_dropdown],
                )
                up_btn.click(
                    fn=go_up,
                    inputs=[current_path],
                    outputs=[current_path, nav_dropdown, breadcrumb, run_dropdown],
                )
                refresh_list_btn.click(
                    fn=refresh_browser,
                    inputs=[current_path],
                    outputs=[nav_dropdown, live_banner, folder_html, run_dropdown],
                )
                run_dropdown.select(
                    fn=select_run_from_dropdown,
                    inputs=[run_dropdown, current_path],
                    outputs=[selected_run],
                )
                live_btn.click(
                    fn=select_live_run,
                    inputs=[current_path],
                    outputs=[selected_run],
                )
                pdf_btn.click(
                    fn=generate_pdf_file, inputs=[selected_run], outputs=[pdf_file],
                )

            # ── TAB 2: Compare Runs ───────────────────────────────
            with gr.Tab("Compare Runs"):
                gr.Markdown("Select 2+ runs to compare their training curves and config differences.")
                compare_select = gr.Dropdown(
                    label="Select runs to compare",
                    choices=all_compare_runs,
                    multiselect=True,
                    interactive=True,
                )
                compare_refresh_btn = gr.Button("🔄 Refresh run list")
                compare_btn = gr.Button("Compare", variant="primary")
                compare_plot = gr.Plot(label="Comparison")
                gr.Markdown("### Config Differences")
                compare_diff = gr.Dataframe(
                    headers=["Parameter"],
                    label="Config Diff",
                )

                def refresh_compare_list():
                    return gr.update(choices=get_all_runs_for_compare())

                compare_refresh_btn.click(
                    fn=refresh_compare_list, outputs=[compare_select],
                )
                compare_btn.click(
                    fn=do_compare, inputs=[compare_select],
                    outputs=[compare_plot, compare_diff],
                )

            # ── TAB 3: Architecture ───────────────────────────────
            with gr.Tab("Architecture"):
                arch_run_box = gr.Textbox(
                    label="Run (relative path)", interactive=True,
                    placeholder="e.g. run_20260408_193542 or tuning/8_75mio_model_tuning/run_...",
                )
                arch_load_btn = gr.Button("Load")
                arch_box = gr.Textbox(
                    label="Model Architecture & Training Config",
                    lines=20, interactive=False,
                )
                with gr.Accordion("Raw JSON — Architecture", open=False):
                    arch_json = gr.JSON()
                with gr.Accordion("Raw JSON — Config", open=False):
                    config_json = gr.JSON()

                def load_arch_from_path(run_path: str):
                    return refresh_arch(run_path)

                arch_load_btn.click(
                    fn=load_arch_from_path,
                    inputs=[arch_run_box],
                    outputs=[arch_box, arch_json, config_json],
                )

        # Dark mode toggle
        dark_btn.click(fn=None, js=_DARK_MODE_JS, outputs=[dark_btn])

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
    demo.launch(
        server_name="0.0.0.0", server_port=port, share=share,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=_CSS,
    )
