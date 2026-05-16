"""Gradio chat interface for ParrotLLM.

Detects the loaded checkpoint's `training_stage` and routes prompts
through the matching template:

- pretrain    → legacy User:/Assistant: loop with full history
- sft / dpo   → Alpaca template (`### Instruction:` / `### Response:`)
                via `format_sft_prompt` — single-turn, since Alpaca was
                trained single-turn (VL07 slide 32 critical rule)

Without this routing, an SFT/DPO checkpoint loaded into the legacy
chat code receives a prompt format it has never seen during training
and silently underperforms.
"""

import glob
import logging
import os
import re
import time

import torch

log = logging.getLogger("parrotllm.chat")

from configs import ProjectConfig
from src.eval.inference import generate_stream, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer


def run_chat(project_config: ProjectConfig, *, device: torch.device) -> None:
    import gradio as gr

    chat_cfg = project_config.chat
    if chat_cfg is None:
        raise ValueError("Chat configuration missing; cannot start chat UI.")
    tokenizer = build_tokenizer()

    state = {"model": None, "config": None, "training_stage": "pretrain"}

    def list_checkpoints():
        ckpt_dir = chat_cfg.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return []
        direct = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        recursive = glob.glob(os.path.join(ckpt_dir, "**", "*.pt"), recursive=True)
        candidates = sorted(
            set(direct + recursive),
            key=lambda path: os.path.getmtime(path),
            reverse=True,
        )
        return candidates

    def _parse_valloss(path):
        # Filenames look like best_step_0001000_epoch_01_valloss_2p6646.pt
        m = re.search(r"valloss_(\d+)p(\d+)", os.path.basename(path))
        return float(f"{m.group(1)}.{m.group(2)}") if m else float("inf")

    def _stage_from_path(path):
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if parent.endswith("_dpo"):
            return "dpo"
        if parent.endswith("_sft"):
            return "sft"
        return "pretrain"

    # ``best_*`` is the early-stopping winner; ``final_*`` is the end-of-run
    # checkpoint. v7's submission winner is ``final_*`` because end-of-run
    # (valloss 2.42) beat the early-stop best at step 900 (valloss 2.47) —
    # see docs/post_training v7_v8_results.md. Treat both as load-candidates.
    AUTO_LOAD_PREFIXES = ("best_", "final_")

    def _best_by_stage(candidates):
        """Lowest-valloss auto-load checkpoint per stage, across all runs.

        Filters out collapsed/length-bias runs whose val_loss is implausibly
        low (< 0.3). DPO v1/v2 collapsed to 0.018 — the runner would auto-
        load that and hand the user a model that emits dialog-shaped garbage.
        """
        COLLAPSE_FLOOR = 0.3
        out = {}
        for p in candidates:
            if not os.path.basename(p).startswith(AUTO_LOAD_PREFIXES):
                continue
            stage = _stage_from_path(p)
            valloss = _parse_valloss(p)
            if valloss < COLLAPSE_FLOOR:
                continue
            if stage not in out or valloss < _parse_valloss(out[stage]):
                out[stage] = p
        return out

    # Manual override: latest SFT/DPO run wins regardless of val_loss when its
    # name matches one of these explicit "v7" markers. Lets the user point
    # the chat at a specific iteration without filename gymnastics.
    PREFERRED_RUNS = (
        "run_20260428_211931_sft",   # sft_v7 (synthetic v7 mixin, 8B base)
        "run_20260428_104023_dpo",   # dpo_v6 (last good DPO)
    )

    def load_ckpt(path):
        if not path:
            return "No checkpoint selected."
        # Peek at training_stage so chat_fn knows which template to apply.
        raw = torch.load(path, map_location="cpu", weights_only=False)
        training_stage = str(raw.get("training_stage", "pretrain")).lower()
        model, ckpt_config = load_model_from_checkpoint(path, device)
        state["model"] = model
        state["config"] = ckpt_config
        state["training_stage"] = training_stage
        state["checkpoint_name"] = os.path.basename(path)
        n_params = model.count_parameters()
        msg = (
            f"Loaded {os.path.basename(path)} | stage={training_stage} | "
            f"{n_params:,} params on {device}"
        )
        log.info(msg)
        return msg

    SFT_STOP_MARKERS = ("\n###", "###", "<|endoftext|>")
    PRETRAIN_STOP_MARKER = "User:"

    def _truncate_at_stop(text: str, stage: str) -> tuple[str, bool]:
        """Apply template-specific stop-string truncation.

        Returns (cleaned_text, hit_stop). The chat UI uses the boolean to
        end streaming early once the model has emitted a marker that
        indicates the response is complete — necessary because EOS
        already terminates the generator, but stage-specific stop
        strings (e.g. ``\\n###``) can also signal "done" before EOS.
        """
        markers = SFT_STOP_MARKERS if stage in {"sft", "dpo"} else (PRETRAIN_STOP_MARKER,)
        for marker in markers:
            i = text.find(marker)
            if i >= 0:
                return text[:i], True
        return text, False

    def chat_fn(
        message, history,
        temperature, top_p, top_k, repetition_penalty,
        max_tokens, no_repeat_ngram, system_prompt,
    ):
        """Streaming chat callback.

        Yields the running response after every generated token. Gradio's
        ChatInterface forwards each yielded string to the chat bubble,
        producing the token-by-token typing effect. The bottom of the
        message gets a small TTFT/throughput line (VL09 slide 10).
        """
        if state["model"] is None:
            yield "Please load a checkpoint first."
            return

        mc = state["config"]["model"]
        stage = state.get("training_stage", "pretrain")
        max_tokens = int(max_tokens)
        temperature = float(temperature)
        top_p = float(top_p)
        top_k = int(top_k)
        repetition_penalty = float(repetition_penalty)
        no_repeat_ngram = int(no_repeat_ngram)
        system_prompt = (system_prompt or "").strip()

        log.info(
            f"[gen] ckpt={state.get('checkpoint_name', '?')} stage={stage} | "
            f"T={temperature:.2f} top_p={top_p:.2f} top_k={top_k} "
            f"rep_pen={repetition_penalty:.2f} max_tok={max_tokens} "
            f"no_rep_ngram={no_repeat_ngram} | "
            f"sys_prompt={system_prompt[:60]!r}{'...' if len(system_prompt) > 60 else ''} | "
            f"msg={message[:60]!r}{'...' if len(message) > 60 else ''}"
        )

        # Thread the (optional) system prompt as a prefix to the CURRENT user
        # message's instruction. History stays untouched, so editing the
        # system prompt mid-conversation only affects the next response —
        # exactly what a TA testing edge cases wants.
        message_with_system = (
            f"{system_prompt}\n\n{message}" if system_prompt else message
        )

        if stage in {"sft", "dpo"}:
            # Multi-turn for the conference demo (factsheet §4.5: "new
            # prompt = current conversation (context) + user text input").
            # Each completed turn is rendered as a full Alpaca block
            # (preamble + ### Instruction + ### Response + answer) so
            # every individual turn is byte-identical to training format
            # (VL07 slide 32); the multi-turn stacking is itself OOD,
            # but the per-turn template parity is preserved.
            parts = []
            for h in history:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    parts.append(format_sft_prompt(content))
                elif role == "assistant":
                    # Strip the trailing "_TTFT ... tokens_" footer added
                    # in this same function; it's UI metadata, not part
                    # of what the model actually produced.
                    clean = re.sub(r"\n*_TTFT[^_]*_\s*$", "", content).strip()
                    parts.append(clean + "\n\n")
            parts.append(format_sft_prompt(message_with_system))
            prompt = "".join(parts)
        else:
            prompt = ""
            for h in history:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += f"User: {message_with_system}\nAssistant:"

        input_ids = tokenizer.encode(prompt)
        max_ctx = mc["context_length"] - max_tokens
        if len(input_ids) > max_ctx:
            input_ids = input_ids[-max_ctx:]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        eos_id = tokenizer.eos_token_id if stage in {"sft", "dpo"} else None

        # VL09 slide 10: TTFT (prompt → first token) and tokens/sec.
        t_start = time.perf_counter()
        ttft: float | None = None

        gen_ids: list[int] = []
        for tok_id in generate_stream(
            state["model"], idx, max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            context_length=mc["context_length"],
            eos_token_id=eos_id,
            no_repeat_ngram_size=int(no_repeat_ngram),
        ):
            if ttft is None:
                ttft = time.perf_counter() - t_start
            gen_ids.append(tok_id)
            # Re-decode the full id sequence each step. BPE tokens can
            # represent partial UTF-8 bytes, so per-token decoding may
            # yield replacement chars; full re-decode resolves them.
            text = tokenizer.decode(gen_ids)
            cleaned, hit_stop = _truncate_at_stop(text, stage)
            yield cleaned.strip()
            if hit_stop:
                break

        elapsed = time.perf_counter() - t_start
        n_decoded = max(len(gen_ids) - 1, 1)
        tps = n_decoded / max(elapsed - (ttft or 0.0), 1e-6)
        footer = (
            f"\n\n_TTFT {1000 * (ttft or 0):.0f} ms · "
            f"{tps:.1f} tok/s · {len(gen_ids)} tokens_"
        )
        final_text, _ = _truncate_at_stop(tokenizer.decode(gen_ids), stage)
        yield final_text.strip() + footer

    available = list_checkpoints()
    # If the config pins a checkpoint outside checkpoint_dir, the recursive
    # glob misses it. Hoist it into the dropdown so it's still selectable.
    cfg_pin = getattr(chat_cfg, "preferred_checkpoint", None)
    if cfg_pin is not None:
        cfg_pin_str = str(cfg_pin)
        if os.path.isfile(cfg_pin_str) and cfg_pin_str not in available:
            available.insert(0, cfg_pin_str)
    best_per_stage = _best_by_stage(available)

    # Prefer an explicitly-marked run if available — overrides val_loss
    # heuristics. Picks the lowest-valloss best_* inside that run.
    def _best_in_run(run_name):
        cands = [p for p in available
                 if run_name in p
                 and os.path.basename(p).startswith(AUTO_LOAD_PREFIXES)]
        if not cands:
            return None
        return min(cands, key=_parse_valloss)

    # Highest priority: explicit pin via chat.preferred_checkpoint in YAML.
    # Lets the team swap the demo target without touching any code.
    preferred_default = None
    if cfg_pin is not None:
        cfg_pin_str = str(cfg_pin)
        if os.path.isfile(cfg_pin_str):
            preferred_default = cfg_pin_str
            log.info(f"chat.preferred_checkpoint: {cfg_pin_str}")
        else:
            log.warning(
                f"chat.preferred_checkpoint not found on disk: {cfg_pin_str} "
                f"— falling back to auto-discovery."
            )

    # Second priority: explicit run-name markers in PREFERRED_RUNS.
    if preferred_default is None:
        for run_name in PREFERRED_RUNS:
            preferred_default = _best_in_run(run_name)
            if preferred_default:
                break

    # Named demo entries from chat.demo_checkpoints — rendered as labeled
    # buttons in the sidebar so the live demo can flip between team
    # checkpoints by name ("Cyril & Christof" / "Gian & Tilman") instead
    # of raw paths. Hoist any missing entries into `available` so the
    # full dropdown also exposes them.
    demo_choices: list[tuple[str, str]] = []
    for entry in (chat_cfg.demo_checkpoints or []):
        entry_path = str(entry.path)
        if not os.path.isfile(entry_path):
            log.warning(
                f"chat.demo_checkpoints: '{entry.name}' path not found, "
                f"skipping: {entry_path}"
            )
            continue
        demo_choices.append((entry.name, entry_path))
        if entry_path not in available:
            available.insert(0, entry_path)
    if demo_choices and preferred_default is None:
        preferred_default = demo_choices[0][1]

    default_ckpt = (
        preferred_default
        or best_per_stage.get("dpo")
        or best_per_stage.get("sft")
        or (available[0] if available else None)
    )
    initial_status = "No checkpoint found in runs/."
    if default_ckpt:
        log.info(f"Auto-loading default checkpoint: {default_ckpt}")
        initial_status = load_ckpt(default_ckpt)

    quick_choices = []
    for stage_key, label in (("dpo", "Best DPO"), ("sft", "Best SFT")):
        p = best_per_stage.get(stage_key)
        if p:
            quick_choices.append((f"{label} (valloss {_parse_valloss(p):.4f})", p))

    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    )

    custom_css = """
    .gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f766e 50%, #115e59 100%);
        border-radius: 18px; padding: 22px 26px; color: #ecfdf5;
        box-shadow: 0 8px 30px rgba(6,78,59,0.25);
    }
    .hero h1 { color: #ecfdf5 !important; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.01em; }
    .hero svg { width: 100%; height: 100%; display: block; }
    .hero .tagline { color: #a7f3d0; font-size: 14px; margin-top: 4px; }
    .hero .pill {
        display: inline-block; background: rgba(255,255,255,0.12);
        backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.18);
        padding: 4px 10px; border-radius: 999px; font-size: 12px;
        margin-right: 6px; color: #ecfdf5;
    }
    .sidebar-card {
        background: var(--block-background-fill); border-radius: 14px;
        padding: 14px; border: 1px solid var(--border-color-primary);
    }
    .status-pill {
        font-family: var(--font-mono); font-size: 12px;
        padding: 10px 12px; border-radius: 10px;
        background: var(--background-fill-secondary);
        border-left: 3px solid #10b981; word-break: break-all;
    }
    footer { display: none !important; }
    .gradio-container .prose { font-size: 14px; }
    """

    # Inline the SVG with the opaque background <rect> stripped, so the logo
    # sits flush on the hero's gradient (no white/black square around it).
    _logo_svg_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "docs", "logos", "black", "parrotlabs_logo_black_bg.svg",
    )
    try:
        with open(_logo_svg_path, "r") as _f:
            _logo_svg = _f.read()
        _logo_svg = re.sub(
            r'<rect[^/]*fill="#0{3,6}"[^/]*/>', "", _logo_svg, count=1
        )
    except OSError:
        _logo_svg = ""  # fall back gracefully if the logo file is missing

    with gr.Blocks(title="ParrotLLM · Chat") as demo:
        # ── Hero header ───────────────────────────────────────────────
        with gr.Row(elem_classes="hero"):
            with gr.Column(scale=0, min_width=110):
                gr.HTML(
                    f"<div style='width:84px;height:84px;display:flex;"
                    f"align-items:center;justify-content:center'>"
                    f"<div style='width:84px;height:84px'>{_logo_svg}</div>"
                    f"</div>"
                )
            with gr.Column(scale=1):
                gr.HTML(
                    "<h1>ParrotLLM</h1>"
                    "<div class='tagline'>40M-parameter LLM, pretrained from scratch · "
                    "instruction-tuned · ParrotLabs FS26</div>"
                    "<div style='margin-top:10px'>"
                    "<span class='pill'>🥇 #1 on PikoGPT Leaderboard</span>"
                    "<span class='pill'>33.6% public_avg</span>"
                    "<span class='pill'>Team ParrotLLM</span>"
                    "</div>"
                )

        # ── Body: sidebar + chat ──────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes="sidebar-card"):
                    gr.Markdown("### Model")
                    status = gr.Markdown(
                        value=f"<div class='status-pill'>{initial_status}</div>",
                    )
                    if demo_choices:
                        demo_radio = gr.Radio(
                            choices=demo_choices,
                            value=default_ckpt if default_ckpt in [p for _, p in demo_choices] else demo_choices[0][1],
                            label="Team checkpoint",
                        )
                    # Hide the generic "Best DPO / Best SFT" quick-load when team
                    # demo checkpoints are configured — those buttons scan the full
                    # runs/ folder and would mislead a demo viewer into thinking the
                    # globally-best checkpoints belong to the currently-selected team.
                    if quick_choices and not demo_choices:
                        quick = gr.Radio(
                            choices=quick_choices, value=default_ckpt,
                            label="Quick load",
                        )

                with gr.Accordion("Sampling", open=True):
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=chat_cfg.temperature,
                        step=0.05, label="Temperature",
                        info="0 = deterministic. Higher = more creative.",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=chat_cfg.top_p, step=0.05,
                        label="Top-p",
                        info="Nucleus cap on cumulative probability.",
                    )
                    top_k_slider = gr.Slider(
                        minimum=0, maximum=200, value=chat_cfg.top_k, step=1,
                        label="Top-k",
                        info="Hard cap on candidate tokens (0 = off).",
                    )
                    rep_pen_slider = gr.Slider(
                        minimum=1.0, maximum=2.0, value=chat_cfg.repetition_penalty,
                        step=0.05, label="Repetition penalty",
                        info="Discourages repeats (1.0 = off).",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=16, maximum=400, value=chat_cfg.max_tokens, step=8,
                        label="Max new tokens",
                    )
                    ngram_slider = gr.Slider(
                        minimum=0, maximum=6, value=3, step=1,
                        label="No-repeat n-gram",
                        info="Hard-bans repeated n-grams (0 = off).",
                    )

                # System prompt — editable at runtime. Threaded into each
                # current-turn instruction; history stays untouched, so
                # changes only affect the NEXT response. Empty box = no
                # system prompt injected (model sees only the user message).
                with gr.Accordion("System prompt", open=True):
                    system_prompt_box = gr.Textbox(
                        value=chat_cfg.system_prompt,
                        label="Prefix prepended to each user message",
                        info=(
                            "Edit any time; takes effect on the next message. "
                            "Clear the box to send the user message with no prefix."
                        ),
                        lines=3,
                        max_lines=6,
                    )
                    reset_system_btn = gr.Button(
                        "Reset to default", size="sm", variant="secondary"
                    )

                with gr.Accordion("Checkpoint", open=False):
                    ckpt_dropdown = gr.Dropdown(
                        choices=available, value=default_ckpt,
                        label="Path", interactive=True,
                    )
                    load_btn = gr.Button("Load", variant="primary", size="sm")

            with gr.Column(scale=3):
                gr.ChatInterface(
                    chat_fn,
                    additional_inputs=[
                        temp_slider, top_p_slider, top_k_slider, rep_pen_slider,
                        max_tokens_slider, ngram_slider, system_prompt_box,
                    ],
                    chatbot=gr.Chatbot(
                        height=620,
                        layout="bubble",
                        buttons=["copy"],
                        placeholder=(
                            "<div style='text-align:center; padding:40px 20px; opacity:0.6'>"
                            "<div style='font-size:42px'>🦜</div>"
                            "<div style='font-size:16px; margin-top:8px; font-weight:500'>"
                            "Say hello to ParrotLLM</div>"
                            "</div>"
                        ),
                    ),
                    # Prompts chosen for what a 40M-param model can actually do well:
                    # one-step creative/continuation/transformation tasks.
                    # Avoided: factual recall (Faust author), arithmetic (12×17),
                    # "list N distinct things" (the model repeats items).
                    examples=[
                        ["Continue this story: The old lighthouse keeper walked down to the shore and"],
                        ["Write a short poem about autumn rain."],
                        ["Write a friendly email opening to a new colleague."],
                        ["Summarize in one sentence: The cat sat lazily on the warm windowsill while the rain fell outside."],
                        ["Write a short bedtime story about a sleepy rabbit."],
                        ["What is the capital of France?"],
                    ],
                )

        load_btn.click(
            lambda p: f"<div class='status-pill'>{load_ckpt(p)}</div>",
            inputs=ckpt_dropdown, outputs=status,
        )
        reset_system_btn.click(
            lambda: chat_cfg.system_prompt,
            outputs=system_prompt_box,
        )
        if quick_choices and not demo_choices:
            def _quick_swap(path):
                return (
                    f"<div class='status-pill'>{load_ckpt(path)}</div>",
                    gr.update(value=path),
                )
            quick.change(_quick_swap, inputs=quick, outputs=[status, ckpt_dropdown])
        if demo_choices:
            def _demo_swap(path):
                return (
                    f"<div class='status-pill'>{load_ckpt(path)}</div>",
                    gr.update(value=path),
                )
            demo_radio.change(
                _demo_swap, inputs=demo_radio, outputs=[status, ckpt_dropdown],
            )

    log.info("Launching chat UI...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False,
        allowed_paths=["docs/logos"],
        theme=theme, css=custom_css,
    )
