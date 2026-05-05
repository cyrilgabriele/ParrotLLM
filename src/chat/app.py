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

    def _best_by_stage(candidates):
        """Lowest-valloss ``best_*`` checkpoint per stage, across all runs.

        Filters out collapsed/length-bias runs whose val_loss is implausibly
        low (< 0.3). DPO v1/v2 collapsed to 0.018 — the runner would auto-
        load that and hand the user a model that emits dialog-shaped garbage.
        """
        COLLAPSE_FLOOR = 0.3
        out = {}
        for p in candidates:
            if not os.path.basename(p).startswith("best_"):
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
        max_tokens, no_repeat_ngram,
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

        if stage in {"sft", "dpo"}:
            # Alpaca was trained single-turn — discard chat history and
            # render only the current instruction. Multi-turn Alpaca
            # would silently violate VL07 slide 32 (train ↔ inference
            # template parity).
            prompt = format_sft_prompt(message)
        else:
            prompt = ""
            for h in history:
                role = h.get("role")
                content = h.get("content", "")
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += f"User: {message}\nAssistant:"

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
    best_per_stage = _best_by_stage(available)

    # Prefer an explicitly-marked run if available — overrides val_loss
    # heuristics. Picks the lowest-valloss best_* inside that run.
    def _best_in_run(run_name):
        cands = [p for p in available
                 if run_name in p and os.path.basename(p).startswith("best_")]
        if not cands:
            return None
        return min(cands, key=_parse_valloss)

    preferred_default = None
    for run_name in PREFERRED_RUNS:
        preferred_default = _best_in_run(run_name)
        if preferred_default:
            break

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
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    .sidebar-card { background: var(--block-background-fill); border-radius: 12px;
                    padding: 12px; }
    """

    with gr.Blocks(title="ParrotLLM Chat") as demo:
        gr.Markdown("## ParrotLLM Chat — instruction-tuned PikoGPT (40M)")

        with gr.Row():
            # ── Sidebar: model + sampling ───────────────────────────
            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes="sidebar-card"):
                    status = gr.Textbox(
                        label="Loaded checkpoint",
                        interactive=False,
                        value=initial_status,
                        lines=3,
                        max_lines=4,
                    )
                    if quick_choices:
                        quick = gr.Radio(
                            choices=quick_choices, value=default_ckpt,
                            label="Quick load (best val loss)",
                        )

                with gr.Accordion("Generation", open=True):
                    # VL09 slide 30 PikoGPT defaults: τ=0.8, top-p=0.9,
                    # rep.pen=1.1. ChatConfig values feed the initial
                    # values so a config override propagates here.
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=chat_cfg.temperature,
                        step=0.05, label="Temperature (τ)",
                        info="0 = greedy. PikoGPT default: 0.8.",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=chat_cfg.top_p, step=0.05,
                        label="Top-p (nucleus)",
                        info="Adaptive cap on cumulative probability. VL09 default: 0.9.",
                    )
                    top_k_slider = gr.Slider(
                        minimum=0, maximum=200, value=chat_cfg.top_k, step=1,
                        label="Top-k",
                        info="Hard cap on candidates. 0 = off. VL09 default: 50.",
                    )
                    rep_pen_slider = gr.Slider(
                        minimum=1.0, maximum=2.0, value=chat_cfg.repetition_penalty,
                        step=0.05, label="Repetition penalty (θ)",
                        info="VL09 slide 25. 1.0 = off; PikoGPT default: 1.1.",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=16, maximum=400, value=120, step=8,
                        label="Max new tokens",
                    )
                    ngram_slider = gr.Slider(
                        minimum=0, maximum=6, value=3, step=1,
                        label="No-repeat n-gram",
                        info="0 = off. 3 = forbids repeating any 3-token sequence — kills 'blue, with a blue, with a blue' loops.",
                    )

                with gr.Accordion("Advanced — manual checkpoint pick", open=False):
                    ckpt_dropdown = gr.Dropdown(
                        choices=available, value=default_ckpt,
                        label="Checkpoint", interactive=True,
                    )
                    load_btn = gr.Button("Load", size="sm")

                gr.Markdown(
                    "_Stages: **pretrain** (raw text continuation) · "
                    "**sft** / **dpo** (Alpaca single-turn). The template is "
                    "picked from the checkpoint's `training_stage` field — "
                    "VL07 slide 32._"
                )

            # ── Main: chat ──────────────────────────────────────────
            with gr.Column(scale=3):
                gr.ChatInterface(
                    chat_fn,
                    additional_inputs=[
                        temp_slider, top_p_slider, top_k_slider, rep_pen_slider,
                        max_tokens_slider, ngram_slider,
                    ],
                    chatbot=gr.Chatbot(
                        height=560,
                        layout="bubble",
                        buttons=["copy"],
                    ),
                )

        load_btn.click(load_ckpt, inputs=ckpt_dropdown, outputs=status)
        if quick_choices:
            def _quick_swap(path):
                return load_ckpt(path), gr.update(value=path)
            quick.change(_quick_swap, inputs=quick, outputs=[status, ckpt_dropdown])

    log.info("Launching chat UI...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=False,
        theme=theme, css=custom_css,
    )
