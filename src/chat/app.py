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

import torch

log = logging.getLogger("parrotllm.chat")

from configs import ProjectConfig
from src.eval.inference import generate, load_model_from_checkpoint
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
        """Lowest-valloss `best_*` checkpoint per stage, across all runs."""
        out = {}
        for p in candidates:
            if not os.path.basename(p).startswith("best_"):
                continue
            stage = _stage_from_path(p)
            if stage not in out or _parse_valloss(p) < _parse_valloss(out[stage]):
                out[stage] = p
        return out

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

    def chat_fn(message, history, temperature, max_tokens):
        if state["model"] is None:
            return "Please load a checkpoint first."

        mc = state["config"]["model"]
        stage = state.get("training_stage", "pretrain")
        max_tokens = int(max_tokens)
        temperature = float(temperature)

        if stage in {"sft", "dpo"}:
            # Alpaca was trained single-turn — discard chat history and
            # render only the current instruction. Multi-turn Alpaca
            # would silently violate VL07 slide 32 (train ↔ inference
            # template parity).
            prompt = format_sft_prompt(message)
        else:
            prompt = ""
            for user_msg, bot_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
            prompt += f"User: {message}\nAssistant:"

        input_ids = tokenizer.encode(prompt)
        max_ctx = mc["context_length"] - max_tokens
        if len(input_ids) > max_ctx:
            input_ids = input_ids[-max_ctx:]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        # SFT/DPO checkpoints are trained to terminate on EOS — pass it so
        # generation stops at the semantic end of the response instead of
        # running the full max_tokens budget into post-EOS garbage.
        eos_id = tokenizer.eos_token_id if stage in {"sft", "dpo"} else None
        output = generate(
            state["model"], idx, max_tokens,
            temperature=temperature,
            top_k=chat_cfg.top_k,
            top_p=chat_cfg.top_p,
            context_length=mc["context_length"],
            eos_token_id=eos_id,
        )
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())

        # Truncate at the template's natural terminator.
        if stage in {"sft", "dpo"}:
            for marker in ("\n###", "###", "<|endoftext|>"):
                idx_m = generated.find(marker)
                if idx_m >= 0:
                    generated = generated[:idx_m]
                    break
        else:
            if "User:" in generated:
                generated = generated[:generated.index("User:")]
        return generated.strip()

    available = list_checkpoints()
    best_per_stage = _best_by_stage(available)
    # DPO sits on top of SFT and inherits its CF guardrails, so DPO is the
    # default chat target; SFT is one click away via the Quick load radio.
    default_ckpt = (
        best_per_stage.get("dpo")
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

    with gr.Blocks(title="ParrotLLM Chat") as demo:
        gr.Markdown(
            "# ParrotLLM Chat\n"
            "Loads any pretrained / SFT / DPO checkpoint. The right prompt "
            "template is auto-detected from the checkpoint's `training_stage`. "
            "**Tip:** at this model scale, lower temperature (0.0–0.3) gives "
            "much more coherent answers than the default 0.7."
        )

        with gr.Row():
            ckpt_dropdown = gr.Dropdown(
                choices=available, value=default_ckpt,
                label="Checkpoint", interactive=True,
            )
            load_btn = gr.Button("Load")
            status = gr.Textbox(
                label="Status", interactive=False, value=initial_status,
            )

        load_btn.click(load_ckpt, inputs=ckpt_dropdown, outputs=status)

        if quick_choices:
            quick = gr.Radio(
                choices=quick_choices, value=default_ckpt,
                label="Quick load (best by val loss)",
            )

            def _quick_swap(path):
                return load_ckpt(path), gr.update(value=path)

            quick.change(_quick_swap, inputs=quick, outputs=[status, ckpt_dropdown])

        with gr.Row():
            temp_slider = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.3, step=0.05,
                label="Temperature (0 = deterministic / greedy; 0.3 = focused)",
            )
            max_tokens_slider = gr.Slider(
                minimum=16, maximum=400, value=120, step=8,
                label="Max new tokens",
            )

        chatbot = gr.ChatInterface(
            chat_fn,
            additional_inputs=[temp_slider, max_tokens_slider],
        )

    log.info("Launching chat UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
