"""Gradio chat interface for ParrotLLM."""

import glob
import logging
import os

import torch

log = logging.getLogger("parrotllm.chat")

from configs import ProjectConfig
from src.eval.inference import generate, load_model_from_checkpoint
from src.posttraining.templates import build_generation_prompt, strip_generated_assistant_text
from src.utils import build_tokenizer


def run_chat(project_config: ProjectConfig, *, device: torch.device) -> None:
    import gradio as gr

    chat_cfg = project_config.chat
    if chat_cfg is None:
        raise ValueError("Chat configuration missing; cannot start chat UI.")
    tokenizer = build_tokenizer()

    state = {"model": None, "config": None}

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

    def load_ckpt(path):
        if not path:
            return "No checkpoint selected."
        model, ckpt_config = load_model_from_checkpoint(path, device)
        state["model"] = model
        state["config"] = ckpt_config
        n_params = model.count_parameters()
        log.info(f"Loaded checkpoint: {os.path.basename(path)} ({n_params:,} params) on {device}")
        return f"Loaded {os.path.basename(path)} ({n_params:,} params) on {device}"

    def chat_fn(message, history):
        if state["model"] is None:
            return "Please load a checkpoint first."

        mc = state["config"]["model"]

        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        context = build_generation_prompt(messages, system_prompt=chat_cfg.system_prompt)

        input_ids = tokenizer.encode(context)
        # truncate to fit context window
        max_ctx = mc["context_length"] - chat_cfg.max_tokens
        if len(input_ids) > max_ctx:
            input_ids = input_ids[-max_ctx:]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = generate(
            state["model"], idx, chat_cfg.max_tokens,
            temperature=chat_cfg.temperature,
            top_k=chat_cfg.top_k,
            top_p=chat_cfg.top_p,
            context_length=mc["context_length"],
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        return strip_generated_assistant_text(generated)

    with gr.Blocks(title="ParrotLLM Chat") as demo:
        gr.Markdown("# ParrotLLM Chat")

        with gr.Row():
            ckpt_dropdown = gr.Dropdown(
                choices=list_checkpoints(), label="Checkpoint",
                interactive=True,
            )
            load_btn = gr.Button("Load")
            status = gr.Textbox(label="Status", interactive=False)

        load_btn.click(load_ckpt, inputs=ckpt_dropdown, outputs=status)

        chatbot = gr.ChatInterface(chat_fn)

    log.info("Launching chat UI...")
    demo.launch()
