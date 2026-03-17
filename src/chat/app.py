"""Gradio chat interface for ParrotLLM."""

import glob
import logging
import os

import torch

log = logging.getLogger("parrotllm.chat")

from configs import ProjectConfig
from src.eval.inference import generate, load_model_from_checkpoint
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
        return sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))

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

        # build context from history
        context = ""
        for user_msg, bot_msg in history:
            context += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        context += f"User: {message}\nAssistant:"

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
        )
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        # stop at next "User:" turn
        if "User:" in generated:
            generated = generated[:generated.index("User:")]
        return generated.strip()

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
