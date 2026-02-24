"""Gradio chat interface for ParrotLLM."""

import glob
import os

import torch
from transformers import AutoTokenizer

from src.eval.inference import generate, load_model_from_checkpoint
from src.utils import get_device, load_config


def run_chat(args) -> None:
    import gradio as gr

    config = load_config(args.config)
    cc = config.get("chat", {})
    device = get_device(cc.get("device", getattr(args, "device", "auto")))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    state = {"model": None, "config": None}

    def list_checkpoints():
        ckpt_dir = cc.get("checkpoint_dir", "checkpoints")
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
        max_ctx = mc["context_length"] - cc.get("max_tokens", 256)
        if len(input_ids) > max_ctx:
            input_ids = input_ids[-max_ctx:]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = generate(
            state["model"], idx, cc.get("max_tokens", 256),
            temperature=cc.get("temperature", 0.7),
            top_k=cc.get("top_k", 50),
            top_p=cc.get("top_p", 0.9),
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

    demo.launch()
