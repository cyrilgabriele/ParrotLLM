"""Interactive REPL for chatting with an SFT-trained ParrotLLM checkpoint.

Uses the Alpaca template the model was trained on (VL07 slide 32 critical
rule). The Gradio chat in src/chat/app.py uses a different format and will
underperform on SFT checkpoints — use this tool instead while the chat
module catches up.

Usage:
    uv run python -m src.scripts.sft_chat \\
        --checkpoint runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt \\
        --max-tokens 120 \\
        --temperature 0.7

Type your instruction, press Enter, watch the model respond. `:q` or `Ctrl-C` exits.
Special commands:
    :temp <float>    change sampling temperature (0 = greedy)
    :max <int>       change max new tokens
    :reload          reload the model (e.g., after a new checkpoint)
    :q               quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.eval.inference import generate
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer, get_device


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _load(checkpoint_path: str, device: torch.device):
    from src.model import ParrotLLM
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if "model" not in cfg:
        cfg = {"model": cfg}
    model = ParrotLLM(cfg).to(device)
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, cfg["model"]


def _truncate_at_terminator(text: str) -> str:
    """SFT models are taught to stop at EOS or at '###'."""
    for marker in ("\n### ", "\n###", "<|endoftext|>"):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].rstrip()
    return text.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = get_device("auto")
    tokenizer = build_tokenizer()

    print(f"Loading {Path(args.checkpoint).name} on {device} ...", flush=True)
    model, mc = _load(args.checkpoint, device)
    print(f"  context_length={mc['context_length']}, params={sum(p.numel() for p in model.parameters()):,}")
    print("Ready. Type your instruction; commands: :temp <f> | :max <int> | :reload | :q")
    print()

    temp = float(args.temperature)
    max_tok = int(args.max_tokens)

    try:
        while True:
            try:
                line = input(">>> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == ":q":
                break
            if line.startswith(":temp "):
                try:
                    temp = float(line.split()[1])
                    print(f"  temperature = {temp}")
                except (IndexError, ValueError):
                    print("  usage: :temp <float>")
                continue
            if line.startswith(":max "):
                try:
                    max_tok = int(line.split()[1])
                    print(f"  max_tokens = {max_tok}")
                except (IndexError, ValueError):
                    print("  usage: :max <int>")
                continue
            if line == ":reload":
                model, mc = _load(args.checkpoint, device)
                print("  reloaded.")
                continue

            prompt = format_sft_prompt(line)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = generate(
                    model, idx, max_tok,
                    temperature=temp, top_k=args.top_k, top_p=args.top_p,
                    context_length=mc["context_length"],
                )
            completion = tokenizer.decode(out[0, len(ids):].tolist())
            print(_truncate_at_terminator(completion))
            print()
    except KeyboardInterrupt:
        pass
    print("\nbye.")


if __name__ == "__main__":
    main()
