#!/usr/bin/env python3
"""Minimal leaderboard submission entrypoint for ParrotLLM."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from src.model.transformer import ParrotLLM


DEFAULT_TOKENIZER_NAME = "openai-community/gpt2"

# Must match the system_prompt used during SFT/DPO training
# (configs/posttraining/*.yaml). Course rule (VL07 slide 32):
# the template at training MUST match the template at inference.
DEFAULT_SYSTEM_PROMPT = "You are ParrotLLM, a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ParrotLLM leaderboard submission")
    parser.add_argument("--stage", required=True, choices=["inference"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--template",
        choices=["alpaca", "raw"],
        default="alpaca",
        help="alpaca: wrap prompt in ### Instruction/### Response (matches SFT/DPO training). "
        "raw: pass prompt unchanged (use only for base-model checkpoints).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to user content under alpaca template. "
        "Must match the value used during training.",
    )
    return parser.parse_args()


def alpaca_wrap(user_prompt: str, system_prompt: str) -> str:
    """Wrap a raw prompt in the Alpaca template used at training time.

    Mirrors src/posttraining/templates.py:render_conversation(template_format='alpaca')
    with add_generation_prompt=True. The system prompt is folded into the first
    user message exactly as normalize_messages() does it.
    """
    sys = (system_prompt or "").strip()
    user = user_prompt if user_prompt is not None else ""
    if sys:
        user = f"{sys}\n\n{user}"
    return f"### Instruction:\n{user}\n\n### Response:\n"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(submission_dir: Path) -> GPT2TokenizerFast:
    local_candidates = [
        submission_dir / "tokenizer",
        submission_dir / "src" / "tokenizer",
        submission_dir / "assets" / "tokenizer",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return GPT2TokenizerFast.from_pretrained(candidate, use_fast=True)
    return GPT2TokenizerFast.from_pretrained(DEFAULT_TOKENIZER_NAME, use_fast=True)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ParrotLLM, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict) or "model" not in raw_config:
        raise ValueError("Checkpoint must contain checkpoint['config']['model'].")

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain checkpoint['model'] state dict.")

    config = {"model": raw_config["model"]}
    model = ParrotLLM(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    context_length: int,
    eos_token_id: int | None = None,
    allowed_first_token_ids: list[int] | None = None,
) -> torch.Tensor:
    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if step == 0 and allowed_first_token_ids:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed_first_token_ids] = 0.0
            logits = logits + mask

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = sorted_logits.softmax(dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = cumulative_probs - sorted_probs > top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break

    return idx


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    submission_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)
    model, config = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer(submission_dir)

    raw_prompt = args.prompt if args.prompt is not None else "The answer is"
    if args.template == "alpaca":
        prompt = alpaca_wrap(raw_prompt, args.system_prompt)
    else:
        prompt = raw_prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Prompt is empty and tokenizer has no EOS token.")
        input_ids = [eos_id]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = generate(
        model,
        idx,
        max_new_tokens=max(0, int(args.max_tokens)),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        context_length=int(config["model"].get("context_length", 1024)),
    )

    generated_ids = output[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(
        generated_ids,
        clean_up_tokenization_spaces=False,
    )

    if args.leaderboard:
        sys.stdout.write(generated_text)
        sys.stdout.flush()
        return

    full_text = tokenizer.decode(output[0].tolist(), clean_up_tokenization_spaces=False)
    print(full_text)


if __name__ == "__main__":
    main()
