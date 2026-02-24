"""Autoregressive generation and leaderboard inference."""

import sys

import torch
from transformers import AutoTokenizer

from src.model import ParrotLLM
from src.utils import get_device, load_config, set_seed


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    context_length: int = 1024,
) -> torch.Tensor:
    """Autoregressive generation. temp=0 for greedy, temp>0 for sampling."""
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (B, vocab)

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ParrotLLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def run_inference(args) -> None:
    config = load_config(args.config)
    ic = config.get("inference", {})
    leaderboard = getattr(args, "leaderboard", False)

    seed = getattr(args, "seed", ic.get("seed", 42))
    set_seed(seed)

    device = get_device(getattr(args, "device", ic.get("device", "auto")))
    assert args.checkpoint, "--checkpoint required for inference"

    model, ckpt_config = load_model_from_checkpoint(args.checkpoint, device)
    mc = ckpt_config["model"]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = args.prompt or "The meaning of life is"
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    max_tokens = getattr(args, "max_tokens", ic.get("max_tokens", 128))
    temperature = getattr(args, "temperature", ic.get("temperature", 0.0))
    top_k = ic.get("top_k", 50)
    top_p = ic.get("top_p", 0.9)

    output = generate(
        model, idx, max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        context_length=mc["context_length"],
    )
    text = tokenizer.decode(output[0].tolist())

    if leaderboard:
        # leaderboard mode: ONLY generated text, no logging
        generated = tokenizer.decode(output[0, len(input_ids):].tolist())
        sys.stdout.write(generated)
    else:
        print(f"[inference] prompt: {prompt}")
        print(f"[inference] output: {text}")
