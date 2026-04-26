"""Side-by-side qualitative comparison of two ParrotLLM checkpoints.

Loads two checkpoints and generates greedy completions on a fixed set of
instruction-style prompts, so you can eyeball what SFT actually changed.

Usage:
    uv run python -m src.scripts.sft_compare \\
        --base   runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt \\
        --sft    runs/run_20260426_202050_sft/checkpoints/best_step_0000600_epoch_00_valloss_2p7040.pt \\
        --max-tokens 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.eval.inference import generate, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer, get_device


# Curated probe prompts — broad coverage of common SFT failure modes.
PROBES: list[dict] = [
    {"instruction": "What is the capital of France?"},
    {"instruction": "Explain photosynthesis in two sentences."},
    {"instruction": "Write a haiku about autumn leaves."},
    {"instruction": "Translate 'Good morning, how are you?' to French."},
    {"instruction": "List three benefits of regular exercise."},
    {"instruction": "Write a Python function that reverses a string."},
    {"instruction": "What causes rainbows?"},
    {"instruction": "Summarize World War II in one sentence."},
]


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _load(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if "model" not in cfg:
        cfg = {"model": cfg}
    from src.model import ParrotLLM
    model = ParrotLLM(cfg).to(device)
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, cfg["model"]


def _generate(model, tokenizer, prompt: str, *, max_tokens: int, ctx_len: int, device: torch.device) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, max_tokens, temperature=0.0, top_k=0, top_p=1.0,
                   context_length=ctx_len)
    completion_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(completion_ids)


def _truncate_at_response_marker(text: str) -> str:
    """SFT models terminate at '###' or EOS. Cut continuations there."""
    for marker in ("\n###", "###", "<|endoftext|>"):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].rstrip()
    return text.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="path to base (pretrained) checkpoint")
    parser.add_argument("--sft", required=True, help="path to SFT checkpoint")
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    device = get_device("auto")
    tokenizer = build_tokenizer()

    print(f"Loading base: {Path(args.base).name}")
    base_model, base_mc = _load(args.base, device)
    print(f"Loading SFT:  {Path(args.sft).name}")
    sft_model, sft_mc = _load(args.sft, device)

    print(f"\n{'='*100}\nGreedy completions, {args.max_tokens} tokens, identical Alpaca prompts\n{'='*100}")

    for i, ex in enumerate(PROBES, 1):
        prompt = format_sft_prompt(ex["instruction"])
        base_out = _generate(base_model, tokenizer, prompt, max_tokens=args.max_tokens,
                             ctx_len=base_mc["context_length"], device=device)
        sft_out = _generate(sft_model, tokenizer, prompt, max_tokens=args.max_tokens,
                            ctx_len=sft_mc["context_length"], device=device)

        print(f"\n[{i}] {ex['instruction']}")
        print("-" * 100)
        print(f"BASE: {_truncate_at_response_marker(base_out)!r}")
        print(f"SFT : {_truncate_at_response_marker(sft_out)!r}")


if __name__ == "__main__":
    main()
