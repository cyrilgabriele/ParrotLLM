"""Smoke-test that no_repeat_ngram_size=3 kills the 'List three primary
colors' repetition loop on the DPO best checkpoint."""

from __future__ import annotations

import torch

from src.eval.inference import generate_stream, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer

CKPT = r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt"


def run(prompt: str, n_gram: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer()
    model, cfg = load_model_from_checkpoint(CKPT, device)
    ctx = cfg["model"]["context_length"]

    full = format_sft_prompt(prompt)
    ids = tok.encode(full)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    gen_ids = []
    for tid in generate_stream(
        model, x, max_new_tokens=200,
        temperature=0.0, top_k=50, top_p=0.9,
        context_length=ctx, eos_token_id=tok.eos_token_id,
        no_repeat_ngram_size=n_gram,
    ):
        gen_ids.append(tid)

    text = tok.decode(gen_ids)
    for marker in ("\n###", "###", "<|endoftext|>"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
            break
    return len(gen_ids), text.strip()


def main():
    # Greedy (temp=0) reliably reproduces the loop — what user saw was a
    # sampled variant. Test the strict case so we know the blocker works.
    for prompt in ("List three primary colors.",
                   "What is a dog?",
                   "What color is the sky?"):
        for n in (0, 3):
            n_tok, text = run(prompt, n)
            print(f"\n[{prompt!r}] no_repeat={n}: {n_tok} tokens")
            print(f"  {text!r}")


if __name__ == "__main__":
    main()
