"""Diagnostic: replicate the chat UI pipeline and inspect raw outputs.

Run via: uv run python tools/debug_chat.py
"""

from __future__ import annotations

import torch

from src.eval.inference import generate, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer


CKPTS = {
    "DPO_best (0.0368)": r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt",
    "SFT_best_radio (2.6646)": r"runs/run_20260426_202050_sft/checkpoints/best_step_0001000_epoch_01_valloss_2p6646.pt",
    "SFT_actual_DPO_base (2.7198)": r"runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt",
}

PROMPTS = [
    "What is the capital of france?",
    "What is the capital of France?",
]


def show_tokens(tokenizer, ids):
    pieces = [tokenizer.decode([t]) for t in ids]
    return " | ".join(repr(p) for p in pieces)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer()
    print(f"Tokenizer: vocab={len(tok)}, eos={tok.eos_token!r} (id={tok.eos_token_id}), "
          f"pad={tok.pad_token!r} (id={tok.pad_token_id})")

    # Verify: tokenizer.encode(...) vs tokenizer(..., add_special_tokens=False).input_ids
    sample = format_sft_prompt(PROMPTS[0])
    a = tok.encode(sample)
    b = tok(sample, add_special_tokens=False).input_ids
    print(f"\nencode parity (chat vs SFT-data path): equal={a == b}, "
          f"len(encode)={len(a)} len(SFT)={len(b)}")
    if a != b:
        print(f"  encode head 5: {a[:5]} | SFT head 5: {b[:5]}")
        print(f"  encode tail 5: {a[-5:]} | SFT tail 5: {b[-5:]}")

    print(f"\nPrompt rendered to model (last 80 chars):")
    print(f"  ...{sample[-80:]!r}")

    for label, path in CKPTS.items():
        print("\n" + "=" * 78)
        print(f"### {label}")
        print(f"### {path}")
        print("=" * 78)
        model, cfg = load_model_from_checkpoint(path, device)
        ctx = cfg["model"]["context_length"]

        for prompt in PROMPTS:
            full = format_sft_prompt(prompt)
            ids = tok.encode(full)
            x = torch.tensor([ids], dtype=torch.long, device=device)

            for temp in (0.0, 0.5):
                out = generate(model, x, max_new_tokens=60,
                               temperature=temp, top_k=50, top_p=0.9,
                               context_length=ctx,
                               eos_token_id=tok.eos_token_id)
                gen_ids = out[0, len(ids):].tolist()
                # Find first EOS / "###" emergence in raw generation
                eos_pos = gen_ids.index(tok.eos_token_id) if tok.eos_token_id in gen_ids else -1
                gen_text = tok.decode(gen_ids)

                print(f"\n[{prompt!r}] temp={temp}")
                print(f"  raw decoded:  {gen_text!r}")
                print(f"  first EOS at: {eos_pos}/{len(gen_ids)} "
                      f"(EOS id={tok.eos_token_id})")
                # Show first 8 tokens individually
                head_n = min(8, len(gen_ids))
                print(f"  first {head_n} tokens: {show_tokens(tok, gen_ids[:head_n])}")


if __name__ == "__main__":
    main()
