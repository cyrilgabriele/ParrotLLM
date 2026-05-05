"""Direct hands-on probe: prof-style verdict on whether the SFT/DPO chain
is at the scale floor or has fixable problems.

Tests four capability buckets and prints honest pass/fail per stage:
  1. Format compliance     — does it follow Alpaca / terminate
  2. Common factual recall — capitals, basic facts (overrepresented in pretrain)
  3. Instruction following — list/translate/define (no factual hook)
  4. Repetition resistance — does the loop pattern beat the model

Run: uv run python -m tools.verdict_probe
"""

from __future__ import annotations

import torch

from src.eval.inference import generate_stream, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer

PIPELINE = [
    ("pretrain",
     r"runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt"),
    ("sft",
     r"runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt"),
    ("dpo",
     r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt"),
]

# (bucket, prompt, expected substring or None)
PROBES = [
    # 1. Common factual recall (high-frequency triplets)
    ("fact", "What is the capital of France?",        "paris"),
    ("fact", "What is the capital of Japan?",         "tokyo"),
    ("fact", "What is the capital of Germany?",       "berlin"),
    ("fact", "What is the capital of Italy?",         "rome"),
    ("fact", "What is the capital of Spain?",         "madrid"),
    ("fact", "What is the capital of Russia?",        "moscow"),
    ("fact", "What color is the sky on a clear day?", "blue"),
    ("fact", "What color is grass?",                  "green"),
    ("fact", "How many days in a week?",              "7"),
    ("fact", "How many planets in our solar system?", "8"),
    # 2. Simple math
    ("math", "What is 2 plus 2?",                     "4"),
    ("math", "What is 10 minus 3?",                   "7"),
    # 3. Instruction-following
    ("instr", "List three primary colors.",           None),
    ("instr", "Name three animals.",                  None),
    ("instr", "Write a short greeting.",              None),
    # 4. Loop-trap prompts (likely to repeat)
    ("loop", "What is the meaning of life?",          None),
    ("loop", "Tell me a joke.",                       None),
]


def run_one(model, tok, ctx, device, stage, prompt):
    if stage == "pretrain":
        text = prompt + "\n"
        eos = None
    else:
        text = format_sft_prompt(prompt)
        eos = tok.eos_token_id
    ids = tok.encode(text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    for tid in generate_stream(
        model, x, max_new_tokens=80,
        temperature=0.0, top_k=50, top_p=0.9,
        context_length=ctx, eos_token_id=eos,
        no_repeat_ngram_size=3,
    ):
        out_ids.append(tid)
    raw = tok.decode(out_ids)
    txt = raw
    for marker in ("\n###", "###", "<|endoftext|>"):
        i = txt.find(marker)
        if i >= 0:
            txt = txt[:i]
            break
    return len(out_ids), txt.strip(), len(out_ids) < 80  # third = terminated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer()

    results: dict[str, list[tuple]] = {}
    for stage, ckpt in PIPELINE:
        model, cfg = load_model_from_checkpoint(ckpt, device)
        ctx = cfg["model"]["context_length"]
        results[stage] = []
        for bucket, prompt, expected in PROBES:
            n_tok, text, terminated = run_one(model, tok, ctx, device, stage, prompt)
            hit = (expected is None) or (expected.lower() in text.lower())
            results[stage].append((bucket, prompt, expected, text, n_tok, terminated, hit))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Per-prompt comparison
    print("=" * 100)
    print("HEAD-TO-HEAD (greedy temp=0, no_repeat_ngram=3, max=80)")
    print("=" * 100)
    for i, (bucket, prompt, expected, *_) in enumerate(
        [(b, p, e, None, None, None, None) for b, p, e in PROBES]
    ):
        print(f"\n[{bucket:<5}] {prompt}")
        if expected:
            print(f"  expected: '{expected}'")
        for stage, _ in PIPELINE:
            _, _, _, text, n_tok, terminated, hit = results[stage][i]
            mark = "OK" if hit and (expected or terminated) else ("--" if expected else "  ")
            term = "TERM" if terminated else "MAX "
            print(f"  {stage:<8} [{term} {n_tok:>2}t] {mark} {text!r}")

    # Per-bucket aggregates
    print("\n" + "=" * 100)
    print("BUCKET AGGREGATES (% correct / terminated cleanly)")
    print("=" * 100)
    buckets = ["fact", "math", "instr", "loop"]
    print(f"{'bucket':<6} " + "  ".join(f"{s:>16}" for s, _ in PIPELINE))
    for b in buckets:
        line = f"{b:<6} "
        for stage, _ in PIPELINE:
            rows = [r for r in results[stage] if r[0] == b]
            n = len(rows)
            hits = sum(1 for r in rows if r[6]) / n
            terms = sum(1 for r in rows if r[5]) / n
            line += f"  hit={hits:>4.0%} term={terms:>4.0%}"
        print(line)


if __name__ == "__main__":
    main()
