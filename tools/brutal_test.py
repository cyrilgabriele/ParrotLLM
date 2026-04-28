"""Ruthless evaluation of the DPO best checkpoint as a real user would
encounter it. Greedy decoding (temp=0), no-repeat-ngram=3, max=80.

For each prompt I label the response: WORKS / PARTIAL / BROKEN, with a
one-line reason. Aggregates at the end.
"""

from __future__ import annotations

import torch

from src.eval.inference import generate_stream, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer

CKPT = r"runs/run_20260428_104023_dpo/checkpoints/best_step_0000360_epoch_00_valloss_0p6450.pt"

# Each row: (category, prompt, expected_substr_or_none, what_a_decent_answer_looks_like)
PROBES = [
    # CAPITALS — most-common factual triplet on the web
    ("capital",   "What is the capital of France?",            "paris"),
    ("capital",   "What is the capital of Germany?",           "berlin"),
    ("capital",   "What is the capital of Japan?",             "tokyo"),
    ("capital",   "What is the capital of Italy?",             "rome"),
    ("capital",   "What is the capital of Spain?",             "madrid"),
    ("capital",   "What is the capital of the United Kingdom?", "london"),
    # COLORS — should be trivial
    ("color",     "What color is the sky?",                    "blue"),
    ("color",     "What color is grass?",                      "green"),
    ("color",     "What color is the sun?",                    "yellow"),
    # COUNTING — basic facts
    ("count",     "How many days are in a week?",              "7"),
    ("count",     "How many months are in a year?",            "12"),
    ("count",     "How many legs does a dog have?",            "4"),
    # MATH
    ("math",      "What is 2 plus 2?",                         "4"),
    ("math",      "What is 5 minus 3?",                        "2"),
    ("math",      "What is 3 times 3?",                        "9"),
    # YES/NO
    ("yesno",     "Is a dog an animal?",                       "yes"),
    ("yesno",     "Is the sun a star?",                        "yes"),
    ("yesno",     "Is water wet?",                             "yes"),
    # DEFINITIONS
    ("define",    "What is a dog?",                            ("animal", "mammal", "pet")),
    ("define",    "What is water?",                            ("liquid", "drink", "h2o")),
    ("define",    "What is a tree?",                           ("plant", "wood", "forest")),
    # INSTRUCTION-FOLLOWING (no fact required)
    ("instr",     "List three colors.",                        None),
    ("instr",     "Write a short poem.",                       None),
    ("instr",     "Say hello.",                                None),
    # OPEN-ENDED (typical chat)
    ("open",      "How are you?",                              None),
    ("open",      "Tell me about yourself.",                   None),
    ("open",      "What can you do?",                          None),
]


def run(model, tok, ctx, device, prompt):
    text = format_sft_prompt(prompt)
    ids = tok.encode(text)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for tid in generate_stream(
        model, x, max_new_tokens=80,
        temperature=0.0, top_k=50, top_p=0.9,
        context_length=ctx, eos_token_id=tok.eos_token_id,
        no_repeat_ngram_size=3,
    ):
        out.append(tid)
    raw = tok.decode(out)
    txt = raw
    for marker in ("\n###", "###", "<|endoftext|>"):
        i = txt.find(marker)
        if i >= 0:
            txt = txt[:i]; break
    return txt.strip(), len(out)


def label(category, prompt, expected, response):
    """Return (label, reason)."""
    rl = response.lower()
    # Hit check
    if isinstance(expected, str):
        hit = expected in rl
    elif isinstance(expected, tuple):
        hit = any(e in rl for e in expected)
    else:
        hit = None  # no fact required

    if hit is True:
        # Even with hit, check for repetition / nonsense
        if "is the" in rl and rl.count("is the") >= 2 and len(response.split()) < 12:
            return "PARTIAL", "echoes prompt structure with the answer mixed in"
        return "WORKS", "contains the expected fact"
    if hit is False:
        # Plausible structure but wrong content?
        if any(w in rl for w in ("is a", "are a", "is the", "are the")):
            return "BROKEN", "right format, wrong/missing fact"
        return "BROKEN", "no expected fact and structure is off"
    # No fact required — judge on coherence
    words = response.split()
    if len(words) < 3:
        return "BROKEN", "too short / empty"
    # Repetition heuristic
    bigrams = list(zip(words, words[1:]))
    rep_ratio = 1 - len(set(bigrams)) / max(1, len(bigrams))
    if rep_ratio > 0.4:
        return "BROKEN", "high repetition"
    if rep_ratio > 0.2:
        return "PARTIAL", "some repetition"
    return "WORKS", "coherent enough"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer()
    model, cfg = load_model_from_checkpoint(CKPT, device)
    ctx = cfg["model"]["context_length"]
    print(f"Checkpoint: {CKPT}")
    print(f"Decoding: greedy temp=0, no_repeat_ngram=3, max=80\n")

    counts = {"WORKS": 0, "PARTIAL": 0, "BROKEN": 0}
    by_cat: dict[str, list[str]] = {}

    for cat, prompt, expected in PROBES:
        resp, n = run(model, tok, ctx, device, prompt)
        verdict, reason = label(cat, prompt, expected, resp)
        counts[verdict] += 1
        by_cat.setdefault(cat, []).append(verdict)
        marker = {"WORKS": "OK ", "PARTIAL": "~~ ", "BROKEN": "BAD"}[verdict]
        print(f"[{marker}] [{cat:<7}] {prompt}")
        print(f"        -> {resp!r}   ({n}t, {reason})")

    # Aggregate
    total = sum(counts.values())
    print(f"\n{'=' * 80}")
    print(f"OVERALL: {counts['WORKS']}/{total} WORKS  "
          f"({counts['WORKS']*100//total}%) | "
          f"{counts['PARTIAL']}/{total} PARTIAL "
          f"({counts['PARTIAL']*100//total}%) | "
          f"{counts['BROKEN']}/{total} BROKEN "
          f"({counts['BROKEN']*100//total}%)")
    print(f"{'=' * 80}")
    for cat, verdicts in by_cat.items():
        n = len(verdicts)
        works = verdicts.count("WORKS")
        partial = verdicts.count("PARTIAL")
        broken = verdicts.count("BROKEN")
        print(f"  {cat:<7} {works}/{n} works, {partial}/{n} partial, {broken}/{n} broken")


if __name__ == "__main__":
    main()
