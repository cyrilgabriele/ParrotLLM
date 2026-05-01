"""Generate ~3k synthetic length/format-constrained instruction-response pairs.

Targeted at the failure modes the SFT scorecard surfaces on IFEval-lite:
  - "reply with one word" → model rambles
  - "return JSON" → model writes prose
  - "pick A or B" → model writes essay
  - "answer in N words" → model ignores constraint

The pairs are deterministic (no LLM in the loop) so they're free to regenerate
and the dataset is reproducible. Each pair is *short*, which gives high
EOS-supervision density per record — directly attacks the EOS-rate gap too.

Schema: JSONL of {prompt, completion} records, consumed by the local_jsonl
loader in src.posttraining.prepare._normalize_local_jsonl_record.

Usage:
    uv run python scripts/build_synthetic_format_dataset.py \\
        --out data/posttraining/custom/synthetic_format.jsonl \\
        [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SINGLE_WORDS = [
    "yes", "no", "ok", "done", "stop", "go", "true", "false", "hello",
    "goodbye", "maybe", "now", "later", "ready", "help", "wait",
]
LETTERS = ["A", "B", "C", "D"]

# (prompt, gold_one_word_answer) — short factual probes
FACT_ONE_WORD = [
    ("What is the capital of France? One word.", "Paris"),
    ("What is the capital of Germany? One word.", "Berlin"),
    ("What is the capital of Japan? One word.", "Tokyo"),
    ("What is the capital of Italy? One word.", "Rome"),
    ("What is the capital of Spain? One word.", "Madrid"),
    ("What is the capital of Russia? One word.", "Moscow"),
    ("What is the capital of Egypt? One word.", "Cairo"),
    ("What is the capital of Greece? One word.", "Athens"),
    ("What color is the sky on a clear day? One word.", "blue"),
    ("What color is grass? One word.", "green"),
    ("What color is fresh snow? One word.", "white"),
    ("What color is fire? One word.", "red"),
    ("What color is the sun? One word.", "yellow"),
    ("What color is coal? One word.", "black"),
    ("How many sides does a triangle have? Just the number.", "3"),
    ("How many sides does a square have? Just the number.", "4"),
    ("How many sides does a hexagon have? Just the number.", "6"),
    ("How many days in a week? Just the number.", "7"),
    ("How many months in a year? Just the number.", "12"),
    ("How many hours in a day? Just the number.", "24"),
    ("Which planet is closest to the sun? One word.", "Mercury"),
    ("Which planet is the largest? One word.", "Jupiter"),
    ("Which is the largest ocean? One word.", "Pacific"),
    ("Which is the longest river? One word.", "Nile"),
    ("Is water wet? Reply yes or no only.", "yes"),
    ("Is fire cold? Reply yes or no only.", "no"),
    ("Is the Earth flat? Reply yes or no only.", "no"),
    ("Is the sun hot? Reply yes or no only.", "yes"),
    ("Do humans have wings? Reply yes or no only.", "no"),
    ("Do birds fly? Reply yes or no only.", "yes"),
]

# (instruction_template, expected response) for length-bounded answers
LENGTH_BOUND_TEMPLATES = [
    ("Answer in three words: What is gravity?", "A pulling force."),
    ("Answer in three words: What is fire?", "Hot burning gas."),
    ("Answer in three words: What is rain?", "Water from clouds."),
    ("Answer in three words: What is wind?", "Moving air outdoors."),
    ("Answer in three words: What is snow?", "Frozen falling water."),
    ("Answer in five words: What is photosynthesis?", "Plants make food from light."),
    ("Answer in five words: What is gravity?", "Force that pulls objects down."),
    ("Answer in five words: What is the sun?", "A very hot bright star."),
    ("Answer in two words: Describe a cat.", "Furry pet."),
    ("Answer in two words: Describe a dog.", "Loyal companion."),
    ("Answer in two words: Describe rain.", "Falling water."),
    ("Answer in two words: Describe an apple.", "Red fruit."),
]


def _gen_word_echo(rng: random.Random, n: int) -> list[dict]:
    """Reply-with-the-word-X template — pure echo, very short response."""
    out = []
    templates = [
        "Reply with the single word: {w}.",
        "Reply with only: {w}",
        "Reply with: {w}",
        "Output the word {w} and nothing else.",
        "Just say {w}.",
        "Echo this single word: {w}",
        "Say only the word {w}.",
        "Your reply must be exactly: {w}",
    ]
    for _ in range(n):
        w = rng.choice(SINGLE_WORDS)
        t = rng.choice(templates)
        out.append({"prompt": t.format(w=w), "completion": w})
    return out


def _gen_letter_echo(rng: random.Random, n: int) -> list[dict]:
    """Reply-with-the-letter-X — covers MCQ-style single-letter outputs."""
    out = []
    templates = [
        "Reply with only the letter {l}.",
        "Output the single letter {l}.",
        "Just the letter {l}, nothing else.",
        "Your answer is {l}. Reply only with that letter.",
        "Choose {l}. Output only the letter.",
    ]
    for _ in range(n):
        ltr = rng.choice(LETTERS)
        t = rng.choice(templates)
        out.append({"prompt": t.format(l=ltr), "completion": ltr})
    return out


def _gen_mcq_guided(rng: random.Random, n: int) -> list[dict]:
    """Pick-A-or-B with the answer hinted in the prompt."""
    out = []
    templates = [
        "Choose A or B. Pick {l}. Answer with only the letter.",
        "Options: A, B, C, D. The correct answer is {l}. Reply with the letter only.",
        "Pick {l}. Reply with the single letter.",
        "Between A, B, C, D — choose {l}. One letter only.",
    ]
    for _ in range(n):
        ltr = rng.choice(LETTERS)
        t = rng.choice(templates)
        out.append({"prompt": t.format(l=ltr), "completion": ltr})
    return out


def _gen_yes_no(rng: random.Random, n: int) -> list[dict]:
    """Reply-yes-or-no with the answer in the prompt."""
    out = []
    templates_yes = [
        "Reply yes or no. The answer is yes.",
        "Just yes or no — say yes.",
        "Yes or no? Say yes.",
        "Output only yes or no. Output yes.",
    ]
    templates_no = [
        "Reply yes or no. The answer is no.",
        "Just yes or no — say no.",
        "Yes or no? Say no.",
        "Output only yes or no. Output no.",
    ]
    for _ in range(n):
        if rng.random() < 0.5:
            out.append({"prompt": rng.choice(templates_yes), "completion": "yes"})
        else:
            out.append({"prompt": rng.choice(templates_no), "completion": "no"})
    return out


def _gen_json_simple(rng: random.Random, n: int) -> list[dict]:
    """Simple JSON-only outputs."""
    out = []
    keys = ["x", "y", "z", "name", "age", "value", "id", "count"]
    for _ in range(n):
        if rng.random() < 0.5:
            k = rng.choice(keys)
            v = rng.randint(0, 99)
            out.append({
                "prompt": f"Return JSON with key \"{k}\" and value {v}.",
                "completion": json.dumps({k: v}),
            })
        else:
            k1, k2 = rng.sample(keys, 2)
            v1 = rng.randint(0, 50)
            v2 = rng.randint(0, 50)
            out.append({
                "prompt": f"Return JSON with keys \"{k1}\" and \"{k2}\", values {v1} and {v2}.",
                "completion": json.dumps({k1: v1, k2: v2}),
            })
    return out


def _gen_math_one_token(rng: random.Random, n: int) -> list[dict]:
    """Trivial arithmetic, one-number response."""
    out = []
    templates = [
        "What is {a} + {b}? Just the number.",
        "{a} + {b} = ?  Reply with the number only.",
        "Compute {a}+{b}. One number, no words.",
        "Add {a} and {b}. Output the result, nothing else.",
    ]
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        t = rng.choice(templates)
        out.append({"prompt": t.format(a=a, b=b), "completion": str(a + b)})
    return out


def _gen_fact_one_word(rng: random.Random, n: int) -> list[dict]:
    """Curated factual one-word probes (rephrased for variety)."""
    out = []
    rephrasings_capital = [
        "What is the capital of {country}? Reply with one word.",
        "Capital of {country}? One word only.",
        "Name the capital of {country}. Single word.",
    ]
    for _ in range(n):
        prompt, gold = rng.choice(FACT_ONE_WORD)
        # Sometimes reuse as-is, sometimes mildly vary the phrasing
        if "capital of " in prompt and rng.random() < 0.3:
            # extract country and reformat
            country = prompt.split("capital of ")[1].split("?")[0].strip()
            template = rng.choice(rephrasings_capital)
            prompt = template.format(country=country)
        out.append({"prompt": prompt, "completion": gold})
    return out


def _gen_length_bound(rng: random.Random, n: int) -> list[dict]:
    """Length-bound short answers (2/3/5 word constraints)."""
    out = []
    for _ in range(n):
        prompt, response = rng.choice(LENGTH_BOUND_TEMPLATES)
        out.append({"prompt": prompt, "completion": response})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/posttraining/custom/synthetic_format.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_word_echo", type=int, default=600)
    parser.add_argument("--n_letter_echo", type=int, default=300)
    parser.add_argument("--n_mcq_guided", type=int, default=300)
    parser.add_argument("--n_yes_no", type=int, default=300)
    parser.add_argument("--n_json", type=int, default=400)
    parser.add_argument("--n_math", type=int, default=400)
    parser.add_argument("--n_fact_word", type=int, default=400)
    parser.add_argument("--n_length_bound", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict] = []
    records += _gen_word_echo(rng, args.n_word_echo)
    records += _gen_letter_echo(rng, args.n_letter_echo)
    records += _gen_mcq_guided(rng, args.n_mcq_guided)
    records += _gen_yes_no(rng, args.n_yes_no)
    records += _gen_json_simple(rng, args.n_json)
    records += _gen_math_one_token(rng, args.n_math)
    records += _gen_fact_one_word(rng, args.n_fact_word)
    records += _gen_length_bound(rng, args.n_length_bound)
    rng.shuffle(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    # Per-category summary
    print(f"wrote {args.out}  ({len(records)} records)")
    print(f"  word echo:     {args.n_word_echo}")
    print(f"  letter echo:   {args.n_letter_echo}")
    print(f"  mcq guided:    {args.n_mcq_guided}")
    print(f"  yes/no:        {args.n_yes_no}")
    print(f"  JSON:          {args.n_json}")
    print(f"  math:          {args.n_math}")
    print(f"  fact one-word: {args.n_fact_word}")
    print(f"  length-bound:  {args.n_length_bound}")


if __name__ == "__main__":
    main()
