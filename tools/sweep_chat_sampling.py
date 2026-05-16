"""Sweep chat sampling configs against the running Gradio UI and pick the
combination that produces the most correct answers on a small factual set.

Why this exists: the chat UI's defaults (temperature/top_p/top_k/rep_pen)
are arbitrary. For the 19.05 demo we want defaults that maximise correct
free-form answers on factual prompts, not benchmark log-likelihood. The
public benchmarks score by log-likelihood and are invariant to sampling.

Run order:
    1. Start the chat UI: uv run python main.py --stage chat \\
                              --config configs/chat/chat_v7.yaml
    2. uv run python tools/sweep_chat_sampling.py

Outputs:
    results/chat_sampling_sweep.json   raw per-config grades
    stdout                              ranked summary
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

from gradio_client import Client

sys.stdout.reconfigure(encoding="utf-8")

# Factual prompts with an "expected keyword" grader. Match is
# case-insensitive substring over the response (after the TTFT footer is
# stripped). Keep prompts diverse so a single decoding choice can't game it.
PROMPTS: list[tuple[str, list[str]]] = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Switzerland?", ["bern"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("What color do you get when you mix blue and yellow?", ["green"]),
    ("On which continent is Egypt located?", ["africa"]),
    ("How many sides does a triangle have?", ["3", "three"]),
    ("What is the chemical formula of water?", ["h2o", "h₂o", "h 2 o"]),
    ("What is the smallest prime number?", ["2", "two"]),
    ("What animal is known as the king of the jungle?", ["lion"]),
    ("What is the boiling point of water in Celsius?", ["100"]),
]

# Grid. Kept small so the sweep finishes in minutes, not hours.
TEMPS = [0.0, 0.3, 0.7, 1.0]
TOP_PS = [0.9, 1.0]
REP_PENS = [1.0, 1.1, 1.3]
TOP_K = 50
NGRAM = 3
MAX_TOK = 60


def grade(response: str, expected: list[str]) -> bool:
    body = re.sub(r"\n*_TTFT[^_]*_\s*$", "", response).strip().lower()
    return any(kw.lower() in body for kw in expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--out", default="results/chat_sampling_sweep.json")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller grid (temps × rep_pens only, top_p=0.9)")
    args = ap.parse_args()

    if args.quick:
        global TOP_PS
        TOP_PS = [0.9]

    grid = list(itertools.product(TEMPS, TOP_PS, REP_PENS))
    print(f"sweep: {len(grid)} configs × {len(PROMPTS)} prompts "
          f"= {len(grid) * len(PROMPTS)} generations")
    client = Client(args.url, verbose=False)

    results: list[dict] = []
    for i, (temp, top_p, rep) in enumerate(grid, 1):
        cfg = dict(temperature=temp, top_p=top_p, top_k=TOP_K,
                   repetition_penalty=rep, max_tokens=MAX_TOK,
                   no_repeat_ngram=NGRAM)
        correct = 0
        per_prompt = []
        t0 = time.perf_counter()
        for prompt, expected in PROMPTS:
            resp = client.predict(message=prompt, **cfg, api_name="/chat_fn")
            ok = grade(resp, expected)
            correct += int(ok)
            per_prompt.append({"prompt": prompt, "response": resp, "ok": ok,
                               "expected": expected})
        elapsed = time.perf_counter() - t0
        score = correct / len(PROMPTS)
        line = (f"[{i}/{len(grid)}] τ={temp:.2f} top_p={top_p:.2f} "
                f"rep={rep:.2f} → {correct}/{len(PROMPTS)} = "
                f"{score:.1%}  ({elapsed:.1f}s)")
        print(line)
        results.append({"config": cfg, "score": score, "correct": correct,
                        "n": len(PROMPTS), "elapsed_s": elapsed,
                        "per_prompt": per_prompt})

    results.sort(key=lambda r: r["score"], reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(f"=== ranked results (n={len(PROMPTS)}) ===")
    print(f"{'rank':<5}{'score':<8}{'τ':<8}{'top_p':<8}{'rep':<6}")
    for r, item in enumerate(results, 1):
        c = item["config"]
        print(f"{r:<5}{item['score']:<8.1%}{c['temperature']:<8.2f}"
              f"{c['top_p']:<8.2f}{c['repetition_penalty']:<6.2f}")

    best = results[0]
    print()
    print(f"BEST: {best['correct']}/{best['n']} = {best['score']:.1%}")
    print(f"      temperature      = {best['config']['temperature']}")
    print(f"      top_p            = {best['config']['top_p']}")
    print(f"      top_k            = {best['config']['top_k']}")
    print(f"      repetition_penalty = {best['config']['repetition_penalty']}")
    print(f"      no_repeat_ngram  = {best['config']['no_repeat_ngram']}")
    print(f"raw results → {out_path}")


if __name__ == "__main__":
    main()
