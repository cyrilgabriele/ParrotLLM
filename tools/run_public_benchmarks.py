"""Single-process public-benchmark harness.

Replicates the unisg-ics-dsnlp/PikoGPT_Leaderboard scoring contract
in-process so we can measure HellaSwag / WinoGrande / OpenBookQA /
LAMBADA accuracy without spawning a fresh Python interpreter per
question. ~50-100x faster than the official runner for local iteration.

Compatible with the cloze-scoring path in src/eval/inference.py:
when --leaderboard mode would emit a single MC letter via cloze
scoring, this harness computes the same letter directly. For LAMBADA
it uses constrained greedy generation with the same lstrip() + split()
normalization the official runner applies.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path

import torch

from src.eval.inference import (
    detect_mc_letters,
    generate,
    load_model_from_checkpoint,
    parse_mc_options,
    score_mc_options,
)
from src.utils import build_tokenizer, set_seed


_LAMBADA_STRIP = ' \t\r\n"\'""''.,;:!?()[]{}'


def normalize_lambada(text: str) -> str:
    if not text:
        return ""
    first = text.lstrip().split()
    if not first:
        return ""
    return first[0].lower().strip(_LAMBADA_STRIP)


@torch.no_grad()
def predict_mc(model, tokenizer, prompt: str, device, ctx_len: int,
               pmi: bool = False) -> str | None:
    options = parse_mc_options(prompt)
    if options is None:
        return None
    scores = score_mc_options(model, tokenizer, prompt, options, device,
                               ctx_len, pmi=pmi)
    return max(scores, key=scores.get)


@torch.no_grad()
def predict_lambada(model, tokenizer, prompt: str, device, ctx_len: int,
                    max_tokens: int = 5) -> str:
    # rstrip: trailing space breaks BPE alignment and collapses argmax
    # onto an underscore-token. Mirrors the inference-path fix.
    input_ids = tokenizer.encode(prompt.rstrip())
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = generate(
        model, idx, max_tokens,
        temperature=0.0, top_k=0, top_p=1.0,
        context_length=ctx_len,
    )
    generated = tokenizer.decode(output[0, len(input_ids):].tolist())
    return generated


def run_benchmark(name: str, jsonl_path: Path, model, tokenizer, device,
                  ctx_len: int, limit: int | None, verbose: bool,
                  pmi: bool = False) -> dict:
    examples = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    if limit is not None:
        examples = examples[:limit]

    t0 = time.time()
    correct = 0
    invalid = 0
    samples = []
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        is_mc = name != "lambada"
        if is_mc:
            pred = predict_mc(model, tokenizer, prompt, device, ctx_len, pmi=pmi)
            gold = ex["answer_letter"]
            ok = pred is not None and pred.upper() == gold.upper()
            if pred is None:
                invalid += 1
        else:
            raw = predict_lambada(model, tokenizer, prompt, device, ctx_len)
            pred = normalize_lambada(raw)
            gold = ex["answer_text"].lower().strip(_LAMBADA_STRIP)
            ok = pred == gold
        if ok:
            correct += 1
        if verbose and i < 5:
            samples.append((prompt[-80:], pred, gold, ok))

    n = len(examples)
    acc = correct / n if n else 0.0
    elapsed = time.time() - t0
    return {
        "name": name,
        "n": n,
        "correct": correct,
        "invalid": invalid,
        "accuracy": acc,
        "elapsed_s": elapsed,
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--limit", type=int, default=500,
                   help="Per-benchmark example cap. Match the leaderboard's "
                        "default n=500. Pass 0 for full set.")
    p.add_argument("--bench", nargs="+",
                   default=["hellaswag", "winogrande", "openbookqa", "lambada"])
    p.add_argument("--data-dir", default="data/leaderboard_benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true",
                   help="Print first 5 (prompt-tail, pred, gold, ok) per benchmark.")
    p.add_argument("--pmi", action="store_true",
                   help="Enable PMI / Calibrate-Before-Use debiasing on MC.")
    p.add_argument("--out-json", default=None,
                   help="Append summary JSON to this path (one object per "
                        "invocation, separated by newline).")
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading {args.checkpoint} on {device} ...")
    model, ckpt_config = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = build_tokenizer()
    ctx_len = ckpt_config["model"]["context_length"]
    n_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"Model loaded: {n_params:,} params, ctx={ctx_len}")

    limit = args.limit if args.limit > 0 else None
    results = {}
    for name in args.bench:
        path = Path(args.data_dir) / f"{name}.jsonl"
        if not path.exists():
            print(f"  ! missing {path}, skipping")
            continue
        print(f"\n[{name}] running...")
        r = run_benchmark(name, path, model, tokenizer, device, ctx_len,
                          limit, args.verbose, pmi=args.pmi)
        results[name] = r
        print(f"  -> {r['correct']}/{r['n']} = {r['accuracy']*100:.1f}% "
              f"(invalid={r['invalid']}, {r['elapsed_s']:.1f}s)")
        if args.verbose:
            for prompt_tail, pred, gold, ok in r["samples"]:
                tag = "OK " if ok else "MISS"
                print(f"  [{tag}] gold={gold!r:>14} pred={pred!r:>14}  ...{prompt_tail!r}")

    if results:
        accs = [r["accuracy"] for r in results.values()]
        public_avg = sum(accs) / len(accs)
        print(f"\n=== summary ===")
        for name, r in results.items():
            print(f"  {name:<12} {r['accuracy']*100:>5.1f}%  "
                  f"(n={r['n']}, invalid={r['invalid']})")
        print(f"  {'public_avg':<12} {public_avg*100:>5.1f}%")

        if args.out_json:
            row = {
                "checkpoint": args.checkpoint,
                "limit": args.limit,
                "pmi": args.pmi,
                "public_avg": public_avg,
                "per_benchmark": {
                    name: {"accuracy": r["accuracy"], "n": r["n"],
                           "invalid": r["invalid"]}
                    for name, r in results.items()
                },
            }
            with open(args.out_json, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
