"""Benchmark-style evaluation of SFT checkpoints — likelihood scoring on the
suites the project's decontam list pins as eval targets:

    HellaSwag     — 4-way commonsense MCQ, accuracy = argmax of choice log-probs
    WinoGrande    — 2-way pronoun MCQ, accuracy = argmax of choice log-probs
    OpenBookQA    — 4-way science MCQ, accuracy = argmax of choice log-probs
    LAMBADA       — last-word prediction, accuracy = exact-match of greedy token
    Wikitext-103  — neutral English perplexity
    OWT val       — neutral OWT perplexity (reuse data/processed/val.bin)

This is the EleutherAI lm-eval-harness "lite" subset. We do likelihood-only
scoring: for MCQ tasks, sum log-prob of each candidate completion conditioned
on its prompt and pick argmax. No generation. No chat template. The chat-style
SFT recipes the project has been tuning may *hurt* these scores by shifting
the model away from neutral next-word likelihood.

Usage:
    uv run python scripts/sft_eval_benchmarks.py \\
        --checkpoint <ckpt1.pt> [--checkpoint <ckpt2.pt> ...] \\
        --baseline <base_ckpt.pt> \\
        --device mps \\
        [--subsample 500] [--out reports/benchmarks.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402

from src.model import ParrotLLM  # noqa: E402
from src.utils import build_tokenizer  # noqa: E402


# ── Likelihood scoring primitive ──────────────────────────────────────────

@torch.no_grad()
def score_continuation(model, tokenizer, prompt: str, completion: str,
                       device: torch.device, ctx_len: int) -> float:
    """Sum log-prob of completion tokens given prompt. Higher = more likely."""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + completion)
    cont_ids = full_ids[len(prompt_ids):]
    if not cont_ids:
        return 0.0
    if len(full_ids) > ctx_len:
        # left-truncate the prompt to fit; keep the completion
        keep_prompt = ctx_len - len(cont_ids)
        prompt_ids = prompt_ids[-max(0, keep_prompt):]
        full_ids = prompt_ids + cont_ids
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x, return_logits=True)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # log-prob of each y_i: log_probs[0, i, y[0,i]]
    target_log_probs = log_probs[0].gather(-1, y[0].unsqueeze(-1)).squeeze(-1)
    # Only sum log-probs of CONTINUATION tokens
    cont_start = len(prompt_ids) - 1  # position predicting first cont token
    cont_end = cont_start + len(cont_ids)
    return float(target_log_probs[cont_start:cont_end].sum().item())


# ── Tasks ─────────────────────────────────────────────────────────────────

def eval_hellaswag(model, tokenizer, device, ctx_len, n: int) -> dict:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n and n < len(ds):
        ds = ds.select(range(n))
    correct = 0
    total = 0
    for ex in ds:
        # lm-eval format: "{activity_label}: {ctx_a} {ctx_b}"
        ctx = (ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        target = int(ex["label"])
        scores = [
            score_continuation(model, tokenizer, ctx + " ", end, device, ctx_len)
            for end in ex["endings"]
        ]
        if int(np.argmax(scores)) == target:
            correct += 1
        total += 1
    return {"task": "hellaswag", "acc": correct / max(1, total), "n": total}


def eval_winogrande(model, tokenizer, device, ctx_len, n: int) -> dict:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    if n and n < len(ds):
        ds = ds.select(range(n))
    correct = 0
    total = 0
    for ex in ds:
        sentence = ex["sentence"]
        # Replace _ with each option, score full sentence, pick higher likelihood
        cand1 = sentence.replace("_", ex["option1"])
        cand2 = sentence.replace("_", ex["option2"])
        s1 = score_continuation(model, tokenizer, "", cand1, device, ctx_len)
        s2 = score_continuation(model, tokenizer, "", cand2, device, ctx_len)
        # answer is "1" or "2"
        target = int(ex["answer"])
        pred = 1 if s1 >= s2 else 2
        if pred == target:
            correct += 1
        total += 1
    return {"task": "winogrande", "acc": correct / max(1, total), "n": total}


def eval_openbookqa(model, tokenizer, device, ctx_len, n: int) -> dict:
    ds = load_dataset("allenai/openbookqa", "main", split="validation")
    if n and n < len(ds):
        ds = ds.select(range(n))
    correct = 0
    total = 0
    for ex in ds:
        q = ex["question_stem"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        target_label = ex["answerKey"]
        prompt = "Question: " + q + "\nAnswer:"
        scores = [
            score_continuation(model, tokenizer, prompt, " " + c, device, ctx_len)
            for c in choices
        ]
        pred_idx = int(np.argmax(scores))
        if labels[pred_idx] == target_label:
            correct += 1
        total += 1
    return {"task": "openbookqa", "acc": correct / max(1, total), "n": total}


def eval_lambada(model, tokenizer, device, ctx_len, n: int) -> dict:
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if n and n < len(ds):
        ds = ds.select(range(n))
    correct = 0
    total = 0
    for ex in ds:
        text = ex["text"]
        last_space = text.rfind(" ")
        if last_space < 0:
            continue
        context = text[:last_space]
        target = text[last_space:]  # leading space + last word
        # Score the gold target's log-prob — we report *exact-match* by greedy decode
        target_ids = tokenizer.encode(target)
        ctx_ids = tokenizer.encode(context)
        if not target_ids or not ctx_ids:
            continue
        seq = ctx_ids[-(ctx_len - len(target_ids)):] + target_ids
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x, return_logits=True)
        # greedy prediction at the position predicting the FIRST target token
        pred_start = len(seq) - len(target_ids) - 1
        pred_ids = logits[0, pred_start:pred_start + len(target_ids)].argmax(dim=-1).tolist()
        if pred_ids == target_ids:
            correct += 1
        total += 1
    return {"task": "lambada", "acc": correct / max(1, total), "n": total}


@torch.no_grad()
def eval_perplexity(model, tokenizer, device, ctx_len, hf_path: str, hf_subset: str | None,
                    split: str, field: str, max_seqs: int, name: str) -> dict:
    if hf_subset is None:
        ds = load_dataset(hf_path, split=split)
    else:
        ds = load_dataset(hf_path, hf_subset, split=split)
    nll_sum = 0.0
    tok_sum = 0
    seqs = 0
    for ex in ds:
        text = ex.get(field) or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text)[:ctx_len]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        logits, _ = model(x, return_logits=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum")
        nll_sum += float(loss.item())
        tok_sum += int(y.numel())
        seqs += 1
        if seqs >= max_seqs:
            break
    avg_nll = nll_sum / max(1, tok_sum)
    return {"task": name, "ppl": math.exp(avg_nll) if math.isfinite(avg_nll) else float("nan"), "n_tokens": tok_sum}


# ── Per-checkpoint runner ─────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: Path, *, device: torch.device, tokenizer,
                       subsample: int) -> dict:
    print(f"  loading {ckpt_path.name}", file=sys.stderr)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ParrotLLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ctx_len = int(ckpt["config"]["model"]["context_length"])
    out: dict = {}
    print("  HellaSwag ...", file=sys.stderr)
    out.update({"hellaswag_acc": eval_hellaswag(model, tokenizer, device, ctx_len, subsample)["acc"]})
    print("  WinoGrande ...", file=sys.stderr)
    out.update({"winogrande_acc": eval_winogrande(model, tokenizer, device, ctx_len, subsample)["acc"]})
    print("  OpenBookQA ...", file=sys.stderr)
    out.update({"openbookqa_acc": eval_openbookqa(model, tokenizer, device, ctx_len, subsample)["acc"]})
    print("  LAMBADA ...", file=sys.stderr)
    out.update({"lambada_acc": eval_lambada(model, tokenizer, device, ctx_len, subsample)["acc"]})
    print("  Wikitext-103 ppl ...", file=sys.stderr)
    out.update({"wikitext_ppl": eval_perplexity(model, tokenizer, device, ctx_len,
                                                "wikitext", "wikitext-103-raw-v1",
                                                "test", "text",
                                                max_seqs=min(subsample, 500),
                                                name="wikitext_ppl")["ppl"]})
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def render_markdown(reports: list[tuple[str, dict]]) -> str:
    cols = ["benchmark"] + [lbl for lbl, _ in reports]
    lines = ["# Leaderboard-aligned benchmarks\n"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    def row(label: str, key: str, fmt: str = "{:.3f}", scale: float = 1.0):
        cells = [label]
        for _, m in reports:
            v = m.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("n/a")
            else:
                cells.append(fmt.format(v * scale))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("| **MCQ accuracy (higher = better)** |" + "|" * len(reports))
    row("HellaSwag (4-way commonsense)", "hellaswag_acc", "{:.1%}")
    row("WinoGrande (2-way pronoun)", "winogrande_acc", "{:.1%}")
    row("OpenBookQA (4-way science)", "openbookqa_acc", "{:.1%}")
    lines.append("| **Completion accuracy** |" + "|" * len(reports))
    row("LAMBADA (last-word exact match)", "lambada_acc", "{:.1%}")
    lines.append("| **Perplexity (lower = better)** |" + "|" * len(reports))
    row("Wikitext-103", "wikitext_ppl", "{:.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--subsample", type=int, default=500,
                        help="Cap per-task examples (use 0 for full).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    tokenizer = build_tokenizer()

    targets: list[Path] = []
    if args.baseline is not None:
        targets.append(args.baseline)
    targets += list(args.checkpoint)

    reports: list[tuple[str, dict]] = []
    seen: set[Path] = set()
    for ckpt_path in targets:
        if ckpt_path in seen:
            continue
        seen.add(ckpt_path)
        print(f"=== Scoring {ckpt_path.parent.parent.name} ===", file=sys.stderr)
        m = evaluate_checkpoint(ckpt_path, device=device, tokenizer=tokenizer,
                                subsample=args.subsample)
        label = ckpt_path.parent.parent.name
        reports.append((label, m))

    md = render_markdown(reports)
    print()
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
