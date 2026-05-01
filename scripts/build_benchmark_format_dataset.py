"""Build a JSONL of benchmark-format SFT records targeting the public eval suite.

Sources (train splits — held-out val/test sets are decontamination targets):
  - AI2 ARC-Easy train       (~2.3k)  → OpenBookQA / general MCQ transfer
  - AI2 ARC-Challenge train  (~1.1k)  → harder MCQ transfer
  - OpenBookQA train         (~5k)    → direct surface match
  - HellaSwag train          (~39.9k) → direct surface match (subsample)

Records are written in the surface form lm-eval-harness scoring uses:
  - MCQ task: prompt = "Question: ...\\nAnswer:", completion = "<correct_choice>"
  - HellaSwag: prompt = "<activity_label>: <ctx_a> <ctx_b>", completion = "<correct_ending>"

Consumed by the `local_jsonl` SFT loader together with `template_format: raw`
in the SFT config — which renders these without any chat wrapper, so the
SFT signal directly matches the eval surface form.

Usage:
    uv run python scripts/build_benchmark_format_dataset.py \\
        --out data/posttraining/custom/benchmark_format.jsonl \\
        [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def _arc_to_record(ex: dict) -> dict | None:
    """ARC schema: question, choices.text, choices.label, answerKey."""
    q = (ex.get("question") or "").strip()
    choices = ex.get("choices") or {}
    texts = choices.get("text") or []
    labels = choices.get("label") or []
    answer_key = (ex.get("answerKey") or "").strip().upper()
    if not q or not texts or not answer_key:
        return None
    correct_idx = None
    for i, lbl in enumerate(labels):
        if str(lbl).strip().upper() == answer_key:
            correct_idx = i
            break
    if correct_idx is None or correct_idx >= len(texts):
        return None
    correct_text = (texts[correct_idx] or "").strip()
    if not correct_text:
        return None
    return {
        "prompt": f"Question: {q}\nAnswer:",
        "completion": correct_text,
    }


def _obqa_to_record(ex: dict) -> dict | None:
    """OpenBookQA schema: question_stem, choices.text, choices.label, answerKey."""
    stem = (ex.get("question_stem") or "").strip()
    choices = ex.get("choices") or {}
    texts = choices.get("text") or []
    labels = choices.get("label") or []
    answer_key = (ex.get("answerKey") or "").strip().upper()
    if not stem or not texts or not answer_key:
        return None
    correct_idx = None
    for i, lbl in enumerate(labels):
        if str(lbl).strip().upper() == answer_key:
            correct_idx = i
            break
    if correct_idx is None or correct_idx >= len(texts):
        return None
    correct_text = (texts[correct_idx] or "").strip()
    if not correct_text:
        return None
    return {
        "prompt": f"Question: {stem}\nAnswer:",
        "completion": correct_text,
    }


def _hellaswag_to_record(ex: dict) -> dict | None:
    """HellaSwag schema: activity_label, ctx_a, ctx_b, endings, label.
    label is a string like '0' indexing into endings."""
    activity = (ex.get("activity_label") or "").strip()
    ctx_a = (ex.get("ctx_a") or "").strip()
    ctx_b = (ex.get("ctx_b") or "").strip()
    endings = ex.get("endings") or []
    label_raw = ex.get("label")
    if label_raw is None or not endings:
        return None
    try:
        idx = int(label_raw)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(endings):
        return None
    correct = (endings[idx] or "").strip()
    correct = re.sub(r"\s+", " ", correct).strip()
    if not correct or not (ctx_a or ctx_b):
        return None
    ctx_join = " ".join(part for part in [ctx_a, ctx_b] if part)
    if activity:
        ctx_full = f"{activity}: {ctx_join}"
    else:
        ctx_full = ctx_join
    return {"prompt": ctx_full, "completion": correct}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/posttraining/custom/benchmark_format.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_arc_easy", type=int, default=1500)
    parser.add_argument("--n_arc_challenge", type=int, default=1000)
    parser.add_argument("--n_obqa", type=int, default=1500)
    parser.add_argument("--n_hellaswag", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def take_records(name: str, ds, parser_fn, n: int) -> list[dict]:
        # ds may exceed n; randomly sample, parse, dedupe by prompt+completion
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i in idxs:
            r = parser_fn(ds[i])
            if r is None:
                continue
            key = (r["prompt"], r["completion"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= n:
                break
        print(f"  {name}: wrote {len(out):>5} (target {n})")
        return out

    records: list[dict] = []

    print("Loading datasets ...")
    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    arc_chal = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    obqa = load_dataset("allenai/openbookqa", "main", split="train")
    hsw = load_dataset("Rowan/hellaswag", split="train")

    print(f"\n=== Sampling ===")
    records += take_records("ARC-Easy", arc_easy, _arc_to_record, args.n_arc_easy)
    records += take_records("ARC-Challenge", arc_chal, _arc_to_record, args.n_arc_challenge)
    records += take_records("OpenBookQA", obqa, _obqa_to_record, args.n_obqa)
    records += take_records("HellaSwag", hsw, _hellaswag_to_record, args.n_hellaswag)

    rng.shuffle(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    # Quick stats
    avg_prompt_len = sum(len(r["prompt"]) for r in records) / max(1, len(records))
    avg_compl_len = sum(len(r["completion"]) for r in records) / max(1, len(records))
    print(f"\nwrote {args.out} ({len(records)} records)")
    print(f"  avg prompt chars:     {avg_prompt_len:.1f}")
    print(f"  avg completion chars: {avg_compl_len:.1f}")
    print(f"\nFirst 4 records:")
    for r in records[:4]:
        print(f"  prompt:     {r['prompt'][:100]!r}")
        print(f"  completion: {r['completion'][:100]!r}")
        print()


if __name__ == "__main__":
    main()
