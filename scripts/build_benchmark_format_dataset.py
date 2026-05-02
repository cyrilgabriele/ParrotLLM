"""Build a JSONL of SFT records that match the EXACT leaderboard prompt format.

The PikoGPT leaderboard (external/PikoGPT_Leaderboard) scores generation:
  - Calls our main.py --stage inference --leaderboard with the benchmark prompt
  - Takes the first non-whitespace character of the model's generation
  - Compares to the gold letter (A/B/C/D for MCQ, A/B for WinoGrande)

So SFT must teach the model to output exactly the answer letter (with leading
space) as the very first generated token after the prompt.

Prompt formats (matched against the leaderboard's preprocessed validation files):

  HellaSwag:
    Context: <ctx_a> <ctx_b>
    A) <ending_0>
    B) <ending_1>
    C) <ending_2>
    D) <ending_3>
    Answer:

  OpenBookQA / ARC (we use "Question:" — same as OpenBookQA val):
    Question: <stem>
    A) <choice_a>
    B) <choice_b>
    C) <choice_c>
    D) <choice_d>
    Answer:

  WinoGrande:
    Context: <sentence with "_" placeholder>
    A) <option1>
    B) <option2>
    Answer:

Completion is always " <correct_letter>" — leading space + uppercase letter.

Sources are TRAIN splits only — val/test are decontamination targets.

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


LETTERS = ["A", "B", "C", "D"]


def _format_mc4(prefix: str, body: str, choices: list[str]) -> str:
    """4-choice prompt: '<prefix>: <body>\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\nAnswer:'."""
    lines = [f"{prefix}: {body}"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def _format_mc2(prefix: str, body: str, choices: list[str]) -> str:
    """2-choice prompt for WinoGrande."""
    lines = [f"{prefix}: {body}"]
    for letter, choice in zip(LETTERS[:2], choices):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def _arc_to_record(ex: dict) -> dict | None:
    """ARC: question, choices.text, choices.label, answerKey.

    ARC sometimes uses {A,B,C,D} labels and sometimes {1,2,3,4}; we always emit
    A/B/C/D in the prompt (preserving the original choice ordering)."""
    q = (ex.get("question") or "").strip()
    choices = ex.get("choices") or {}
    texts = choices.get("text") or []
    labels = choices.get("label") or []
    answer_key = (ex.get("answerKey") or "").strip().upper()
    if not q or not texts or not answer_key or len(texts) != 4:
        return None
    correct_idx = None
    for i, lbl in enumerate(labels):
        if str(lbl).strip().upper() == answer_key:
            correct_idx = i
            break
    if correct_idx is None or correct_idx >= 4:
        return None
    cleaned = [(t or "").strip() for t in texts]
    if any(not c for c in cleaned):
        return None
    prompt = _format_mc4("Question", q, cleaned)
    completion = LETTERS[correct_idx]  # "A"/"B"/"C"/"D"
    return {"prompt": prompt, "completion": completion}


def _obqa_to_record(ex: dict) -> dict | None:
    """OpenBookQA: question_stem, choices.text, choices.label, answerKey."""
    stem = (ex.get("question_stem") or "").strip()
    choices = ex.get("choices") or {}
    texts = choices.get("text") or []
    labels = choices.get("label") or []
    answer_key = (ex.get("answerKey") or "").strip().upper()
    if not stem or not texts or not answer_key or len(texts) != 4:
        return None
    correct_idx = None
    for i, lbl in enumerate(labels):
        if str(lbl).strip().upper() == answer_key:
            correct_idx = i
            break
    if correct_idx is None or correct_idx >= 4:
        return None
    cleaned = [(t or "").strip() for t in texts]
    if any(not c for c in cleaned):
        return None
    prompt = _format_mc4("Question", stem, cleaned)
    completion = LETTERS[correct_idx]
    return {"prompt": prompt, "completion": completion}


def _hellaswag_to_record(ex: dict) -> dict | None:
    """HellaSwag: ctx_a, ctx_b, endings, label."""
    ctx_a = (ex.get("ctx_a") or "").strip()
    ctx_b = (ex.get("ctx_b") or "").strip()
    endings = ex.get("endings") or []
    label_raw = ex.get("label")
    if label_raw is None or len(endings) != 4:
        return None
    try:
        idx = int(label_raw)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= 4:
        return None
    body = " ".join(part for part in [ctx_a, ctx_b] if part)
    if not body:
        return None
    cleaned = [re.sub(r"\s+", " ", (e or "")).strip() for e in endings]
    if any(not c for c in cleaned):
        return None
    prompt = _format_mc4("Context", body, cleaned)
    completion = LETTERS[idx]
    return {"prompt": prompt, "completion": completion}


def _winogrande_to_record(ex: dict) -> dict | None:
    """WinoGrande: sentence (with "_"), option1, option2, answer ('1'/'2')."""
    sentence = (ex.get("sentence") or "").strip()
    o1 = (ex.get("option1") or "").strip()
    o2 = (ex.get("option2") or "").strip()
    answer = (ex.get("answer") or "").strip()
    if not sentence or not o1 or not o2 or answer not in ("1", "2"):
        return None
    if "_" not in sentence:
        return None
    correct_idx = 0 if answer == "1" else 1
    prompt = _format_mc2("Context", sentence, [o1, o2])
    completion = LETTERS[correct_idx]
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("data/posttraining/custom/benchmark_format.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_arc_easy", type=int, default=2000)
    parser.add_argument("--n_arc_challenge", type=int, default=1000)
    parser.add_argument("--n_obqa", type=int, default=2000)
    parser.add_argument("--n_hellaswag", type=int, default=4000)
    parser.add_argument("--n_winogrande", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def take_records(name: str, ds, parser_fn, n: int) -> list[dict]:
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
        print(f"  {name:<14}: wrote {len(out):>5} (target {n})")
        return out

    print("Loading train splits ...")
    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    arc_chal = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    obqa = load_dataset("allenai/openbookqa", "main", split="train")
    hsw = load_dataset("Rowan/hellaswag", split="train")
    wgr = load_dataset("allenai/winogrande", "winogrande_xl", split="train")

    print("\n=== Sampling ===")
    records: list[dict] = []
    records += take_records("ARC-Easy", arc_easy, _arc_to_record, args.n_arc_easy)
    records += take_records("ARC-Challenge", arc_chal, _arc_to_record, args.n_arc_challenge)
    records += take_records("OpenBookQA", obqa, _obqa_to_record, args.n_obqa)
    records += take_records("HellaSwag", hsw, _hellaswag_to_record, args.n_hellaswag)
    records += take_records("WinoGrande", wgr, _winogrande_to_record, args.n_winogrande)

    rng.shuffle(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    avg_p = sum(len(r["prompt"]) for r in records) / max(1, len(records))
    avg_c = sum(len(r["completion"]) for r in records) / max(1, len(records))
    print(f"\nwrote {args.out} ({len(records)} records)")
    print(f"  avg prompt chars:     {avg_p:.0f}")
    print(f"  avg completion chars: {avg_c:.1f}  (always 1: just the letter)")
    print(f"\nFirst 2 records (rendered prompt + completion):\n")
    for r in records[:2]:
        print("  --- prompt ---")
        for line in r["prompt"].split("\n"):
            print(f"    {line}")
        print(f"  --- completion: {r['completion']!r} ---\n")


if __name__ == "__main__":
    main()
