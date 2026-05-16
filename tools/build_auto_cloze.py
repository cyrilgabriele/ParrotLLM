"""Generate large-scale LAMBADA-style cloze examples from Wikitext-103.

Strategy: take a clean sentence from Wikitext-103 train, hold out the
last content-word token, present the rest as the prompt. Decontaminate
hard against the four leaderboard validation files (HellaSwag,
WinoGrande, OpenBookQA, LAMBADA) and Wikitext-103 test, so no
benchmark or eval-set sentence ever lands in the training pool.

Output format matches the existing SFT synthetic pipeline:
    {"instruction": <prompt>, "response": " <target_word>"}

The model is then SFT'd to emit ' <word>' as a continuation — exactly
the leaderboard's LAMBADA decoding contract (after rstrip the
benchmark prompt has no trailing space, the model's first emitted
token is a space-prefixed word, the runner strips and matches).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

# Words we don't want as the masked target: too easy / too short / function words.
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "with", "on", "at",
    "by", "it", "is", "was", "were", "are", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "but", "if", "as", "from", "into", "than", "then", "so", "such", "no",
    "not", "any", "all", "each", "every", "some", "many", "few", "much",
    "more", "most", "less", "very", "too", "also", "just", "only", "own",
    "same", "other", "another", "his", "her", "its", "their", "our", "your",
    "my", "me", "him", "us", "them", "they", "we", "you", "he", "she", "i",
    "who", "what", "which", "where", "when", "how", "why", "would", "could",
    "should", "may", "might", "can", "will", "shall", "must", "about",
    "after", "before", "during", "while", "between", "among", "without",
    "through", "above", "below", "over", "under",
}

DECONTAM_BENCHMARK_FILES = [
    "data/leaderboard_benchmarks/hellaswag.jsonl",
    "data/leaderboard_benchmarks/winogrande.jsonl",
    "data/leaderboard_benchmarks/openbookqa.jsonl",
    "data/leaderboard_benchmarks/lambada.jsonl",
]


def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def _sentences(text: str) -> list[str]:
    # Split on sentence-ending punctuation followed by whitespace. Keeps the
    # punctuation off the right side of each sentence (we strip it anyway
    # when finding the last content word).
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_decontam_hashes() -> set[str]:
    """Hash every sentence (>=20 chars) and every option text from the four
    benchmark validation files. Any candidate cloze whose source sentence
    or hash collides with this set is dropped.
    """
    hashes: set[str] = set()
    for fp_str in DECONTAM_BENCHMARK_FILES:
        fp = Path(fp_str)
        if not fp.exists():
            print(f"  ! decontam source missing: {fp}")
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bag: list[str] = []
                for field in ("prompt", "context", "answer_text"):
                    v = ex.get(field)
                    if isinstance(v, str):
                        bag.append(v)
                for v in ex.get("choices", []) or []:
                    if isinstance(v, str):
                        bag.append(v)
                for blob in bag:
                    for sentence in _sentences(blob):
                        if len(sentence) >= 20:
                            hashes.add(_hash(sentence))
    return hashes


def extract_cloze(sentence: str) -> tuple[str, str] | None:
    """Return ``(prompt, target_word)`` or None if the sentence is unusable.

    Filters: 6-25 words, last word is alphabetic and length >= 4 and not in
    STOPWORDS, prompt body is at least 30 chars after trimming.
    """
    sentence = sentence.strip()
    if not sentence or len(sentence) < 25 or len(sentence) > 400:
        return None
    # Remove a trailing ., !, or ? — we want the final word, not punctuation.
    body = re.sub(r"[.!?]+\s*$", "", sentence).rstrip()
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", body)
    if not (6 <= len(words) <= 25):
        return None
    last = words[-1]
    if len(last) < 4 or last.lower() in STOPWORDS:
        return None
    # Locate last occurrence of `last` (token-bounded) in the body so we can
    # split. last_index of word in body, case-preserving.
    last_idx = body.rfind(last)
    if last_idx < 0:
        return None
    prompt = body[:last_idx].rstrip()
    if len(prompt) < 30:
        return None
    return prompt, last


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=25000,
                   help="Target number of cloze rows.")
    p.add_argument("--out", type=Path,
                   default=Path("data/synthetic/sft_v8_auto_cloze.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-source-rows", type=int, default=300_000,
                   help="Hard cap on Wikitext rows to scan; bounds runtime.")
    args = p.parse_args()

    print("loading decontam hashes from leaderboard benchmark files ...")
    decontam = load_decontam_hashes()
    print(f"  {len(decontam)} blocklist hashes")

    print("loading wikitext-103 train (cached) ...")
    train_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    test_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    print("hashing wikitext-103 test sentences (additional decontam) ...")
    for row in test_ds:
        for s in _sentences(row.get("text", "")):
            if len(s) >= 20:
                decontam.add(_hash(s))
    print(f"  {len(decontam)} blocklist hashes after wt103-test")

    rng = random.Random(args.seed)
    rows: list[dict] = []
    seen_prompt_hashes: set[str] = set()
    blocked = 0
    scanned = 0

    PER_ROW_CAP = 3   # cap cloze rows extractable from a single wikitext row
    SAMPLE_PROB = 0.4 # probabilistic skip for further diversity

    for example in train_ds:
        scanned += 1
        if scanned > args.max_source_rows:
            break
        if len(rows) >= args.target:
            break
        text = example.get("text") or ""
        if not text or text.lstrip().startswith(" =") or text.lstrip().startswith("="):
            continue
        produced_here = 0
        for sentence in _sentences(text):
            if len(rows) >= args.target:
                break
            if produced_here >= PER_ROW_CAP:
                break
            if rng.random() > SAMPLE_PROB:
                continue
            res = extract_cloze(sentence)
            if res is None:
                continue
            prompt, target = res
            shash = _hash(sentence)
            if shash in decontam:
                blocked += 1
                continue
            phash = _hash(prompt)
            if phash in seen_prompt_hashes:
                continue
            seen_prompt_hashes.add(phash)
            rows.append({"instruction": prompt, "response": " " + target})
            produced_here += 1

    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    targets = Counter(r["response"].strip().lower() for r in rows)
    print(f"\nwrote {len(rows):,} cloze examples to {args.out}")
    print(f"  scanned {scanned:,} wikitext rows, "
          f"blocked {blocked:,} for decontam overlap")
    print(f"  unique target words: {len(targets):,}")
    print(f"  top 10 targets: {targets.most_common(10)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
