"""Reformat ~1500 examples from public Q&A train splits into the raw
leaderboard MC template.

Sources (TRAIN splits only — never test/validation):

  - SciQ-train               (~12k)  -> 4-choice
  - allenai/ai2_arc Easy-train (~2k)  -> 4-choice
  - allenai/ai2_arc Challenge-train (~1k) -> 4-choice
  - tau/commonsense_qa-train (~10k)  -> 5-choice (downcast to 4 by dropping
                                       a random distractor that isn't gold)
  - ybisk/piqa-train         (~16k)  -> 2-choice (A/B)

Output is hash-decontaminated by build_sft_datasets() at training time
against the *test/validation* splits of the leaderboard bank (visible
4 + extended 8). We also do a local pre-filter against the same set so
this script's output is itself clean if we want to inspect it.

Each pulled row -> raw JSONL line:

    {
      "instruction": "Context: <stem>\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\nAnswer:",
      "response": " <letter>"
    }

Output: data/synthetic/sft_v6_public.jsonl  (~1500 rows after caps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

OUT_PATH = Path("data/synthetic/sft_v6_public.jsonl")
SEED = 42
TARGET_TOTAL = 1500


def _normalise_for_hash(text: str) -> str:
    return " ".join((text or "").lower().split())


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_decontam_hashes() -> set[str]:
    """Mirror build_sft_datasets' decontam: hash every test/validation
    string from the visible 4 + extended 8 benchmarks."""
    from src.post_training.sft.data import DECONTAM_LOADERS, build_decontam_index

    texts: list[str] = []
    for name, loader in DECONTAM_LOADERS.items():
        try:
            texts.extend(loader())
            print(f"  decontam: loaded {name}")
        except Exception as e:
            print(f"  decontam: SKIP {name} ({e!r})")
    idx = build_decontam_index(texts)
    print(f"  decontam: {len(idx)} unique hashes")
    return idx


def _make_instruction(stem: str, choices: list[str], answer_idx: int) -> dict | None:
    """Render a single example. Returns None if shapes are degenerate."""
    if not stem or len(choices) < 2 or answer_idx >= len(choices):
        return None
    if len(choices) > 4:
        # Drop random non-gold distractors until we have 4
        keep = [answer_idx]
        candidates = [i for i in range(len(choices)) if i != answer_idx]
        random.shuffle(candidates)
        keep.extend(candidates[:3])
        keep.sort()
        new_idx = keep.index(answer_idx)
        choices = [choices[i] for i in keep]
        answer_idx = new_idx
    letters = "ABCDE"[: len(choices)]
    block = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    instruction = f"Context: {stem.strip()}\n{block}\nAnswer:"
    return {"instruction": instruction, "response": f" {letters[answer_idx]}"}


def _is_clean(ex: dict, decontam: set[str]) -> bool:
    blob = ex["instruction"] + " " + ex["response"]
    return _sha1(_normalise_for_hash(blob)) not in decontam


def pull_sciq(rng: random.Random, n: int, decontam: set[str]) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    out: list[dict] = []
    rows = list(ds)
    rng.shuffle(rows)
    for r in rows:
        choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        ans_idx = 0
        order = list(range(4))
        rng.shuffle(order)
        choices = [choices[i] for i in order]
        ans_idx = order.index(0)
        ex = _make_instruction(r["question"], choices, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_arc(rng: random.Random, n: int, decontam: set[str], cfg: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", cfg, split="train")
    out: list[dict] = []
    rows = list(ds)
    rng.shuffle(rows)
    for r in rows:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans_label = r["answerKey"]
        if ans_label not in labels:
            continue
        ans_idx = labels.index(ans_label)
        ex = _make_instruction(r["question"], texts, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def pull_csqa(rng: random.Random, n: int, decontam: set[str]) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="train")
    out: list[dict] = []
    rows = list(ds)
    rng.shuffle(rows)
    for r in rows:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        ans_label = r["answerKey"]
        if not ans_label or ans_label not in labels:
            continue
        ans_idx = labels.index(ans_label)
        ex = _make_instruction(r["question"], texts, ans_idx)
        if ex is None or not _is_clean(ex, decontam):
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_TOTAL)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    rng = random.Random(SEED)

    print("Loading decontam index …")
    decontam = _load_decontam_hashes()

    # Quotas — all draw from train splits; sum ~ args.target.
    # PIQA train was dropped because the HF mirror moved to a script-only
    # loader incompatible with current datasets >= 2.x. CSQA picks up the
    # slack because it is the closest commonsense-style dataset still
    # available script-free.
    quotas = {
        "sciq":         450,
        "arc_easy":     300,
        "arc_chall":    150,
        "csqa":         600,
    }
    # Scale to target
    scale = args.target / sum(quotas.values())
    quotas = {k: max(1, int(round(v * scale))) for k, v in quotas.items()}
    print(f"Quotas: {quotas}")

    rows: list[dict] = []
    print("Pulling SciQ-train …")
    rows += pull_sciq(rng, quotas["sciq"], decontam)
    print(f"  +{len(rows)} cumulative")
    print("Pulling ARC-Easy-train …")
    rows += pull_arc(rng, quotas["arc_easy"], decontam, cfg="ARC-Easy")
    print(f"  +{len(rows)} cumulative")
    print("Pulling ARC-Challenge-train …")
    rows += pull_arc(rng, quotas["arc_chall"], decontam, cfg="ARC-Challenge")
    print(f"  +{len(rows)} cumulative")
    print("Pulling CommonsenseQA-train …")
    rows += pull_csqa(rng, quotas["csqa"], decontam)
    print(f"  +{len(rows)} cumulative")

    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for ex in rows:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Per-letter histogram (sanity — letters should be roughly uniform)
    from collections import Counter
    letters = Counter(ex["response"].strip() for ex in rows)
    print(f"Wrote {len(rows)} examples to {args.out}")
    print(f"Letter distribution: {dict(letters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
