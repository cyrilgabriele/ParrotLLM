"""Reformat HellaSwag's TRAIN split into the leaderboard's raw MC template.

HellaSwag train (39k rows) and validation/test (10k each) are disjoint by
construction (split at dataset-creation time). Decontam against the
validation hashes (handled by build_sft_datasets at training time)
catches the unlikely paraphrase overlap.

Why this targets the v6 → v7 HellaSwag gap (-2.4pp vs baseline):

  - v6's synthetic data was Q&A style ("What is the capital of X?"). The
    model never saw narrative-continuation prompts of the HellaSwag form
    ("A man is sitting on a roof. he ___" + 4 endings).
  - This script reformats real HellaSwag train rows into the raw template
    ("Context: <ctx_a> <ctx_b>\\nA) <e1>\\nB) <e2>\\nC) <e3>\\nD) <e4>\\nAnswer:")
    + " <letter>". Same distribution as the test split.

Output: data/synthetic/sft_v7_hellaswag_style.jsonl (~1500 rows)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT_PATH = Path("data/synthetic/sft_v7_hellaswag_style.jsonl")
SEED = 43
TARGET_TOTAL = 1500


def _make_row(ctx: str, endings: list[str], answer_idx: int) -> dict | None:
    """Produce a single raw-format row. Drop if any field is degenerate."""
    if not ctx or len(endings) < 4 or answer_idx >= len(endings):
        return None
    if any(not e for e in endings[:4]):
        return None
    letters = "ABCD"
    block = "\n".join(f"{letters[i]}) {endings[i]}" for i in range(4))
    instruction = f"Context: {ctx.strip()}\n{block}\nAnswer:"
    return {"instruction": instruction, "response": f" {letters[answer_idx]}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_TOTAL)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    rng = random.Random(SEED)
    from datasets import load_dataset

    print("Loading HellaSwag train …")
    ds = load_dataset("Rowan/hellaswag", split="train")
    rows = list(ds)
    rng.shuffle(rows)
    print(f"Got {len(rows)} train rows.")

    out: list[dict] = []
    skipped = 0
    for r in rows:
        ctx_a = r.get("ctx_a", "")
        ctx_b = r.get("ctx_b", "")
        ctx = (ctx_a + " " + ctx_b).strip()
        endings = list(r.get("endings", []))
        try:
            ans_idx = int(r.get("label", -1))
        except (TypeError, ValueError):
            skipped += 1
            continue
        ex = _make_row(ctx, endings, ans_idx)
        if ex is None:
            skipped += 1
            continue
        out.append(ex)
        if len(out) >= args.target:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for ex in out:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    letters = Counter(ex["response"].strip() for ex in out)
    print(f"Wrote {len(out)} HellaSwag-style examples to {args.out}")
    print(f"Letter distribution: {dict(letters)}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
