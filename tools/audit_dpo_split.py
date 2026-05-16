"""Audit the DPO data pipeline for train/val leakage.

Hypothesis: the unexplained val_loss=0.0175 (DPO v2) plus monotonically
falling val curve is caused by orca_dpo_pairs having duplicate prompts
that the random 5%-pair split distributes across both halves. Even with
a perfect leak-free split function, near-duplicates in the source make
val behave like a train sample.

Method: replay the data pipeline up to the split, count
  - unique prompts in the full dataset
  - prompt-level overlap between train and val
  - chosen/rejected length asymmetry (a separate "trivial preference"
    failure mode that also drives val_acc to 99%).
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset


def main():
    print("Loading Intel/orca_dpo_pairs ...")
    ds = load_dataset("Intel/orca_dpo_pairs", split="train")
    n = len(ds)
    print(f"Total preference pairs: {n:,}")

    # Build prompt strings as the trainer does.
    prompts = []
    chosen_lens = []
    rejected_lens = []
    for row in ds:
        sys = row.get("system") or ""
        q = row.get("question") or ""
        prompt = f"{sys}\n\n{q}".strip() if sys else q.strip()
        prompts.append(prompt)
        chosen_lens.append(len(row.get("chosen") or ""))
        rejected_lens.append(len(row.get("rejected") or ""))

    counts = Counter(prompts)
    unique = len(counts)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    max_dup = max(counts.values())
    n_repeated_prompts = sum(1 for c in counts.values() if c > 1)

    print(f"\n--- Prompt uniqueness ---")
    print(f"Unique prompts:            {unique:,}  ({unique/n:.1%} of total)")
    print(f"Prompts with duplicates:   {n_repeated_prompts:,}")
    print(f"Total redundant rows:      {repeated:,}")
    print(f"Max copies of one prompt:  {max_dup}")

    # Replay the trainer's split: shuffle indices with seed=42, take 5% val.
    import random
    rng = random.Random(42)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(round(n * 0.05)))
    val_idx = set(indices[:n_val])
    val_prompts = {prompts[i] for i in val_idx}
    train_prompts = {prompts[i] for i in range(n) if i not in val_idx}
    overlap = val_prompts & train_prompts

    print(f"\n--- Prompt-level overlap at 5% val_fraction (seed 42) ---")
    print(f"Train pairs:   {n - n_val:,}  | unique train prompts: {len(train_prompts):,}")
    print(f"Val   pairs:   {n_val:,}      | unique val   prompts: {len(val_prompts):,}")
    print(f"Val prompts also in train: {len(overlap):,} "
          f"({len(overlap)/max(1,len(val_prompts)):.1%} of unique val prompts)")

    # Length asymmetry — a 99% val_acc could also be explained by the
    # chosen response simply being far longer than the rejected one.
    avg_chosen = sum(chosen_lens) / n
    avg_rejected = sum(rejected_lens) / n
    longer_chosen = sum(c > r for c, r in zip(chosen_lens, rejected_lens))
    print(f"\n--- Chosen vs rejected length asymmetry ---")
    print(f"Avg chosen length (chars):    {avg_chosen:7.1f}")
    print(f"Avg rejected length (chars):  {avg_rejected:7.1f}")
    print(f"Pairs where chosen > rejected: {longer_chosen:,}/{n:,} "
          f"({longer_chosen/n:.1%})")
    print(f"Pairs where chosen < rejected: {n - longer_chosen:,}/{n:,} "
          f"({(n-longer_chosen)/n:.1%})")


if __name__ == "__main__":
    main()
