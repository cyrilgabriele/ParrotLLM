# Post-training Experiments v3 — scientific addendum to v2

This document records three controlled experiments run after the
initial v1/v2 sweep. Each experiment changes a single variable from
the v2 baseline so that any metric delta can be cleanly attributed.

Same metrics across all experiments (factsheet §4.3):

- **Pillar #1**: perplexity on Wikitext-103 test + OWT val
- **Pillar #2**: accuracy on LAMBADA / HellaSwag / WinoGrande / OpenBookQA, n=500
- **Chat usability** (`tools/brutal_test.py`): 27-prompt hand-graded probe

---

## Experiment 1 — DPO val/train split audit

### Hypothesis

The unexplained DPO v1+v2 behaviour (val_loss → 0.018, val_acc 99%+,
monotonically falling val curve) is caused by train/val leakage in the
random pair-level split inside `src/post_training/dpo/data.py`.

### Method

`tools/audit_dpo_split.py` replays the data pipeline up to the split
and reports:

1. Prompt-uniqueness in the source dataset
2. Prompt-level overlap between the train and val partitions at the
   exact `(seed=42, val_fraction=0.05)` settings the trainer uses
3. Length asymmetry between chosen and rejected completions

### Result (executed, 2026-04-27)

```
Total preference pairs:    12,859
Unique prompts:            12,615   (98.1% of total)
Prompts with duplicates:   54
Total redundant rows:      244
Max copies of one prompt:  32

Train pairs:   12,216   (unique prompts: 11,986)
Val   pairs:      643   (unique prompts:    639)
Val prompts also in train: 10  (1.6% of unique val prompts)

avg chosen   length:   702.3 chars
avg rejected length:  1060.9 chars   (+51% longer)
chosen > rejected:   2,732 / 12,859  (21.2%)
chosen < rejected:  10,127 / 12,859  (78.8%)
```

### Interpretation

**Hypothesis rejected.** Prompt-level overlap is only 1.6% — not enough
to drive val_loss to 0.018.

**New hypothesis confirmed by inspection:** the orca_dpo_pairs dataset
has severe length asymmetry. Rejected responses are ~50% longer than
chosen on 78.8% of pairs. With the default DPO loss
`-log σ(β · (logp_π_chosen − logp_ref_chosen − logp_π_rejected + logp_ref_rejected))`
the per-sequence log-probs sum (rather than average) over tokens, so
longer sequences accumulate more low-probability mass. The model
learns "shorter == better" rather than "more helpful == better". The
99% val accuracy is the model recognising length, not preference
quality. This is the textbook DPO length-bias failure mode (Rafailov
et al. 2023, Appendix C).

### Action

Experiment 2: enable `length_normalize_logp` in the DPO config.

---

## Experiment 2 — DPO v3 with length normalisation

### Hypothesis

Setting `length_normalize_logp: true` (which divides per-pair logp by
completion token count) will:

- Reduce val_acc from ~99% toward ~70-80% (true preference signal at
  this scale, consistent with Zephyr-7B numbers)
- Reduce or eliminate the monotonic val_loss collapse
- Reduce remaining CF cost on Wikitext-103 (length-biased gradient was
  partially driving the policy off the SFT reference)

### Variable

Single change from `dpo_v2_balanced.yaml`:
`length_normalize_logp: false → true`. Everything else (β=0.2,
SFT base, dataset, schedule, eval cadence) held identical.

### Pre-registered metrics

- Best val_loss (expect higher than v2's 0.0175)
- Best val_acc (expect lower, ~0.7-0.8)
- WT-103 PPL at best checkpoint (expect ≤ v2's 91.64)
- OWT PPL at best checkpoint (expect ≤ v2's 37.06)
- MC mean accuracy at best checkpoint (expect ≥ v2's 30.25%)
- brutal_test works rate (expect ≥ v2's 40%)

### Result

_Pending — currently training in background as
`runs/run_20260427_*_dpo` from `configs/post_training/dpo_v3_length_norm.yaml`._

---

## Experiment 3 — SFT v3 with higher pretraining-mix

### Hypothesis

VL07 slide 25 lists pretraining-mix as one of five CF mitigations. v2
used `pretraining_mix_ratio: 0.05`, the lowest setting. Measured CF
cost: WT-103 +8.2%, OWT +4.6%. Raising to 0.20 should narrow this gap
without breaking the chat-template behaviour SFT v2 already learned
(Alpaca-format gradient drops from 95% to 80% of batches — still
dominant).

### Variable

Single change from `sft_v2_cf_mitigated.yaml`:
`pretraining_mix_ratio: 0.05 → 0.20`. Everything else (LR=1e-5,
2 epochs, batch_size=8, gradient_accumulation_steps=8, base
checkpoint, decontamination set) held identical.

### Pre-registered metrics

- WT-103 PPL at best checkpoint (expect ≤ +3% above pretrain, vs v2's
  +8.2%)
- OWT PPL at best checkpoint (expect ≤ +2% above pretrain)
- MC mean accuracy (expect ≥ v2's 30.55%)
- brutal_test format compliance (expect ≥ v2 — chat template should
  still hold)

### Result

_Pending — will run after DPO v3 completes (GPU is shared)._

---

## Experiment 4 — SFT v4 with synthetic factual mixin

### Hypothesis

SFT v2 brutal_test failure modes are concentrated in 0/3 categories:
math, yes/no, counting. These prompts have no factual hook in the
Alpaca corpus. A small synthetic dataset (~1000 pairs) of capitals,
colors, basic arithmetic, day/month counts, and yes/no questions
mixed at ~5% of SFT batches should specifically lift fact-recall on
the brutal_test buckets that are currently 0% without harming the
other metrics.

### Status

**Designed but not executed in this session.** The current SFT data
builder loads via `datasets.load_dataset(name, split=...)`. A small
extension is required to also consume a local JSONL of synthetic
pairs and concatenate before normalisation. Spec for future work:

```python
# src/post_training/sft/data.py::build_sft_datasets — add parameter:
synthetic_jsonl_path: str | None = None
# If set, load with load_dataset("json", data_files=path, split="train")
# and concatenate to the HF rows before normalisation.
```

Synthetic data spec — generate ~1000 examples covering:

- 50 capital cities (`{instruction: "What is the capital of X?",
  output: "The capital of X is Y."}`)
- 30 simple arithmetic (`What is N1 + N2?`, etc.)
- 20 yes/no factuals (`Is the sun a star?`)
- 30 counts (`How many days in a week?`)
- 20 colors (`What color is the sky?`)
- 10 simple definitions

Decontaminate against the 4 leaderboard test splits before mixing.
This was the smallest-leverage / most-data-work item, hence deferred.

### Pre-registered metrics

- brutal_test fact-bucket hit rate (expect 0/3 → 2-3/3 on math /
  count / yesno)
- brutal_test overall (expect 40% → 55-65%)
- WT-103 PPL (no regression vs v3)
- MC accuracy (no regression vs v3 — possibly +1-2pp on OpenBookQA
  where fact recall counts)
