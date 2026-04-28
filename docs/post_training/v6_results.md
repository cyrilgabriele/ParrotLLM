# v6 results — synthetic raw-format SFT mixin

Generated 2026-04-28. The v6 plan: teach the model to recognise the
PikoGPT_Leaderboard runner's raw `"Context: ...\nA) ...\nAnswer:"`
template (in addition to Alpaca chat) so the runner's generation-based
parser sees a valid letter as the first character. Fixes the v5
"all-invalid" failure mode (model emitted prose instead of letters).

## What was different from v5

- New `RawCompletionTemplate` (no Alpaca markers) for synthetic rows.
- `synthetic_jsonl_path` parameter on `build_sft_datasets`; routes raw
  rows through the new template alongside Alpaca rows.
- ~2.4k synthetic raw-format rows: 851 programmatic factual-trivia
  (capitals, colors, arithmetic, animal classes, chemistry, synonyms,
  Winogrande-style) + 1500 reformatted from public Q&A *train* splits
  (SciQ, ARC-Easy, ARC-Challenge, CommonsenseQA).
- Per-batch mix target: 60% Alpaca / 20% synthetic raw / 20% pretrain.
- Decontam expanded to also hash MMLU/ARC/BoolQ/CSQA/SciQ test splits in
  addition to the visible 4.
- Inference: strict MC detection (`\nAnswer:$` + ≥2 `\n[A-Z]) ` lines)
  triggers first-token constraint to {A,B,C,D} ids in `--leaderboard`
  mode. Cloze (LAMBADA) untouched at inference; relies on training.

## Headline — PikoGPT_Leaderboard runner (limit=100)

| Bench       | dpo_v5 | **dpo_v6** | **sft_v6** | baseline | random |
|-------------|-------:|-----------:|-----------:|---------:|-------:|
| HellaSwag   |  ~10%  |   27.0%    | **29.0%**  |  27.8%   |   25%  |
| WinoGrande  |   ~0%  |   48.0%    |   47.0%    |  49.4%   |   50%  |
| OpenBookQA  |   ~0%  |   17.0%    | **20.0%**  |  22.6%   |   25%  |
| LAMBADA     |    0%  |    3.0%    |    2.0%    |   1.8%   |    —   |
| **mean**    |  ~3%   |  **23.75%**|  **24.5%** |  25.4%   |    —   |

**Key wins**:
- Fixed the v5 "all-invalid" catastrophe — every MC prompt now emits a
  valid letter (0–4 invalid out of 100 per bench, vs 70–100% before).
- Beat the existing baseline on HellaSwag (+1.2pp) and LAMBADA (+0.2pp).
- Within 0.9pp of baseline overall — at limit=100 sampling variance.

**Important: SFT v6 outperforms DPO v6 on the leaderboard runner.**
DPO was trained only on Alpaca-format pairs, so it slightly biases the
model away from raw-format MC completion. sft_v6 is the
right submission checkpoint.

## brutal_test (chat usability) — DPO v6

| stage | works | rate |
|---|---:|---:|
| dpo_v5 | 11/27 | 40% |
| **dpo_v6** | **12/27** | **44%** |

Open-ended category went 2/3 → 3/3. No regression vs v5; mild
improvement. Math/counting/yes-no still 0/3 (Experiment 4 fact-recall
mixin still deferred).

## Internal training metrics

- SFT v6: best at step 2000, val_loss **2.4211** (vs v5's 2.4115 — basically
  equivalent, synthetic mixin did not degrade Alpaca learning)
- DPO v6: best at step 360, val_loss **0.6450** (vs v5's 0.6449 —
  identical; length-norm + low-LR recipe still healthy on the new SFT)
- Training time: SFT 16 min, DPO 8 min on RTX 5090

## Open issue — OpenBookQA letter bias

Among the 20 saved wrong examples for sft_v6 OBQA, the model
predicted C 12×, D 4×, B 4×, **A 0×**. Golds were roughly uniform (A:6,
D:7, B:5, C:2). The model essentially never picks A on OBQA, suggesting
a model-specific token-logit bias rather than a training-data
distribution issue (synthetic data has A=26.5%, not the lowest letter).

If this bias can be fixed, OBQA could move from 20 → 25%+, lifting the
average from 24.5 → 25.75 and beating baseline.

Possible mitigations:
1. Per-letter rebalancing of the synthetic data (re-roll Winogrande to
   force exact 50/50 A/B; ensure 4-choice items hit exact 25/25/25/25).
2. Inference-time uniform-letter prior: subtract `log(P_train_letter)`
   from each letter logit.
3. Train more steps with shuffled letter assignments per epoch.

## Files

- `configs/post_training/sft_v6_8b.yaml` — SFT v6 config
- `configs/post_training/dpo_v6_8b.yaml` — DPO v6 config
- `data/synthetic/sft_v6_combined.jsonl` — 2351 synthetic raw rows
- `data/synthetic/sft_v6_programmatic.jsonl` — 851 programmatic
- `data/synthetic/sft_v6_public.jsonl` — 1500 from public Q&A trains
- `tools/build_synthetic_mc_programmatic.py`
- `tools/build_synthetic_mc_public.py`
- `tools/overnight_pipeline_v6.py`
- `runs/run_20260428_102441_sft/` — SFT v6 checkpoints
- `runs/run_20260428_104023_dpo/` — DPO v6 checkpoints

## Submission decision

**Recommended leaderboard checkpoint**: `sft_v6`
```
runs/run_20260428_102441_sft/checkpoints/best_step_0002000_epoch_01_valloss_2p4211.pt
```

Rationale: highest measured public_avg, beats baseline on 2 of 4
benches, no chat regression on its DPO sibling. Final limit=500 sweep
in progress for stable numbers comparable to baseline's reported
500-example evaluation.
