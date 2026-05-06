# DPO continuation-pair length audit

**Date:** 2026-05-06
**Branch:** `sft-dpo-gian`
**Data audited:** `data/posttraining/dpo_pairs_continuation/train.jsonl` (24,500 pairs across 7 sources, per `manifest.json`).
**Trainer audited:** `src/posttraining/dpo/trainer.py` + `src/posttraining/dpo/loss.py`.

## TL;DR

- **Within a pair, chosen vs rejected response lengths are almost perfectly balanced** (overall mean diff = -0.02 tokens, median diff = 0; per-source means within ~0.5 tokens). There is no systematic within-pair length bias that would let DPO "cheat" by preferring the shorter or longer side.
- **Across sources, response lengths are wildly unbalanced.** HellaSwag's mean response length is **13.89 tokens (chosen)** and 14.09 (rejected); WinoGrande's is **1.60** (both); CommonsenseQA's is 2.17 / 2.19; SciQ's is 2.67 / 2.59.
- **The current trainer SUMS log-probs over response tokens** (`sequence_logprob_from_labels` in `src/posttraining/dpo/loss.py`, called from `_forward_logp` in `src/posttraining/dpo/trainer.py`). It does NOT mean-over-tokens. Combined with the cross-source length asymmetry, this means **HellaSwag — 30.6% of pairs — contributes ~67.9% of the unnormalized gradient signal**. WinoGrande contributes 20.4% of the pairs but only 5.2% of the signal.
- **Verdict: the cross-source bias is large enough to justify a length-normalized retrain on this dataset. Within-pair bias is not.** The headline number is the 2.2x gradient over-weighting of HellaSwag relative to its pair share.

## Method

1. Loaded all 24,500 train records and reproduced the `_run_prepare_dpo_continuation` source ordering: dev (500 pairs) is filled first from the first source(s) in `dpo.sources` order; the remainder of each source's `target_pairs` lands in `train.jsonl` in source order. Mapping for our manifest:
   - hellaswag: 8,000 - 500 dev = **7,500 train** records (rows 1..7,500)
   - winogrande: 5,000 (rows 7,501..12,500)
   - openbookqa: 3,000 (rows 12,501..15,500)
   - arc_easy: 2,000 (rows 15,501..17,500)
   - arc_challenge: 1,000 (rows 17,501..18,500)
   - sciq: 3,000 (rows 18,501..21,500)
   - commonsense_qa: 3,000 (rows 21,501..24,500)
   Total: 24,500. Matches `n_train`.
2. For each pair, computed `chosen_response_len = len(chosen_tokens) - prompt_len` and `rejected_response_len = len(rejected_tokens) - prompt_len`.
3. Computed per-source and overall mean / median / p25 / p75 / min / max / sum.
4. Computed the fraction of pairs where `rejected_len >= chosen_len` and `rejected_len > chosen_len`.
5. Estimated gradient-signal share per source as `(chosen_sum + rejected_sum) / total`, since sum-of-logp DPO weights each pair by the number of unmasked response tokens (combined chosen+rejected forward). This is a coarse proxy assuming roughly uniform per-token log-prob magnitude across sources, but it's directionally accurate.

## Results

### Overall (24,500 pairs)

| metric | chosen response | rejected response |
|---|---|---|
| mean   | 6.32 | 6.30 |
| median | 3.0  | 3.0  |
| p25    | 2    | 2    |
| p75    | 9    | 9    |
| min    | 1    | 1    |
| max    | 81   | 111  |
| sum    | 154,741 | 154,234 |

- `rejected_len >= chosen_len` in **64.2%** of pairs.
- `rejected_len >  chosen_len` in **33.6%** of pairs.
- `rejected_len - chosen_len`: mean **-0.02**, median **0.0**.

The within-pair distribution is symmetric. The 64.2% "rejected >= chosen" figure is dominated by ties at very short lengths (e.g. WinoGrande's two-token options, where 73.9% of pairs are tied or the rejected is shifted by a single token).

### Per source

| source | pairs | chosen mean | chosen median | rejected mean | rejected median | rejected >= chosen | tokens (chosen+rejected) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hellaswag        | 7,500 | 13.89 | 12.0 | 14.09 | 13.0 | 56.3% | 209,802 |
| winogrande       | 5,000 |  1.60 |  2.0 |  1.60 |  2.0 | 73.9% |  16,034 |
| openbookqa       | 3,000 |  3.98 |  3.0 |  3.44 |  3.0 | 54.9% |  22,259 |
| arc_easy         | 2,000 |  4.93 |  4.0 |  4.90 |  4.0 | 68.8% |  19,650 |
| arc_challenge    | 1,000 |  6.26 |  6.0 |  6.12 |  6.0 | 63.6% |  12,376 |
| sciq             | 3,000 |  2.67 |  2.0 |  2.59 |  2.0 | 69.5% |  15,781 |
| commonsense_qa   | 3,000 |  2.17 |  2.0 |  2.19 |  2.0 | 69.3% |  13,073 |

Within each source, chosen and rejected means agree to within ~0.5 tokens and the medians agree exactly. **There is no per-source within-pair length cheat.**

### Estimated gradient-signal share (sum-of-logp DPO)

| source | pairs % | gradient tokens % | signal per pair (vs uniform) |
|---|---:|---:|---:|
| hellaswag        | 30.6% | **67.9%** | **2.22x** |
| winogrande       | 20.4% |  5.2% | 0.25x |
| openbookqa       | 12.2% |  7.2% | 0.59x |
| arc_easy         |  8.2% |  6.4% | 0.78x |
| arc_challenge    |  4.1% |  4.0% | 0.98x |
| sciq             | 12.2% |  5.1% | 0.42x |
| commonsense_qa   | 12.2% |  4.2% | 0.35x |

The "signal per pair" column is `(tokens_share / pairs_share)`: a value of 1.0 means a source contributes its fair share of gradient relative to its pair count. HellaSwag is at **2.22x** (massively over-weighted); WinoGrande is at **0.25x** (under-weighted by 4x); CommonsenseQA is at **0.35x**.

This is the load-bearing finding. Although we curated a 7-source mix where HellaSwag is only 30.6% of pairs by count, the sum-of-logp loss makes it dominate as if it were ~68% of the dataset. The short-form sources (WinoGrande, CommonsenseQA, SciQ) — which together are 44.8% of pairs — contribute only 14.5% of the gradient.

## Verdict

**Yes — retraining DPO with length-normalized log-probs is justified by this data.**

Reasoning:

1. The current sum-of-logp formulation reduces our 7-source curation to an effective 2-source mix (HellaSwag + everything else, with HellaSwag at ~2x the weight).
2. The under-weighted sources (WinoGrande, CommonsenseQA, SciQ, OBQA) cover commonsense and short-answer reasoning — exactly the domains where short-response benchmarks live (PIQA, HellaSwag-letter, etc.). Down-weighting them by 3-4x pushes the policy toward HellaSwag-shaped continuations and away from the diversity we paid for in pair curation.
3. There is no within-pair length bias to worry about, so length-normalization will not change the relative chosen-vs-rejected signal on any individual pair — it only reweights across pairs. This is a low-risk change: the per-pair preference direction is preserved; only the magnitude scales by `1 / response_len`.
4. The fix is a one-line change in `sequence_logprob_from_labels`: divide the masked sum by the per-row mask sum (clamped to >= 1 to avoid div-by-zero). The dpo loss formula and beta semantics are unchanged at the per-pair level; only the mean across pairs in a batch becomes more uniform across sources.

**Recommendation:** retrain DPO with a length-normalized variant. Suggested implementation: add a `length_normalize: bool` flag to the loss helper (default False to preserve the existing trainer's behavior for any in-flight runs), expose it via `dpo.length_normalize` in the config, and run a short A/B (e.g. 1k-step smoke) before committing to a full rerun. Beta will likely need re-tuning since the loss magnitude per pair drops by ~1/mean_response_len ≈ 1/6.

## Code change recommended

This audit identified a likely-impactful issue but does NOT itself include the
fix. The trainer change is intentionally separate work — see the recommendation
above. The rationale for not changing `loss.py` in the same commit:

- Existing in-progress DPO runs (and the dpo-letter pipeline that shares the
  same loss helper) depend on the current sum semantics. A silent change would
  invalidate their beta hyperparameters mid-run.
- The fix should be opt-in via a config flag and tested with a short A/B,
  not toggled globally on a one-shot edit.

## Files inspected

- `/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/data/posttraining/dpo_pairs_continuation/train.jsonl`
- `/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/data/posttraining/dpo_pairs_continuation/manifest.json`
- `/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/src/posttraining/dpo/loss.py`
- `/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/src/posttraining/dpo/trainer.py`
- `/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM/src/posttraining/dpo/prepare.py`
