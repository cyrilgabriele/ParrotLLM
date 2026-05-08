# DPO β-down: trade LAMBADA for hidden-benchmark gains

**Date:** 2026-05-08
**Branch:** `sft-dpo-gian`

## Goal

Produce a new submission checkpoint that wins more on the hidden benchmark
than it loses on LAMBADA, by pushing the existing `dpo_continuation` recipe
harder along a single axis: lower β.

This is motivated by a TA hint that (a) public MC scores are dominated by
the inference-side cloze setup ("just an inference trick"), (b) our LAMBADA
is currently the strong public score, and (c) the hidden benchmark rewards
something that DPO/SFT improves and that pulls against raw next-token
quality. Conclusion: it is acceptable to trade some LAMBADA for hidden gains.

## Baseline (what we are trying to beat)

The current submission `Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt`,
provenance:
- SFT base: `runs/posttraining/sft_benchmark/run_20260506_010816_sft_lr_5e-07/checkpoints/best_loss_0p9853_epoch_0000_step_0000300.pt`
- DPO: continuation-preference, β=0.1, lr=2e-6, 1 epoch, 24,500 pairs (no length-normalize), best at step 2900 (loss 0.4400).

Bench at limit=200:

| HellaSwag | WinoGrande | OpenBookQA | LAMBADA | Sum |
|---|---|---|---|---|
| 22.0 | 58.5 | 36.5 | 36.5 | 153.5 |

Length-normalized retrain (`dpo_continuation_lengthnorm`, same β=0.1, step 2600, loss 0.394) regressed to 149.5 — confirming the audit's prediction that length-norm rescales the loss by ~6× and effectively makes β=0.1 too constraining. Step 6000 of the same run failed to benchmark (OOM). We do not need that result to act.

## Approach (selected: Path A)

Single-axis change from the recipe that produced the submission: halve β.

| Knob | Submission | New |
|---|---|---|
| `beta` | 0.1 | **0.05** |
| `length_normalize` | off | off (unchanged) |
| `learning_rate` | 2.0e-6 | 2.0e-6 (unchanged) |
| `num_epochs` | 1.0 | 1.0 (unchanged) |
| Reference checkpoint | `sft_benchmark` step 300 | same |
| Pairs | `data/posttraining/dpo_pairs_continuation` (24,500) | same |
| `runs_dir` | `runs/posttraining/dpo_continuation` | `runs/posttraining/dpo_continuation_beta005` |

Halving β doubles the per-pair divergence budget. With the same data and lr,
this is the cleanest test of "push DPO harder."

### Approaches considered and rejected

- **Path B (LN + β=0.02):** would also fix the cross-source weighting bias the
  audit identified, but introduces two simultaneous changes vs. the known-good
  submission recipe. The empirical β-rescaling under LN is theoretical; we'd
  rather isolate the β knob first. Defer to a follow-up if A succeeds.
- **Path C (LN + β=0.02 + 2 epochs):** highest risk of preference-overfit /
  collapse. Two-epoch DPO on a 24.5k pair set typically degrades base
  capabilities. Skip until we have evidence A's gains are leaving room.

## Files to change

- **NEW** `configs/posttraining/dpo_continuation_beta005.yaml` — copy of
  `dpo_continuation.yaml` with `beta: 0.05` and `runs_dir: runs/posttraining/dpo_continuation_beta005`. No other deltas.

No source-code changes. No new datasets. Reuses the prepared pairs at
`data/posttraining/dpo_pairs_continuation/` (no `dpo-prepare` re-run).

## Execution

1. Write the new config.
2. `uv run python main.py --stage dpo --config configs/posttraining/dpo_continuation_beta005.yaml`
   (no `dpo-prepare` step — pairs already exist).
3. Pick the lowest-loss `best_loss_*.pt` from
   `runs/posttraining/dpo_continuation_beta005/run_*/checkpoints/`.
4. Benchmark on the leaderboard quick tier (`--limit 200`) using the same
   command pattern as `scripts/overnight_dpo_compare.sh`.
5. Compare the four scores against the baseline. Decision rule:
   - If sum > 153.5 with LAMBADA still ≥ 32 → swap into the submission.
     Update `Submissions/parrotlabs_parrotllm/runs/MANIFEST.md` provenance and
     re-upload to HF.
   - If sum > 153.5 but LAMBADA collapsed (< 28) → keep current submission;
     start a more conservative variant (β=0.07).
   - If sum ≤ 153.5 → discard, run Path B as the next experiment.

## Out of scope (explicit YAGNI)

- Changing the SFT base. We're testing one DPO knob.
- New preference data. The 24,500-pair set is reused as-is.
- Length-normalization. Deferred to a Path B follow-up if A wins.
- Any inference-side change to `Submissions/parrotlabs_parrotllm/main.py`.
- Hyperparameter sweeps. One run, one β value (0.05). If A fails we pick a
  next single value, not a grid.

## Risks

- **DPO collapse at β=0.05.** Lower β allows more divergence, which can let
  the policy drift into low-coherence regions. Mitigation: keep `eval_every: 100` and `keep_best_checkpoints: 2` so we can pick a mid-training checkpoint
  if late steps look unhealthy in the metrics.
- **Step 6000 lengthnorm bench was OOM-killed.** Same machine will run this
  training. The DPO trainer is memory-tested at 4 batch / 1024 ctx within the
  42 GiB MPS ceiling, but the user should close other heavy processes before
  starting (and the `max_train_pair_tokens: 512` cap is preserved).

## Success criterion

A new submission checkpoint with quick-tier sum > 153.5 and LAMBADA ≥ 32.
