# SFT v7 / v8 + cloze inference — final results & insights

Generated 2026-04-29. Documents the work landed on `sft-christof` between
the v6 baseline and the official PikoGPT_Leaderboard submission.

---

## Headline

**SFT v7 + PMI calibration: 33.60% public_avg on the official runner (n=500), #1 on the leaderboard by +8.20pp over the instructor baseline (25.40%).**

| Bench | Baseline | **SFT v7 + PMI** | Δ |
|-------|---------:|----------------:|------:|
| HellaSwag | 27.8% | **32.2%** | +4.4 |
| WinoGrande | 49.4% | **54.0%** | +4.6 |
| OpenBookQA | 22.6% | **25.0%** | +2.4 |
| LAMBADA | 1.8% | **23.2%** | +21.4 |
| **public_avg** | **25.40%** | **33.60%** | **+8.20** |

LAMBADA is the dominant gap: a one-line `rstrip` fix unblocked the entire
benchmark (it had been emitting underscore tokens because of trailing-space
BPE misalignment).

---

## Goal & constraints

- **Goal**: maximize `public_avg` on the four-benchmark PikoGPT_Leaderboard suite.
- **Architecture**: fixed at 40M params (d_model=384, n_layers=14, n_heads=6, d_ff=768, ctx=1024).
- **Pretrain**: fixed at 8B tokens on OWT subset `data/processed/filter_c`.
- **Hard rule**: **no benchmark validation data ever in training.** All synthetic data is decontaminated by SHA-1 against HellaSwag, WinoGrande, OpenBookQA, LAMBADA validation files.

---

## What's new vs v6

### Production fixes in `src/eval/inference.py`

1. **Cloze MC scoring** (`score_mc_options`): instead of greedily generating one letter token, the model now scores each option's *text* under the bare question stem and emits the letter of the highest-likelihood option. Substitution-cloze handles WinoGrande's `_` placeholder by substituting each option into the blank and scoring the post-blank tail.
2. **LAMBADA `rstrip` fix** (`run_inference`): leaderboard LAMBADA prompts arrive with a single trailing space. That trailing space breaks GPT-2 BPE alignment and collapses argmax onto the literal underscore token, taking LAMBADA from a working ~14-25% to 0%. One-line fix restored it.
3. **PMI calibration** (`pmi=True` default in `--leaderboard`): for each option, subtract the model's unconditional log-likelihood of the option text given a neutral `"Answer:"` prefix. Cancels the per-letter / option-text-frequency bias. Empirically +0.4pp on V7, +0.7pp on DPO v6, with no downside on substitution-cloze (WinoGrande gates PMI off internally for that path).

### New tooling

| Path | Purpose |
|------|---------|
| `tools/run_public_benchmarks.py` | Single-process leaderboard harness, ~50× faster than subprocess-per-question. Produces identical numbers to the official runner. |
| `tools/soup_checkpoints.py` | Weight-averaging tool. Refuses to soup ckpts with mismatched configs / state-dict keys. |
| `tools/build_auto_cloze.py` | LAMBADA-style cloze data generator from Wikitext-103 train, with hard SHA-1 decontamination against all four leaderboard validation files + Wikitext-103 test. |
| `tools/overnight_pipeline_v8.sh` | End-to-end overnight pipeline: PMI ablation → souping → SFT v8 training → benchmarks → official runner. |
| `tools/overnight_morning_brief.py` | Appends a TL;DR + ranked table to `OVERNIGHT_REPORT.md`. |

### New configs / data

- `configs/post_training/sft_v7_8b.yaml` — broader synthetic mix (HellaSwag-train + OBQA-train + WinoGrande-train + 800 cloze).
- `configs/post_training/sft_v8_8b.yaml` — v7 + 25k Wikitext-103 auto-cloze rows.
- `data/synthetic/sft_v7_combined.jsonl` (7.2k rows), `sft_v8_combined.jsonl` (32.2k rows), `sft_v8_auto_cloze.jsonl` (25k rows).

---

## Decisions log

### What worked

| Change | Lift | Rationale |
|--------|-----:|-----------|
| LAMBADA `rstrip` | LAMBADA 0% → 22.2% | BPE alignment fix at the prompt boundary. Pure inference change. |
| Cloze MC scoring | Roughly neutral on harness; cleaner failure modes (no invalid letters) | Robustness; matches lm-eval-harness convention. |
| PMI calibration | +0.4-0.7pp `public_avg` | Removes per-option surface bias; lifts OBQA from below-random (23.8) to random (25.0). |
| SFT v7 (broader synthetic mix) | +0.5pp over v6 | More HellaSwag/OBQA/WinoGrande-shaped training. |

### What didn't work

| Change | Result | Why we think so |
|--------|--------|-----------------|
| Auto-cloze SFT v8 (25k Wikitext-103 rows) | Tied v7 (33.4% on harness) | LAMBADA stayed at 22.0-22.2%. Probable cause: ~5-10% of generated rows had BPE sub-word targets (e.g. "guez" from "Rodríguez"); the noise crowded out the signal. Stricter sub-word filtering is the obvious next try. |
| Model souping (late v7 ckpts; v6+v7 cross) | ≤ v7 alone | Souping needs ingredients close to the same loss-basin minimum. Including v7's `best_step_900` (val_loss 2.47) dragged the average down from `final` (2.42). |

---

## Honest interpretation

The big wins shipped in the **daytime** session (cloze + LAMBADA `rstrip`) and took the official number from ~25% to ~33%. The **overnight** session added only **+0.6pp** (PMI calibration). Auto-cloze training and souping were a bust.

**OBQA at 25.0% = exactly random for 4-way MC.** This means the model has effectively no factual knowledge to discriminate options; PMI restored it to "guessing fairly" from "below random due to letter bias". For comparison: GPT-2 small (124M, 3× our params) only manages ~27% on OBQA. **Beating random on OBQA at 40M is an open problem that needs more parameters or distillation from a knowledge-rich teacher** — not something a one-night training run can crack.

---

## Practical ceiling

For 40M params / 8B tokens with the inference fixes above, **33-34% public_avg is realistic**. Pushing past 35% would require one of:

1. **Distillation** from a strong teacher (Llama-3-8B class) into our 40M student via Phi-style synthetic textbook data. Multi-day data prep + retraining.
2. **More pretraining tokens** (e.g., 30B+) — out of scope at the course level.
3. **Better synthetic SFT data** with stricter quality filtering and benchmark-shaped formats. Incremental, +1-2pp per try.

Architectural changes are unlikely to help at this size.

---

## Code & data references

- Final winning checkpoint: `runs/run_20260428_211931_sft/checkpoints/final_step_0001966_epoch_01_valloss_2p4231.pt`
- Official result JSONs: `Results/ParrotLLM/final_step_0001966_epoch_01_valloss_2p4231/` in the leaderboard repo.
- Full overnight execution log: `OVERNIGHT_REPORT.md` (in repo root).
- Harness sweep raw rows: `results/overnight_sweep.json`.
- Macbook setup for teammates: `docs/RUNNING_ON_MACBOOK.md`.
