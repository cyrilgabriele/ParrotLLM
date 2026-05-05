# SFT / DPO Design — Benchmark-Targeted Post-Training

**Date:** 2026-05-05
**Status:** Spec, not yet implemented
**Scope:** Post-training (SFT + DPO) on the existing pretrained ParrotLLM checkpoint, optimizing for the PikoGPT leaderboard benchmarks (HellaSwag, OpenBookQA, WinoGrande, LAMBADA, plus hidden benchmarks).

This spec was developed independently of the current SFT/DPO implementation. It is anchored on the benchmark mechanics and course constraints, not on the existing code.

---

## 1. Goal

Maximize accuracy on the four public leaderboard benchmarks and remain robust to the unknown hidden benchmarks, under the strict factsheet reading: the model performs autoregressive **greedy decoding (argmax over the full vocabulary at every step)** at inference time, with `--temperature 0`.

Constraints accepted:
- Inference is `python main.py --stage inference --prompt "..." --leaderboard --temperature 0 --seed 0`. The submission program prints only the generated continuation to stdout.
- The runner parses MC outputs by `gen.lstrip()[0].upper()` (must be a valid letter); LAMBADA by the first whitespace-separated word, lowercased and punctuation-stripped.
- 60s timeout per inference call.
- Training data must not include the eval splits (`hellaswag-validation`, `openbookqa-validation`, `winogrande-validation`, `lambada-test`); train splits are fair game.
- External-model checkpoint loading and direct teacher-student distillation are forbidden. Public LLM-generated datasets (Alpaca, OpenOrca, OpenHermes, etc.) are allowed; actively running an external LLM API to generate our own training data is **not** pursued (uncertain under the rule, requires TA confirmation we do not have).
- Architecture is fixed at the existing 35.8M-parameter checkpoint; this spec does not change it.

Non-goals:
- Cloze-scoring inference (would be higher accuracy but is out of scope under the strict-greedy interpretation we chose).
- Synthetic data generation via external APIs.
- Architecture changes, longer context, additional pretraining.

---

## 2. Strategic decision

**Inference path: Option 1 — raw greedy on the raw benchmark prompt.**

The submission's `inference --leaderboard` mode passes the prompt unchanged into the model and runs full-vocabulary argmax decoding. No prompt rewrap, no constrained decoding, no cloze scoring. The model must learn to emit the correct letter (MC) or correct word (LAMBADA) as its first non-whitespace generated content.

**Why Option 1 over Option 2 (rewrap-then-greedy):**
- Hidden benchmarks may not be format-detectable; rewrap requires correct detection or the model sees an unfamiliar distribution.
- Auditable in one sentence in the tech report ("inference passes the prompt unchanged and runs greedy decoding").
- Anything Option 2 could encode in its rewrap can equivalently be encoded as a format variant in SFT data (Tier 5b below).

**Where the lift comes from** (decomposed honestly):
1. Format reflex (~+5pp from baseline): model emits a letter at all instead of running into invalid generations.
2. Pattern matching (~+2–4pp): SFT teaches the easier commonsense patterns.
3. Latent LM signal routed via DPO (~+2–4pp): the pretrained model already has some preference between continuations; DPO trains the model to express that preference as the letter logit.
4. Letter-bias removal (~+1–2pp): choice-order randomization + DPO on all distractors prevents always-A bias from costing accuracy.

These stack, not redundantly. Realistic strict-greedy targets after SFT+DPO on a competent pretrained checkpoint:

| Bench | Target |
|---|---|
| HellaSwag | 30–35% |
| OpenBookQA | 32–37% |
| WinoGrande | 54–59% |
| LAMBADA | 8–18% |

---

## 3. Components

### 3.1 SFT data mix

Total target: ~390–420k examples (the breakdown below sums to that range), 1–2 epochs. Sized to comfortably fit the 2×24h V100 compute budget at effective batch ~128 (a single epoch is ~3–3.5k steps).

**Tier 1 — direct format match, oversampled** (~130k effective)

| Source | Train size | Multiplier | Effective | Notes |
|---|---|---|---|---|
| HellaSwag-train | 39,905 | 1× | ~40k | Direct task match |
| WinoGrande-train (XL) | 40,398 | 1× | ~40k | Direct task match (binary template) |
| OpenBookQA-train | 4,957 | 5× | ~25k | Tiny but exact format; oversample |
| ARC-Easy-train | 2,251 | 5× | ~11k | OBQA-shape, science |
| ARC-Challenge-train | 1,119 | 5× | ~5.5k | Harder OBQA-shape |
| SciQ-train | 11,679 | 1× | ~12k | Crowdsourced science 4-way |

**Tier 2 — commonsense breadth** (~70k)

| Source | Train size | Use | Notes |
|---|---|---|---|
| CosmosQA-train | 25,262 | full | 4-way narrative commonsense |
| PIQA-train | 16,113 | full | Binary physical commonsense |
| SocialIQa-train | 33,410 | subsample to ~15k | 3-way social commonsense |
| CommonsenseQA-train | 9,741 | full, subsample 5→4 | 5-way → drop hardest distractor |

**Tier 3 — MC reasoning** (~30k)

| Source | Train size | Use | Notes |
|---|---|---|---|
| RACE-train | 87,866 | subsample to ~20k | Reading comprehension; full size would dominate |
| LogiQA-train | 7,376 | full | Logical reasoning, 4-way |
| QASC-train | 8,134 | full, 8→4 | Subsample distractors to 4 |

**Tier 4 — LAMBADA regularizer / continuation preservation** (~40k)

| Source | Construction | Use | Notes |
|---|---|---|---|
| Children's Book Test (CBT) | use as-is (cloze format) | ~10k | Purpose-built for last-word prediction |
| BookCorpus passages | construct `(passage_minus_last_word, last_word)` pairs | ~20k | Sample 256–512-token passages, mask the final word |
| Wikitext-103-train | raw 256–512-token chunks | ~10k | Plain LM-loss regularizer; do NOT use Wikitext-103-test |

**Do NOT use OpenWebText** as a regularizer here — that is the pretraining data, the model has already trained on it, and re-training on it wastes capacity that should go to bench-aligned signal.

**Tier 5 — public LLM-generated MC-shape data** (~80–120k)

| Source | Filter | Effective size | Notes |
|---|---|---|---|
| OpenOrca | rows whose source template is one of `mmlu`, `arc_easy`, `arc_challenge`, `openbookqa`, `cosmos_qa`, `social_iqa`, `quail`, `race`, `commonsense_qa`, `winogrande`, `boolq` (FLAN-style MC templates) | ~50–80k | Drop the GPT-4 rationale text — the model can't fit CoT in 3 tokens at inference. Keep `(prompt, gold_letter)`. |
| OpenHermes 2.5 | same MC-shape filter | ~20k | Better-curated than OpenOrca |
| Tulu-3-SFT-mixture | the FLAN-MC slice + persona-MC slice | ~20k | Already partially in current config |
| MMLU auxiliary train | use as-is | ~30k subsample | 4-way MC across 57 subjects; **decontaminate aggressively** — overlap with hidden benches plausible |

**Things deliberately NOT in the mix** (allowed but harmful for benchmark accuracy):
- WildChat / ShareGPT / Vicuna / UltraChat: chat distribution shifts the model away from corpus-like next-token prediction.
- OpenOrca/Hermes rationale text as targets: trains the model to generate CoT, which it cannot fit in 3 tokens.
- OASST, Dolly-15k: human-written but chat-shaped; same problem.
- MetaMathQA, AQuA-RAT: math-heavy, off-distribution from the public benches; only useful if hidden benches turn out to include math.

These belong in a **separate chat-demo checkpoint** (different objective, different model) used for the pseudo-conference demo, not in the benchmark checkpoint.

### 3.2 Format augmentation

Applied at training time during data preparation, deterministically per-example via a hash of the example ID (so dataset is reproducible across runs).

**Mandatory: choice-order permutation.** For every MC example, randomize which letter holds the gold answer, balanced uniform over A/B/C/D (or A/B for binary). The gold-letter distribution after augmentation must be uniform within ±1pp.

**Probabilistic: format paraphrase** (applied with p ≈ 0.4 per example):

- Header: `Context:` ↔ `Question:` ↔ `Passage:` ↔ `` (empty) ↔ `Read the following:`
- Choice marker: `A)` ↔ `(A)` ↔ `A.` ↔ `A:` ↔ `(a)`
- Answer cue: `Answer:` ↔ `The answer is` ↔ `Correct answer:` ↔ `Choice:` ↔ `Answer (A/B/C/D):`

For binary tasks (WinoGrande, PIQA, BoolQ), augment to two-option variants: `A/B`, `Yes/No`, `True/False`.

The model learns "MC-shaped prompt → emit gold letter token" as a generalized reflex, not a specific format match. This is the primary defense against unknown hidden-bench formats.

### 3.3 SFT loss masking

For Tier 1–3, 5: loss is computed **only on the gold answer token (and any whitespace/newline preceding it)**. The full prompt (context, choices, "Answer:") is in the input but masked out of the loss. This concentrates gradient on the only thing that matters for letter argmax.

For Tier 4: loss on the masked-out last word (CBT, BookCorpus pairs) or full LM loss (raw Wikitext-103-train chunks).

### 3.4 DPO data

Total target: ~30–50k preference pairs.

**Pair type A — letter-preference** (~70%, ~21–35k pairs): For each Tier 1+2+5 example, build `chosen = "<prompt>\nAnswer: <gold_letter>"`, `rejected = "<prompt>\nAnswer: <distractor_letter>"` for each wrong letter. 4-way MC → 3 pairs per example; binary → 1 pair per example.

**Critical: error-mine the pair source.** Run the post-SFT checkpoint on the training data, identify examples where the gold letter is *not* the argmax under greedy decoding, and use only those examples for DPO. DPO on already-correct examples wastes optimization signal; DPO on errors directly closes the logit gap on the cases that actually fail.

**Pair type B — continuation-preference** (~20%, ~6–10k pairs): For each Tier 1+2 example, `chosen = correct full continuation text` (e.g., the gold continuation in HellaSwag) and `rejected = a distractor continuation text`. This sharpens the underlying continuation-preference signal that the letter logit is downstream of (lift source #3 from §2). Worth the addition because continuation pairs and letter pairs train somewhat orthogonal capabilities.

**Pair type C — UltraFeedback-MC-filter** (~10%, ~3–5k pairs): Filter UltraFeedback to MC-shaped examples only. Public, GPT-4-ranked, allowed under the rule. Use chosen/rejected as-is.

### 3.5 Decontamination

Run **before** any training. For every candidate example across Tiers 1–5:
1. Compute SHA-1 hash of the example's `context` (or `question`, depending on schema) after lowercasing and stripping whitespace.
2. Compare against the hash set of:
   - HellaSwag-validation contexts
   - OpenBookQA-validation question stems
   - WinoGrande-validation contexts
   - LAMBADA-test contexts (full passages)
3. Drop any candidate whose hash matches.
4. Also run a 13-gram overlap check (per HELM/lm-eval-harness convention) — drop candidates with ≥13 contiguous tokens overlapping with any eval-split context. This catches paraphrases that change a few words.

Output a decontamination report: per-tier dropped counts. Sanity check: dropped count should be **non-zero** for ARC, MMLU-aux, OpenOrca, OpenHermes, RACE — these have ancestor relationships with the public benches. A zero count means the hashing is broken.

### 3.6 Training phase order

1. **Phase 1 — SFT.** All tiers mixed, 1–2 epochs. Cosine LR with warmup.
2. **Phase 1.5 — error mining.** Run the Phase-1 checkpoint over Tier 1+2+5 in inference mode (greedy, no DPO yet). Record the examples where gold ≠ argmax. This is the source set for DPO pair type A.
3. **Phase 2 — DPO.** ~1 epoch over the ~30–50k pairs from §3.4. Reference model is the Phase-1 SFT checkpoint. Beta ≈ 0.1 (standard DPO default; tune on a held-out slice).
4. **Phase 3 (optional) — short SFT polish.** A small (~10k example) pass on Tier 1 only, low LR, to recover any LAMBADA-shape ability that Phase 2 may have eroded. Skip unless LAMBADA actually regressed.

LAMBADA is checked after every phase. If LAMBADA regresses by >2pp from the pretrained baseline at any point, increase Tier 4's weight in the next phase or insert a Phase 3.

### 3.7 Sanity-check gates

Before each phase commits a new "best" checkpoint, all of these must pass on a held-out 500-example slice:

1. Letter distribution audit: gold-letter distribution in the output is within ±5pp of uniform.
2. Format-stranger test: 100 examples in 5 unseen format paraphrases (different separators, headers, answer cues). Accuracy must be ≥80% of in-distribution accuracy.
3. LAMBADA non-regression: <2pp drop from pretrained baseline.
4. Decontamination report: every dataset's drop count is non-zero where expected.

---

## 4. Files / artifacts produced

- `data/posttraining/sft_mix/tier1.jsonl` … `tier5.jsonl` — prepared per-tier SFT data, post-decontamination, post-augmentation.
- `data/posttraining/sft_mix/decontam_report.json` — per-tier dropped counts.
- `data/posttraining/dpo_pairs/letter_pairs.jsonl`, `continuation_pairs.jsonl`, `ultrafeedback_filtered.jsonl` — DPO pair data.
- `data/posttraining/dpo_pairs/error_mining.json` — output of Phase 1.5.
- `runs/posttraining/sft/run_*/checkpoints/sft.pt` — Phase 1 checkpoint.
- `runs/posttraining/dpo/run_*/checkpoints/dpo.pt` — Phase 2 checkpoint (the submission target).
- `configs/posttraining/sft_benchmark.yaml`, `dpo_benchmark.yaml` — pinned configs for reproducibility.

The submission's `--checkpoint` argument points at the Phase 2 (DPO) output.

---

## 5. What's deliberately out of scope

- The chat-demo checkpoint. That's a separate model with a chat-style SFT mix (OASST/Dolly/Tulu-3) and is not used for benchmark inference.
- Ensembling, multi-checkpoint averaging, calibration logits.
- Cloze inference. If we ever revisit the strict-greedy decision, the SFT mix above would still be approximately right but the DPO design and the inference path would change.
- Synthetic data generation via external API. Pending TA approval; not in this spec.
- Pretraining changes. The base checkpoint is fixed.

## 6. Open risks

- **Pretraining quality is load-bearing.** Lift sources 2 and 3 require the underlying LM to have real continuation preferences. If the pretrain checkpoint is undertrained or mismatched, SFT/DPO has nothing to amplify and we collapse back to "format reflex + random." Validate the pretrain checkpoint's Wikitext PPL and zero-shot LAMBADA before committing compute to this plan.
- **Hidden benchmark format drift.** Format augmentation defends against this but not perfectly. If hidden benches are e.g. open-ended QA without explicit choices, this entire approach is misaligned.
- **Decontamination gaps.** 13-gram overlap is the standard but not airtight; paraphrased eval items may slip through. Manual spot-check 50 random training examples per tier for any obvious eval contamination.
- **Tier 5 noise.** OpenOrca and OpenHermes contain template parsing failures (truncated prompts, missing letters, hallucinated choices). Build a strict validator on top of the MC-format filter — drop any example where the gold letter cannot be parsed cleanly out of the source.
- **DPO over-sharpening.** DPO can collapse the model toward letter emission so aggressively that LAMBADA breaks. The Phase-2 LAMBADA gate (§3.7) is the tripwire; Phase 3 is the recovery path.
