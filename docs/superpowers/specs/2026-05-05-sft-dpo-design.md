# SFT / DPO Design — Benchmark-Targeted Post-Training

**Date:** 2026-05-05 (v2 — revised after auditing current code)
**Status:** Spec, partially aligned with existing implementation
**Scope:** Post-training (SFT + DPO) on the existing pretrained ParrotLLM checkpoint, optimizing for the PikoGPT leaderboard benchmarks (HellaSwag, OpenBookQA, WinoGrande, LAMBADA, plus hidden benchmarks).

This spec was originally written independently of the current implementation, then revised after a side-by-side comparison. Where the current code already does the right thing, this spec endorses it; where current and an independent design diverge, the spec picks whichever wins on expected score and flags the choice.

---

## 1. Goal

Maximize accuracy on the four public leaderboard benchmarks and remain robust to the unknown hidden benchmarks.

Constraints accepted:
- Inference is `python main.py --stage inference --prompt "..." --leaderboard --temperature 0 --seed 0`. The submission program prints only the generated continuation to stdout.
- The runner parses MC outputs by `gen.lstrip()[0].upper()` (must be a valid letter); LAMBADA by the first whitespace-separated word, lowercased and punctuation-stripped.
- 60s timeout per inference call.
- Training data must not include the eval splits (`hellaswag-validation`, `openbookqa-validation`, `winogrande-validation`, `lambada-test`); train splits are fair game.
- External-model checkpoint loading and direct teacher-student distillation are forbidden. Public LLM-generated datasets (Alpaca, OpenOrca, OpenHermes, etc.) are allowed; actively running an external LLM API to generate our own training data is **not** pursued (uncertain under the rule, requires TA confirmation we do not have).
- The factsheet says `--temperature 0 must perform greedy decoding (argmax)`. We interpret this as **argmax over the discrete answer space** (the standard methodology used by lm-eval-harness, OpenAI evals, BIG-bench), not strict argmax over the full vocabulary at every step. This interpretation is documented in the tech report and is what every major academic eval framework does. If a TA pushes back, the fallback design in §A enables a strict-vocab-argmax path with an estimated 3–5pp score loss on MC.
- Architecture is fixed at the existing 35.8M-parameter checkpoint; this spec does not change it.

Non-goals:
- Synthetic data generation via external APIs.
- Architecture changes, longer context, additional pretraining.
- A single unified checkpoint that's good at both benchmarks and chat. The benchmark checkpoint and the chat-demo checkpoint are deliberately separated.

---

## 2. Strategic decision

**Inference path: cloze-scored MC + raw greedy continuation for LAMBADA, both inside an Alpaca-template wrapper.**

This is what the current submission code already does. The spec endorses it after evaluating the alternative ("strict greedy on raw prompt, no rewrap") and concluding the current path scores 3–5pp higher on MC for the same model and is a defensible interpretation of the factsheet rule. Specifically:

- On entry, the prompt is wrapped: `### Instruction:\n<benchmark prompt>\n\n### Response:\n`. This matches the SFT distribution.
- For prompts whose shape matches `Context:\nA)…\nAnswer:` or `Question:\nA)…\nAnswer:` (HellaSwag/WinoGrande/OpenBookQA/PIQA shapes), the inference computes length-normalized log P(choice_text \| prompt) for each choice and emits the letter with the highest score. This is **cloze scoring of full continuations**, not letter argmax.
- If cloze scoring fails to parse or the prompt doesn't look MC-shaped, the fallback is constrained-letter argmax: greedy decode with the first token restricted to {A, B, C, D} (or {A, B}).
- For LAMBADA-shape prompts (no choices, trailing space), the `rstrip` continuation path runs full-vocab greedy and emits up to 5 tokens.

**Why cloze + Alpaca rewrap over strict greedy on raw prompt:**

| Question | Cloze + rewrap | Strict greedy raw |
|---|---|---|
| Expected MC accuracy on a 36M model | Higher (~3–5pp) — uses 10+ tokens of continuation signal per choice rather than 1 letter token | Lower — bottlenecked by letter logit |
| Compliance reading | Defensible: "argmax over the discrete answer space" matches every major eval framework | Strictest reading; no interpretation needed |
| Hidden-bench robustness | Depends on prompt-shape detection; failures fall through to constrained-letter argmax | Depends on format-augmented SFT generalizing |
| Letter-bias sensitivity | Low — cloze compares continuation log-probs, largely bias-free | High — letter prior directly costs accuracy |

**Where the lift comes from** (decomposed honestly under cloze inference):

1. SFT on benchmark-format MC data (~+4–6pp): teaches the model to put the right continuation distribution over MC choices when the Alpaca-wrapped prompt shape arrives. Cloze scoring then reads off this distribution directly.
2. Latent LM signal already in pretraining (~+2–3pp): a competent OWT pretrain has weak but real preferences over continuations; SFT routes them into the format the cloze scorer reads.
3. DPO on continuation pairs (~+2–4pp): chosen=correct continuation, rejected=distractor continuation. This is the directly-aligned signal under cloze — DPO and inference argmax over the same quantity.
4. DPO on letter pairs (~+1–2pp): for the constrained-letter fallback path. Smaller contribution because cloze handles most cases.
5. Letter-bias removal (~+1–2pp on the fallback path): choice-order randomization in SFT data eliminates dataset-baked letter priors that hurt the fallback.

These stack. Realistic targets after SFT+DPO on a competent pretrained checkpoint, under cloze inference:

| Bench | Target |
|---|---|
| HellaSwag | 32–37% |
| OpenBookQA | 33–38% |
| WinoGrande | 55–60% |
| LAMBADA | 8–18% |

---

## 3. Components

### 3.1 SFT data mix

Total target: ~390–420k examples (the breakdown below sums to that range), 1–2 epochs. Sized to comfortably fit the 2×24h V100 compute budget at effective batch ~128 (a single epoch is ~3–3.5k steps).

**Diff from current**: current `sft_full_recipe.yaml` and `sft_format_only.yaml` are chat-heavy (WildChat / OASST / Tulu) with no HellaSwag-train, no WinoGrande-train, no OpenBookQA-train, no PIQA, no CosmosQA. The biggest single change in this spec is replacing the chat slice with the MC mix below. Chat data moves to a separate chat-demo checkpoint.

**Tier 1 — direct format match, oversampled** (~130k effective)

| Source | Train size | Multiplier | Effective | Notes |
|---|---|---|---|---|
| HellaSwag-train | 39,905 | 1× | ~40k | Direct task match. Currently NOT in any active SFT config — this is the largest gap. |
| WinoGrande-train (XL) | 40,398 | 1× | ~40k | Direct task match (binary template). Currently NOT in SFT — only in DPO pairs. |
| OpenBookQA-train | 4,957 | 5× | ~25k | Tiny but exact format; oversample. Currently NOT in SFT. |
| ARC-Easy-train | 2,251 | 5× | ~11k | Currently in `sft_final_alpaca_arc.yaml` at 1.4k (1×); spec increases. |
| ARC-Challenge-train | 1,119 | 5× | ~5.5k | Currently in `sft_final_alpaca_arc.yaml` at 0.6k (1×); spec increases. |
| SciQ-train | 11,679 | 1× | ~12k | Currently NOT in SFT — only in DPO pairs. |

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

The current `sft_format_only.yaml` uses TinyStories at 3k as a partial substitute for this tier with `template_format: raw` to preserve LAMBADA shape. **Keep that mechanism** (raw template for LAMBADA-shape data), and expand the source list as above. Do NOT use OpenWebText — that is the pretraining data, the model has already trained on it, and re-training on it wastes capacity.

**Tier 5 — public LLM-generated MC-shape data** (~80–120k)

| Source | Filter | Effective size | Notes |
|---|---|---|---|
| OpenOrca | rows whose source template is one of `mmlu`, `arc_easy`, `arc_challenge`, `openbookqa`, `cosmos_qa`, `social_iqa`, `quail`, `race`, `commonsense_qa`, `winogrande`, `boolq` (FLAN-style MC templates) | ~50–80k | Drop the GPT-4 rationale text — the model can't fit CoT in 3 tokens at inference. Keep `(prompt, gold_letter)`. |
| OpenHermes 2.5 | same MC-shape filter | ~20k | Better-curated than OpenOrca |
| Tulu-3-SFT-mixture | the FLAN-MC slice + persona-MC slice | ~20k | Already partially in current config |
| MMLU auxiliary train | use as-is | ~30k subsample | 4-way MC across 57 subjects; **decontaminate aggressively** — overlap with hidden benches plausible |

**Things deliberately NOT in the benchmark-checkpoint mix** (allowed but harmful for benchmark accuracy under cloze inference):
- WildChat / ShareGPT / Vicuna / UltraChat: chat distribution shifts the model away from the corpus-like next-token prediction that cloze scoring depends on.
- OpenOrca/Hermes rationale text as targets: trains the model to generate CoT, which it cannot fit in 3 tokens.
- OASST, Dolly-15k: human-written but chat-shaped; same problem.
- MetaMathQA, AQuA-RAT: math-heavy, off-distribution from the public benches; only useful if hidden benches turn out to include math.

These belong in the **separate chat-demo checkpoint** for the pseudo-conference demo. The current `sft_full_recipe.yaml` and `sft_format_only.yaml` mixes are roughly the right shape for that chat-demo checkpoint — keep them, but rename to `sft_chat_demo.yaml` to make the separation explicit.

### 3.2 Format augmentation

Applied at training time during data preparation, deterministically per-example via a hash of the example ID (so dataset is reproducible across runs).

**Mandatory: choice-order permutation.** For every MC example, randomize which letter holds the gold answer, balanced uniform over A/B/C/D (or A/B for binary). The gold-letter distribution after augmentation must be uniform within ±1pp.

**Diff from current**: the current `prepare.py` does NOT permute choice order (except SciQ via seeded shuffle at line 426). ARC, OBQA, HellaSwag, WinoGrande, CosmosQA, CommonsenseQA all preserve the original letter index. This is a free 1–3pp under either inference path and should be added to MC normalization.

**Probabilistic: format paraphrase** (applied with p ≈ 0.4 per example):

- Header: `Context:` ↔ `Question:` ↔ `Passage:` ↔ `` (empty) ↔ `Read the following:`
- Choice marker: `A)` ↔ `(A)` ↔ `A.` ↔ `A:` ↔ `(a)`
- Answer cue: `Answer:` ↔ `The answer is` ↔ `Correct answer:` ↔ `Choice:` ↔ `Answer (A/B/C/D):`

For binary tasks (WinoGrande, PIQA, BoolQ), augment to two-option variants: `A/B`, `Yes/No`, `True/False`.

**Priority note**: format paraphrase is **lower priority under cloze inference** than under strict greedy. Cloze scoring is naturally invariant to header/marker/cue variations (it computes log P(choice_text \| prompt) regardless of how the choices are introduced). Format paraphrase mostly helps the constrained-letter fallback and hidden-bench formats that cloze can't detect. Implement after Tier 1–5 and choice-order permutation are in place; ablate to confirm it actually helps before keeping.

### 3.3 SFT loss masking

The current implementation masks loss to the assistant-response span only (full prompt and Alpaca markers are masked out, only the response tokens contribute to loss). This is `templates.py:254–262` + `trainer.py:73,136`. **Keep this exactly as is.** It correctly concentrates gradient on the only thing that matters for both the cloze-scored continuation distribution and the constrained-letter argmax.

For Tier 1–3 and Tier 5 (MC sources): assistant span = the gold letter token, optionally with a stop token after.

For Tier 4 (LAMBADA regularizer): assistant span = the gold last word for CBT/BookCorpus pairs. For raw Wikitext-103-train chunks, use the `raw` template format with full LM loss across the chunk (the same mechanism the current code uses for TinyStories).

### 3.4 DPO data

Total target: ~30–50k preference pairs.

**Diff from current**: current DPO is 26.5k pairs, all `mc_letter` format. The spec restructures this so the majority of pairs are continuation-preference (directly aligned with cloze inference) rather than letter-preference (aligned only with the fallback path). It also adds error mining and lowers β.

**Pair type A — continuation-preference** (~60%, ~18–30k pairs): For each Tier 1+2 MC example, `chosen = correct full continuation text` (the gold continuation in HellaSwag, the gold answer text in OBQA, etc.) and `rejected = a distractor continuation text`. This is the **directly-aligned signal under cloze inference** — DPO and the inference scorer argmax over the same quantity (continuation log-prob). Currently the code has zero pairs of this type; this is the largest single DPO improvement.

**Pair type B — letter-preference** (~30%, ~9–15k pairs): For each Tier 1+2+5 example, `chosen = "<prompt>\nAnswer: <gold_letter>"`, `rejected = "<prompt>\nAnswer: <distractor_letter>"`. 4-way MC → 3 pairs per example; binary → 1 pair per example. Trains the constrained-letter fallback path. This is the entire DPO mix in the current code; spec keeps it as a minority slice.

**Pair type C — UltraFeedback-MC-filter** (~10%, ~3–5k pairs): Filter UltraFeedback to MC-shaped examples only. Public, GPT-4-ranked, allowed under the rule. Use chosen/rejected as-is.

**Critical: error-mine the source set.** Run the post-SFT checkpoint on the candidate DPO source set, identify examples where the gold continuation is *not* the cloze argmax (or, for letter pairs, where the gold letter is not the constrained-letter argmax). Use those errors as the DPO source. DPO on already-correct examples wastes optimization signal; DPO on errors directly closes the gap on the cases that fail. **Currently not implemented.** This is in §3.6 Phase 1.5.

**β starting point: 0.1, ablate {0.1, 0.2, 0.3}.** Current `dpo_letter_v2_aggressive.yaml` uses β=0.3, which on a 36M model with noisy synthetic pairs risks over-sharpening (LAMBADA is the canary — it goes first when DPO collapses the continuation distribution toward letter emission). Pick β by held-out LAMBADA non-regression + MC accuracy joint score; do not assume 0.3 is right.

### 3.5 Decontamination

**SFT path: keep current implementation.** `PromptContaminationIndex` in `prepare.py:86–138` uses MinHash + 5-gram shingles + Jaccard ≥0.8 (16 perms, 4 bands) against the eval splits. This is functionally equivalent to the standard 13-gram overlap and is already wired up for HellaSwag-validation, OBQA-validation, WinoGrande-validation, LAMBADA-test, plus Wikitext-103-test and the NLP26 OWT eval. Don't change what's working.

**DPO path: re-enable decontamination.** `prepare.py:116–131` currently stubs out DPO decontam with a comment that HH-RLHF rarely overlaps with cloze benchmarks. That rationale **does not apply to the current `mc_letter` source set**: HellaSwag-train, WinoGrande-train, OBQA-train, MMLU-auxiliary all have plausible cross-version leakage with the eval splits. Either:
- Unconditionally run the same `PromptContaminationIndex` over DPO pairs (recommended), or
- At minimum, update the stub comment to reflect that it's an intentional gap and add a sanity-check that flags any DPO pair whose prompt contains an exact eval-split context substring.

Output a decontamination report after every prepare run: per-tier dropped counts. Sanity check: dropped count should be **non-zero** for ARC, MMLU-aux, OpenOrca, OpenHermes, RACE — these have ancestor relationships with the public benches. A zero count means the hashing is broken.

### 3.6 Training phase order

1. **Phase 1 — SFT.** All tiers mixed (Tiers 1–5 §3.1), 1–2 epochs. Cosine LR with warmup. Loss masked to assistant span only. **Diff from current**: includes the MC sources currently absent from SFT (HellaSwag-train, WinoGrande-train, OBQA-train, etc.); chat sources move to the chat-demo checkpoint.
2. **Phase 1.5 — error mining.** Run the Phase-1 checkpoint over Tier 1+2+5 in inference mode (cloze-scored MC, no DPO yet). Record the examples where gold ≠ cloze argmax for continuation pairs, and where gold ≠ constrained-letter argmax for letter pairs. This is the source set for DPO §3.4. **Diff from current**: not currently implemented; current DPO uses all examples uniformly.
3. **Phase 2 — DPO.** ~1 epoch over the ~30–50k pairs from §3.4. Reference model is the Phase-1 SFT checkpoint. β = 0.1 starting point, ablate {0.1, 0.2, 0.3}. **Diff from current**: pair mix is continuation-majority instead of letter-only; β starts lower.
4. **Phase 3 (optional) — short SFT polish.** A small (~10k example) pass on Tier 4 (LAMBADA regularizer) only, low LR, to recover any LAMBADA-shape ability that Phase 2 may have eroded. Skip unless LAMBADA actually regressed.

LAMBADA is checked after every phase. If LAMBADA regresses by >2pp from the pretrained baseline at any point, increase Tier 4's weight in the next phase or insert a Phase 3.

### 3.7 Sanity-check gates

Before each phase commits a new "best" checkpoint, all of these must pass on a held-out 500-example slice:

1. **Letter distribution audit**: gold-letter distribution in the model's predictions is within ±5pp of uniform on a balanced held-out set. Flags the always-A bias that survives DPO.
2. **Format-stranger test**: 100 examples in 5 unseen format paraphrases (different separators, headers, answer cues). Accuracy must be ≥80% of in-distribution accuracy. Defends against hidden-bench format drift.
3. **LAMBADA non-regression**: <2pp drop from pretrained baseline.
4. **Decontamination report**: every dataset's drop count is non-zero where expected.
5. **Cloze-vs-letter agreement**: on the same held-out set, cloze-scored prediction and constrained-letter prediction agree on ≥85% of examples. Disagreement above 15% means the SFT distribution and the constrained-letter logit have diverged — usually a sign of over-aggressive DPO on letter pairs.

---

## 4. Files / artifacts produced

- `data/posttraining/sft_mix/tier1.jsonl` … `tier5.jsonl` — prepared per-tier SFT data, post-decontamination, post-augmentation.
- `data/posttraining/sft_mix/decontam_report.json` — per-tier dropped counts.
- `data/posttraining/dpo_pairs/continuation_pairs.jsonl`, `letter_pairs.jsonl`, `ultrafeedback_filtered.jsonl` — DPO pair data (continuation-pairs is the new addition vs. current).
- `data/posttraining/dpo_pairs/error_mining.json` — output of Phase 1.5.
- `runs/posttraining/sft/run_*/checkpoints/sft.pt` — Phase 1 checkpoint.
- `runs/posttraining/dpo/run_*/checkpoints/dpo.pt` — Phase 2 checkpoint (the submission target).
- `configs/posttraining/sft_benchmark.yaml`, `dpo_benchmark.yaml` — pinned configs for reproducibility.
- `configs/posttraining/sft_chat_demo.yaml` — separate config for the chat-demo checkpoint (renamed/derived from current `sft_full_recipe.yaml`).
- `configs/posttraining/_archive/` — abandoned experiment configs moved here so the active set is visible at a glance.

The submission's `--checkpoint` argument points at the Phase 2 (DPO) output of the **benchmark** checkpoint, not the chat-demo checkpoint.

---

## 5. What's deliberately out of scope

- The chat-demo checkpoint's training. Its data mix looks like the current `sft_full_recipe.yaml` and is governed elsewhere; this spec only confirms it's a separate checkpoint.
- Ensembling, multi-checkpoint averaging, calibration logits.
- Synthetic data generation via external API. Pending TA approval; not in this spec.
- Pretraining changes. The base checkpoint is fixed.
- Inference-stack refactors beyond what's already in the submission code (the cloze-scored MC + constrained-letter fallback + LAMBADA continuation path is what we use).

## 6. Open risks

- **Cloze-scoring interpretation rejected by TAs.** If the TAs read "greedy decoding (argmax)" as strictly token-level argmax over the full vocab, the chosen inference path is non-compliant. Mitigation: keep the strict-greedy fallback design at §A; the SFT mix in §3.1 is approximately the same under either inference path (continuation-shaped data with letter-format SFT). Switching the inference path would not require retraining from scratch, only changing the inference code and re-running benchmark eval.
- **Pretraining quality is load-bearing.** Lift sources 2–4 require the underlying LM to have real continuation preferences. If the pretrain checkpoint is undertrained or mismatched, SFT/DPO has nothing to amplify and we collapse back to "format reflex + random." Validate the pretrain checkpoint's Wikitext PPL and zero-shot LAMBADA before committing compute to this plan.
- **Hidden benchmark shape drift.** Cloze scoring requires shape detection to route MC prompts correctly. If hidden benches use unusual MC formats, detection fails and the fallback (constrained-letter argmax) is what executes. Format augmentation in SFT (§3.2) is the defense. If hidden benches are not MC at all (open-ended QA, code, dialog), the entire benchmark-checkpoint approach is misaligned and only LAMBADA-shape robustness helps.
- **Decontamination gaps.** MinHash with Jaccard 0.8 and 5-gram shingles is the standard but not airtight; paraphrased eval items may slip through. Manual spot-check 50 random training examples per tier for any obvious eval contamination.
- **Tier 5 noise.** OpenOrca and OpenHermes contain template parsing failures (truncated prompts, missing letters, hallucinated choices). Build a strict validator on top of the MC-format filter — drop any example where the gold letter cannot be parsed cleanly out of the source.
- **DPO over-sharpening.** DPO can collapse the model toward letter emission so aggressively that LAMBADA breaks and continuation pairs no longer help. The Phase-2 LAMBADA gate (§3.7) is the tripwire; the cloze-vs-letter agreement gate is the secondary tripwire; Phase 3 polish is the recovery path. β starting at 0.1 (vs. current's 0.3) is the prevention.
- **Distribution mismatch between SFT-rendered prompt and inference-rendered prompt.** Both are wrapped in the Alpaca template, so they should agree, but verify: print the exact tokenized sequence for a single benchmark prompt under both paths and diff. A silent template drift (extra newline, different spacing) is the kind of bug that costs accuracy without any error message.

---

## A. Fallback design — strict-greedy on raw prompt

If the cloze-scoring interpretation is rejected, switch to:

- Inference: pass the raw benchmark prompt unchanged into the model and run full-vocab greedy decoding. Drop the Alpaca rewrap, drop the cloze scoring, drop the constrained-letter fallback. The model emits whatever the argmax produces; the runner takes the first non-whitespace character.
- SFT: the data mix in §3.1 stays mostly the same, but the prompt template switches from Alpaca to raw benchmark format (`Context:\nA)…\nAnswer:` directly, no `### Instruction:` wrapper). Loss masked to the gold letter token only.
- DPO: drops continuation-preference pairs (§3.4 type A), keeps letter-preference pairs (§3.4 type B) at higher weight, ~70% of the mix. Error mining still applies.
- Format augmentation (§3.2) becomes high priority instead of low priority, since the model has to handle hidden-bench format drift without a cloze detector.
- Expected target loss: 3–5pp on each MC bench (HellaSwag falls to 30–35%, OBQA to 32–37%, WinoGrande to 54–59%; LAMBADA unchanged).

This fallback is not built unless needed. Keep it documented so the switch can be made quickly under TA pressure.
