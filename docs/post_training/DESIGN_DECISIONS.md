# Post-Training Design Decisions

A complete, citation-anchored record of every design decision in the SFT and DPO stages on the `sft-christof` branch. Every entry follows the same structure: **decision** → **source** → **reason**. Sources are slide numbers from the FS2026 lectures (`docs/post_training/course_materials/VL07_*.pdf`, `VL08_*.pdf`), file paths with line numbers in this repo, the technical-plan document `docs/post_training/SFT.md`, the experiment-records files `docs/post_training/experiments_v3.md` and `v6_results.md`, or the original DPO paper (Rafailov et al. 2023).

This document is intended as the substrate for the tech-report's post-training chapter and as the single defence the team can fall back on if any specific knob is challenged.

---

## 0. Pipeline shape

### 0.1 Two-stage post-training: SFT first, then DPO

- **Source:** VL07 slide 7 ("Alignment Pipeline"); VL07 slide 12; FS2026 fact sheet p.1 ("2×2 split during post-training"); `docs/post_training/SPLIT_PROPOSAL.md` lines 6–11.
- **Reason:** SFT and DPO teach *different things*. SFT teaches **format** ("I answer in assistant format") via masked CE on instruction-response pairs; DPO teaches **quality** ("I prefer the better of two helpful answers") via the Bradley-Terry contrast on preference pairs. Both stages are course-mandated; shipping only one fails the assignment. Order is non-negotiable: DPO from the base checkpoint would have no signal until completions are already in assistant format, and DPO's frozen reference model must itself be on the assistant manifold (the DPO trainer initialises both policy and reference from the SFT checkpoint — `src/post_training/dpo/trainer.py:8–17`).

### 0.2 We do not do RLHF/PPO

- **Source:** VL08 slide 37 ("DPO is the **single most pragmatic choice** for your PikoGPT project"); VL08 slide 48 At-a-Glance row "For PikoGPT" (RLHF: "Likely overkill"; DPO: "Recommended"); VL08 slide 39 (Zephyr-7B: DPO on UltraFeedback matched 70B-RLHF on MT-Bench at a fraction of the cost).
- **Reason:** RLHF requires four models in memory simultaneously (policy, reference, reward model, value network — VL08 slide 35) plus a PPO loop; DPO halves both the memory footprint and the engineering surface (2 models, supervised loss). At our model scale and team size the additional quality ceiling RLHF buys is not worth the stability and engineering risk.

---

## 1. SFT — objective and loss

### 1.1 Cross-entropy loss, masked to response tokens

- **Source:** VL07 slides 14–15 ("SFT: The Core Idea" → "The Masked Loss"); `src/post_training/sft/collator.py` `IGNORE_INDEX = -100`; `src/model/transformer.py` forward path branch on `labels=...`; `docs/post_training/SFT.md` §2.1.
- **Reason:** SFT is mathematically identical to pretraining except instruction-token positions have their labels replaced by the sentinel `-100`, which `torch.nn.CrossEntropyLoss(ignore_index=-100)` skips. Without masking, instruction tokens dominate the gradient (VL07 slide 16 shows ~30–70 % depending on task type), which is wasted because instruction-generation is the pretraining objective the model already has. With masking, 100 % of gradient signal teaches "given this instruction prefix, produce *this* response" — the only behaviour change SFT actually targets.

### 1.2 Mask-boundary preflight assertion

- **Source:** `src/post_training/sft/trainer.py:518–534`; `docs/post_training/SFT.md` §4.4 "mask-boundary off-by-one".
- **Reason:** The single failure mode that silently destroys an SFT run is a mask-boundary bug: if the boundary is misplaced and lands after every response token, the loss sees zero supervised positions, gradients are zero, and the run trains for hours doing nothing. The preflight runs one batch through the collator, counts supervised tokens, and hard-fails if the count is zero. Cheap insurance against the worst-case silent failure.

---

## 2. SFT — chat template

### 2.1 Alpaca format as primary template

- **Source:** VL07 slide 32 ("Alpaca format is the simplest choice. No tokenizer changes required"); VL07 slide 48 ("Use Alpaca dataset for chat alignment through SFT"); `src/post_training/sft/template.py` `DEFAULT_ALPACA_TEMPLATE`.
- **Reason:** ChatML and Llama-2 templates would require adding new special tokens to the tokenizer and randomly initialising new embedding rows (VL07 slide 31 "Both require new special tokens"). For a ~40 M model with a small SFT budget, learning useful representations for cold-init rows in 1–2 epochs is fragile. Alpaca's plain-text markers (`### Instruction:`, `### Response:`) tokenise as already well-trained GPT-2 tokens, so no embedding-row surgery is needed and the project's 50 258-vocab tokenizer is preserved unchanged through the entire pipeline.

### 2.2 Critical rule: training and inference templates must match byte-for-byte

- **Source:** VL07 slide 32 ("Critical rule").
- **Reason:** The model conditions on prompt structure; if inference rendering differs from training rendering by even one character, the model is being asked to extrapolate to an OOD prefix and output quality silently degrades. Single source of truth lives in the rendering module so SFT, DPO, the chat REPL, and the leaderboard runner all see the same string.

### 2.3 Secondary "raw-completion" template added at v6

- **Source:** `src/post_training/sft/template.py` `RawCompletionTemplate`; `docs/post_training/v6_results.md`.
- **Reason:** The PikoGPT_Leaderboard runner uses a generation-based MC parser that takes the *first generated token* and matches it to `{A,B,C,D}`. With Alpaca-only training, the v5 model emitted Alpaca-formatted prose for raw-format prompts; first tokens were never letters; 70–100 % of MC prompts scored 0. The fix trains the model on both formats simultaneously (60 % Alpaca / 20 % synthetic raw / 20 % pretrain per batch); each *inference path* dispatches to the format it was trained on, satisfying the slide 32 critical rule once per path. Measured impact: leaderboard mean 3 % → 24.5 % from v5 to v6.

---

## 3. SFT — full fine-tuning vs PEFT

### 3.1 Full fine-tuning, not LoRA / QLoRA / Prefix Tuning

- **Source:** VL07 slide 35 ("For PikoGPT (~40M): full fine-tuning fits on one GPU"); VL07 slide 42 ("LoRA … not ideal for large domain shifts"); VL07 slide 48 ("LoRA weights count as model weights (max 40M)"); VL07 slide 25 (LoRA is the *fifth* CF mitigation).
- **Reason:** The cost arguments that motivate PEFT at 7 B+ scale do not apply at 35.76 M. A 32 GB RTX 5090 has ~140× memory headroom for full FT on this model. LoRA also buys no parameter-budget savings because adapters count toward the 40 M cap, and the lecture explicitly warns LoRA underperforms full FT on large domain shifts — and SFT is exactly such a shift (text-distribution → instruction-distribution). The remaining LoRA appeal (constructive zero-CF on base weights) is unnecessary because we measure CF directly via the WT-103 tripwire (§5).

---

## 4. SFT — hyperparameters

The SFT v6 8B-base config (`configs/post_training/sft_v6_8b.yaml`) is the canonical reference for the values below.

### 4.1 Optimiser: AdamW with `β1=0.9, β2=0.95, ε=default`

- **Source:** VL04 (pretraining lecture) "AdamW recipe card"; `src/post_training/sft/trainer.py:160–173` `_build_optimizer`; `configs/post_training/sft_v6_8b.yaml:62–63`.
- **Reason:** Identical to the pretraining optimiser; the lecture is explicit that β1/β2/ε should not be tuned during fine-tuning ("Focus on the learning rate"). Keeping these constant means the optimiser-state distribution at SFT-step-0 matches the regime the pretraining run ended in, removing one source of variance.

### 4.2 Weight decay: 0.1 on 2D parameters, 0.0 on 1D

- **Source:** `src/post_training/sft/trainer.py:160–173`; VL04 "AdamW recipe card"; `configs/post_training/sft_v6_8b.yaml:60`.
- **Reason:** Standard split. 2D params are weight matrices where L2 regularisation is meaningful; 1D params are biases and layer-norm scales/shifts where decay distorts the learned distribution. Same split as pretraining.

### 4.3 Learning rate: peak 1e-5, min 1e-6, cosine schedule with 50-step warmup

- **Source:** VL07 slide 25 ("Use smaller learning rates" — qualitative); VL07 slide 48 ("choose a small LR" — qualitative); SFT.md §5 ("≈1/10 of pretraining peak"); `configs/post_training/sft_v6_8b.yaml:58–59,64–66`.
- **Reason:** The course gives no explicit number, only "small LR." 1e-5 is ~1/30 of the pretraining peak (3e-4 in this project), more conservative than the original SFT.md §5 plan of 2e-5 because the v1/v2 SFT runs surfaced earlier signs of catastrophic forgetting than expected. Cosine schedule mirrors pretraining for consistency. Warmup is short (50 steps ≈ 3 % of an epoch) because we are not starting from random weights; the model is already in a sensible regime.

### 4.4 Effective batch size 64 sequences (= 8 × 8 grad-accum)

- **Source:** `configs/post_training/sft_v6_8b.yaml:67–68`; SFT.md §5 ("Effective batch 64 sequences ≈ 64 k tokens").
- **Reason:** Matches the pretraining effective batch, so optimiser dynamics at SFT-step-0 are continuous with the end of pretraining. The split into 8 micro-batches × 8 accumulation steps fits comfortably in 32 GB VRAM with 1024-token sequences and BF16 activations.

### 4.5 Sequence length 1024

- **Source:** `configs/post_training/sft_v6_8b.yaml:69`; FS2026 fact sheet (model context length).
- **Reason:** Matches the pretraining context length. Longer sequences would require positional-extrapolation tricks not in scope for the project. 99 % of Alpaca examples fit comfortably in 1024 tokens.

### 4.6 Epochs: 2 (with early stopping)

- **Source:** `configs/post_training/sft_v6_8b.yaml:65`; SFT.md §5 ("Alpaca original uses 3; we start at 2 because our base model is much smaller than LLaMA and more prone to overfitting"); VL07 slide 25 ("Early Stopping").
- **Reason:** Stanford Alpaca trained Llama-7B for 3 epochs. At 35.76 M parameters the model has ~200× less capacity, so the same data passes more times relative to capacity and overfit risk is higher. Two epochs plus the early-stopping patience (5 non-improving evals; `configs/post_training/sft_v6_8b.yaml`) gives the run room to converge while bounding the CF risk.

### 4.7 Mixed precision: BF16 on Ampere+ (RTX 5090)

- **Source:** `src/post_training/sft/trainer.py:178–185` `_autocast_for`; VL04 / VL06 "Double the Speed" guidance.
- **Reason:** BF16 has the same exponent range as FP32 but half the storage; on Ampere+ hardware it gives a ~2× throughput win without the FP16 numerical instability that requires a `GradScaler`. Older hardware automatically falls back to FP16 + GradScaler; CPU runs to plain FP32. This mirrors the pretraining trainer exactly so checkpoints are interchangeable across stages.

### 4.8 `torch.compile(mode='default')`, not `reduce-overhead`

- **Source:** `src/post_training/sft/trainer.py:444–456`; commit `9c78d3a`.
- **Reason:** `reduce-overhead` uses CUDA graphs which fail when the model is called in different modes within one run (train, eval, WT-103 probe) — the documented failure is "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run." Default mode keeps the kernel-fusion gain (~1.4–1.8× on this size) without the CUDA-graph trap.

### 4.9 Gradient clip 1.0

- **Source:** `configs/post_training/sft_v6_8b.yaml:61`; VL04 "Gradient Clipping" guidance.
- **Reason:** Standard. Loss spikes are rare with masked SFT but the clip is a no-cost tail-risk hedge against any single batch with anomalously high gradient magnitude.

---

## 5. SFT — catastrophic-forgetting controls

### 5.1 Wikitext-103 perplexity tripwire at +10 % over base

- **Source:** `src/post_training/sft/trainer.py:113–131` `_wt103_should_stop`; `docs/post_training/SFT.md` §6.
- **Reason:** The lecture (VL07 slide 25) names CF qualitatively and prescribes early stopping but gives no numerical bound. The tripwire operationalises this into a measurable rule: WT-103 is OOD-Wikipedia-prose, never seen during pretraining, so a perplexity rise indicates lost world knowledge rather than instruction overfit. Threshold is baseline-relative (`(current − baseline)/baseline`) so it is invariant to base-checkpoint, tokenizer, and sequence-length changes across runs.

### 5.2 Pretraining-data mix at 20 % of batches

- **Source:** VL07 slide 25 mitigation #1 ("Mix pretraining data with SFT data"); SFT.md §3.4 ("5–10 % of each batch"); `configs/post_training/sft_v6_8b.yaml` `pretraining_mix_ratio: 0.20`.
- **Reason:** Each batch has a 20 % chance of being drawn from the pretraining `.bin` file with the standard next-token loss instead of the masked SFT loss. This keeps the model's gradients periodically pulled toward the pretraining manifold, reducing CF cost. The actual ratio (20 %) is higher than the original SFT.md plan (5–10 %) because v3+ measurements showed pretraining mix is the most effective single CF mitigation per unit of training time, so we leaned on it harder.

### 5.3 Early stopping: patience = 5 non-improving evals

- **Source:** VL07 slide 25 mitigation #2 ("Early Stopping"); `src/post_training/sft/trainer.py:134–142` `_should_stop_early`; `configs/post_training/sft_v6_8b.yaml` (patience field).
- **Reason:** Saves the best-validation-loss checkpoint and stops if 5 consecutive eval cycles fail to improve on it. This is the gentler CF defence: if the WT-103 tripwire (§5.1) is the hard stop, early stopping is the soft stop that catches "model converged on the SFT objective; further training adds no value but accumulates CF cost."

### 5.4 NaN / Inf guard on loss

- **Source:** `src/post_training/sft/trainer.py:145–155` `_is_nonfinite`; `src/post_training/sft/trainer.py:572–587`.
- **Reason:** A single non-finite loss propagated into AdamW corrupts the first- and second-moment buffers permanently — every subsequent step produces useless updates with no error message. The guard checks before `.backward()`, discards the in-flight accumulation window if a NaN is seen, and logs a warning. Particularly relevant when the pretraining-mix path produces all-padded sequences or when an empty supervised-token batch sneaks through.

---

## 6. SFT — data

### 6.1 Primary corpus: Stanford Alpaca (52 k pairs)

- **Source:** VL07 slide 22 "Quick Poll" answer (course's recommendation: "Mix of GPT-4 + open data"); VL07 slide 32 + slide 48 ("Use Alpaca[3] dataset for chat alignment through SFT"); `configs/post_training/sft_v6_8b.yaml:46–47`.
- **Reason:** Alpaca is GPT-4-distilled, single-turn, well-studied with published hyperparameters, and uses no special tokens. The lecture pre-poll recommendation is "C: mix of both" (slide 22) — synthetic + open-data — and Alpaca is the synthetic side of that mix. Quality dominates quantity past ~1 k examples for a 40 M model (LIMA, Zhou et al. 2023), so 52 k is more than sufficient.

### 6.2 Synthetic raw-format mixin (~2.4 k rows, v6 only)

- **Source:** `data/synthetic/sft_v6_combined.jsonl`; `tools/build_synthetic_mc_programmatic.py`; `tools/build_synthetic_mc_public.py`; ADR-0001.
- **Reason:** See §2.3 above. ~851 programmatic factual-trivia rows (capitals, colours, arithmetic, animal classes, chemistry, synonyms, Winogrande-style) plus ~1 500 reformatted from public Q&A *train* splits (SciQ, ARC-Easy, ARC-Challenge, CommonsenseQA). Train-split-only; test-split decontamination is enforced (§6.4).

### 6.3 Per-batch mixture: 60 % Alpaca / 20 % synthetic raw / 20 % pretrain

- **Source:** `docs/post_training/v6_results.md` line 18; `configs/post_training/sft_v6_8b.yaml` (`synthetic_oversample: 7`, `pretraining_mix_ratio: 0.20`).
- **Reason:** 20 % of batches are pretraining (CF defence, §5.2). The remaining 80 % is Alpaca + synthetic in 3:1 ratio, achieved by oversampling the ~2.4 k synthetic rows 7× to reach ~17 k effective rows, which together with ~52 k Alpaca rows yields 25 % synthetic share within the SFT pool — i.e. 20 % of all batches. The arithmetic is documented in the config header.

### 6.4 Decontamination against benchmark test splits

- **Source:** SFT.md §3.3 ("Mandatory"); `src/post_training/sft/data.py` decontam pipeline; `configs/post_training/sft_v6_8b.yaml:50–58`.
- **Reason:** The leaderboard scores models on LAMBADA, HellaSwag, WinoGrande, OpenBookQA plus hidden benchmarks (VL08 slide 29 references "Piko-Intelligence-Index"). If any test text leaks into SFT data, scores are invalid. The pipeline computes a SHA-1 hash of each (instruction + response) string and intersects against the four visible benchmark test sets plus six hidden-bench-safe candidates (ARC Easy/Challenge, BoolQ, CommonsenseQA, SciQ, MMLU). Hits are dropped before training; counts are logged.

### 6.5 95 / 5 train / val split

- **Source:** `configs/post_training/sft_v6_8b.yaml:48`; SFT.md §8.2.
- **Reason:** Conventional split. Validation is used for the early-stopping signal and the best-checkpoint criterion. Held-out fraction is small because the train signal is what we want to maximise, not held-out predictive power per se.

---

## 7. DPO — objective and loss

### 7.1 Bradley-Terry contrastive loss

- **Source:** VL08 slides 13–14, 33; `src/post_training/dpo/trainer.py:74–113` `dpo_loss`.
- **Reason:** The Bradley-Terry preference model `P(A ≻ B) = σ(R(A) − R(B))` translates pairwise human preferences into a smooth, differentiable training signal. DPO substitutes the implicit reward `r(y) = β log π(y)/π_ref(y)` into Bradley-Terry, yielding a supervised loss on the policy directly without training a separate reward model — VL08 slide 33's full derivation. The asymmetric loss curve (slide 14: steep gradient on wrong rankings, weak gradient on correct ones) means the model learns fastest from its mistakes and is naturally stable on already-correct pairs.

### 7.2 Length-normalised per-sequence log-probs

- **Source:** `docs/post_training/experiments_v3.md`; `src/post_training/dpo/trainer.py` `per_sequence_logp` with `length_normalize_logp=true`.
- **Reason:** The orca_dpo_pairs dataset has rejected responses ~51 % longer than chosen on average (78.8 % of pairs), so summed log-probs systematically reward shorter completions regardless of preference quality — VL08 slide 50 reward-hacking glossary entry "length exploitation." Dividing each per-sequence log-prob by completion-token count removes the bias. The textbook DPO loss (VL08 slide 33) uses sums; we deviate deliberately and document the deviation. Validated by controlled single-variable experiment (`experiments_v3.md`): val_acc fell from 99 % (length-cheating artefact) to ~80 % (true preference signal, consistent with Zephyr-7B numbers — VL08 slide 39).

### 7.3 Frozen reference model is the SFT checkpoint

- **Source:** `src/post_training/dpo/trainer.py:8–17` ("POLICY (π_θ) … initialised from the SFT checkpoint passed via --checkpoint. REFERENCE (π_ref): a frozen, eval-mode copy of the same SFT checkpoint, used only for log-prob computation. No grad flows through it"); VL08 slide 32 ("Reference Model (Frozen SFT)"); VL08 slide 50 glossary "Reference Model".
- **Reason:** DPO's implicit-KL anchoring `β log π/π_ref` is a regulariser pulling the policy back toward the reference. The reference must be on the assistant manifold or the regulariser fights the loss. The SFT checkpoint is the natural choice: it has just been taught to produce assistant-format responses, so anchoring there means "stay assistant-shaped, just prefer the chosen completion."

---

## 8. DPO — hyperparameters

The DPO v6 config (`configs/post_training/dpo_v6_8b.yaml`) is the canonical reference for the values below.

### 8.1 β = 0.2 (KL leash strength)

- **Source:** `configs/post_training/dpo_v2_balanced.yaml` header (DPO v1 at β=0.1 over-drifted: WT-103 PPL +1.5, val_acc 98.9 %); VL08 slide 19 ("The Tug-of-War Objective"); VL08 slide 21 ("The Math of the Tether"); Rafailov et al. 2023 reports β ∈ [0.1, 0.5] as the typical range.
- **Reason:** β scales the implicit KL leash; too low and the policy drifts off the reference manifold, too high and the policy cannot move at all. The DPO v1 run at β=0.1 measurably drifted (the v2 config header records the failure); doubling β to 0.2 doubled the leash strength and brought drift back into the operating bound. β=0.2 is well inside the published-paper range.

### 8.2 LR: peak 1e-6, min 1e-7, cosine

- **Source:** `configs/post_training/dpo_v6_8b.yaml:51–52`; SFT.md §5 (DPO LR ≈ 1/10 of SFT); VL08 slide 21 ("not allowed to change too drastically").
- **Reason:** DPO's contrast can produce large gradients (the asymmetric Bradley-Terry curve has steep gradient on wrong rankings) and the policy must stay on the reference manifold. A peak LR of 1e-6 — 10× lower than SFT's 1e-5 — gives the optimiser many small steps in which the model can adjust its preference ranking without destabilising the broader distribution. The v3-vs-v5 measurements showed lower LR consistently improves the WT-103 drift number without sacrificing val_acc.

### 8.3 Weight decay: 0.0

- **Source:** `configs/post_training/dpo_v6_8b.yaml:53`.
- **Reason:** DPO already has an explicit regulariser in the form of the implicit KL leash to the reference. Adding L2 weight decay on top double-regularises and slows convergence without measurable benefit in the v3 ablation runs. The pretraining and SFT stages still use 0.1 on 2D params; DPO is the exception because of the KL anchor.

### 8.4 Effective batch 32 (= 4 × 8 grad-accum), 1 epoch

- **Source:** `configs/post_training/dpo_v6_8b.yaml:60–61,64`.
- **Reason:** DPO needs 4 forward passes per batch (policy×{chosen,rejected} + reference×{chosen,rejected}, the latter under `no_grad`), so micro-batch is half SFT's. Effective batch of 32 sequences is enough for stable preference signal at our scale (~12 k pairs ÷ 32 ≈ 380 steps per epoch). One epoch is sufficient because (a) the dataset is small and over-training risks length-cheating even with normalisation, (b) early stopping (patience=2) catches convergence anyway.

### 8.5 WT-103 tripwire tightened to +5 % for DPO

- **Source:** `configs/post_training/dpo_v4_length_norm_low_lr.yaml:17`; `configs/post_training/dpo_v6_8b.yaml:67` (5 %); ADR-0002.
- **Reason:** DPO's contrastive loss can move the policy off the reference manifold faster than SFT's masked CE — the asymmetric Bradley-Terry gradient (VL08 slide 14) plus orca_dpo_pairs' length asymmetry make even length-normalised DPO more drift-prone than SFT. Tighter bound means the run halts earlier on CF onset, preserving more of the SFT base's world knowledge. SFT keeps the looser 10 % bound.

### 8.6 Early-stop patience: 2 evals (vs SFT's 5)

- **Source:** `configs/post_training/dpo_v6_8b.yaml:73`.
- **Reason:** DPO converges faster than SFT (smaller dataset, contrastive signal) and the val_loss-vs-step curve typically peaks earlier. Patience of 2 catches the peak without wasting compute on the long flat tail. SFT's higher patience (5) is for the more gradual SFT loss curve.

---

## 9. DPO — data

### 9.1 Intel/orca_dpo_pairs (~12 k pairs)

- **Source:** `configs/post_training/dpo_v6_8b.yaml:24`; `src/post_training/dpo/data.py:6–10` ("12k Alpaca-style pairs of GPT-4 vs original-LLaMA responses; small enough that our 35M-param model can iterate without long wall-clocks").
- **Reason:** Three properties drove the choice:
  1. **Format match.** Pairs are already in `{prompt, chosen, rejected}` shape with Alpaca-compatible single-turn prompts, so the DPO collator can reuse the SFT Alpaca template (slide 32 critical rule applied twice — once per completion).
  2. **Scale match.** ~12 k pairs is appropriate for our model size; UltraFeedback (~64 k pairs, VL08 slide 40) and Anthropic HH (~170 k, VL08 slide 11) are designed for 7 B+ models and would inflate iteration time without proportional quality gain at 35.76 M parameters.
  3. **GPT-4 vs base-Llama framing.** The chosen/rejected contrast is GPT-4 quality vs LLaMA-7B quality — well-aligned with the "format already learned, now learn quality" goal (VL07 slide 7 → DPO arrow).

The trade-off is the length asymmetry that motivated ADR-0003. UltraFeedback would have been more length-balanced; we chose to fix the loss instead of the data.

### 9.2 Decontamination shared with SFT pipeline

- **Source:** `src/post_training/dpo/data.py:30–33` (imports `build_decontam_index`, `filter_contaminated` from `src/post_training/sft/data.py`).
- **Reason:** Same SHA-1 intersection logic as SFT, against the same 10 benchmarks. Single source of truth — if the SFT decontam set changes, DPO inherits the change automatically.

### 9.3 Alpaca-rendered prompts even for DPO

- **Source:** `src/post_training/dpo/template.py` `DPO_DEFAULT_TEMPLATE`; VL08 slide 32 critical rule.
- **Reason:** The SFT model was taught Alpaca format. DPO's frozen reference is the SFT model. If DPO rendered prompts in any other format, the reference's log-probs would be on OOD prefixes and the implicit-KL anchor would be uninformative. Using the same Alpaca template keeps both policy and reference on-distribution.

---

## 10. Evaluation

### 10.1 Three independent evaluation pillars

- **Source:** `docs/post_training/experiments_v3.md` ("Same metrics across all experiments (factsheet §4.3)"); FS2026 fact sheet §4.3; `docs/post_training/v6_results.md`.
- **Reason:** No single metric captures both "did the model learn instructions" and "did it lose world knowledge" and "is it actually useful in chat." Three pillars:
  1. **Pillar 1 — Perplexity** on Wikitext-103 test + OpenWebText val. Detects catastrophic forgetting.
  2. **Pillar 2 — Multiple-choice accuracy** on LAMBADA / HellaSwag / WinoGrande / OpenBookQA at n=500 via the official PikoGPT_Leaderboard runner. Primary deliverable.
  3. **Chat usability** via `tools/brutal_test.py` — a 27-prompt hand-graded probe covering open-ended, math/counting, yes/no, factual recall, etc. Sanity check that the model is conversational, not just benchmark-fit.

### 10.2 Leaderboard contract: PikoGPT_Leaderboard runner with `--limit 100/500`

- **Source:** SFT.md §8.7; `docs/post_training/v6_results.md`.
- **Reason:** Course-mandated evaluation harness. Forking and PR-ing into the official repo means our scores are computed by the same code as the baseline's, with the same prompt format and decoding policy. `--limit 100` for fast iteration; `--limit 500` for the final reportable number.

### 10.3 Submission checkpoint = SFT v6 (not DPO v6)

- **Source:** `docs/post_training/v6_results.md` line 41 ("**SFT v6 outperforms DPO v6 on the leaderboard runner.** DPO was trained only on Alpaca-format pairs, so it slightly biases the model away from raw-format MC completion. sft_v6 is the right submission checkpoint").
- **Reason:** Measured outcome. DPO trained only on Alpaca-rendered pairs (orca_dpo_pairs) so its generation distribution drifts slightly away from the raw format the leaderboard runner needs. SFT v6 was the last stage that saw the raw format, so its MC parsing accuracy is higher. This is a project-level decision documented in `v6_results.md` and revisitable in v7 if a length-balanced, raw-format-friendly preference dataset becomes available.

---

## 11. Open questions / future-work flags

These are decisions that were *not* made on this branch but are worth flagging for any v7 revision:

1. **Switch DPO data to UltraFeedback** (VL08 slide 40 explicit recommendation; would remove length-asymmetry root cause rather than patching it via length-normalisation). Cost: ~5× larger dataset → ~5× DPO wall-clock; uncertain whether 35.76 M model benefits from the additional 50 k pairs.
2. **OpenBookQA letter bias** (`v6_results.md` "Open issue"): the v6 model never picks A on OBQA. Possible mitigations are listed in `v6_results.md` lines 73–80; not addressed in v6.
3. **Tightening the SFT WT-103 bound to 5 %** to match DPO. SFT v2–v6 were healthy at 10 %; the question is whether 5 % would change v7's CF margin without aborting useful runs.
4. **Multi-turn data**. Current pipeline is single-turn (Alpaca, orca_dpo_pairs). VL07 slide 30 covers multi-turn structure but the project does not need it. Out of scope for this submission; relevant if the chat REPL is extended post-deadline.
