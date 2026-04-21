# Supervised Fine-Tuning (SFT) — Technical Plan

**Pair A workstream for ParrotLLM / PikoGPT Challenge, FS2026.**
Synthesised from VL07 *Post-Training I: Supervised Fine-Tuning* (Handschuh,
Wiegand), the PikoGPT fact sheet, VL08 *RLHF & DPO* (context), Cyril's SFT
guide, and the relevant primary literature. Intended audience: a
PhD-student collaborator picking up the implementation cold.

---

## 0. Executive summary

The objective of SFT in this project is to take the 35.8 M-parameter
ParrotLLM base checkpoint (post-pretraining on OpenWebText, best
val-perplexity checkpoint from the 8 B-token run) and teach it to follow
instructions, i.e. produce assistant-format responses for user
prompts. The deliverable is an *Instruct* checkpoint that (i) improves
or at least preserves public-benchmark scores (HellaSwag, WinoGrande,
LAMBADA, OpenBookQA) on the course leaderboard, and (ii) serves as the
initialisation for the DPO stage run by Pair B.

The core technical operation is a next-token-prediction fine-tune of the
whole model (or LoRA adapters) on (instruction, response) pairs, with
the loss masked to the **response tokens only**. The parameter budget
(≤40 M total, LoRA weights included, see fact sheet) makes *full
fine-tuning* tractable on a single modern GPU; LoRA is therefore
optional and primarily useful for fast iteration on data-mix ablations.

The course's recommended recipe for this project is: **Alpaca-formatted
instruction data, small learning rate, early stopping, mix of pretraining
tokens to prevent catastrophic forgetting**. The recommended chat format
is the Stanford Alpaca template because it uses plain-text delimiters
(`### Instruction:`, `### Response:`) and requires **no new special
tokens in the GPT-2 tokenizer** — a material advantage given our
tokenizer is already extended by exactly one token (`<|pad|>`, vocab
50 258).

---

## 1. The alignment problem (VL07 §1)

A pretrained decoder-only LM optimises `P(next_token | context)`. Given
the prompt *"What is the capital of France?"*, the base ParrotLLM does
not answer — it **continues the text distribution**, often producing
more questions (*"What is the capital of Germany? What is the capital
of Italy? What is the largest city in…"*). This is not a bug: it is
exactly what cross-entropy pretraining optimises on web text, where
question-answer pairs are rare and question-question lists are common.

Alignment is the subset of post-training that changes this behaviour.
VL07 frames the pipeline as three capability layers (slide 7):

| Stage | Data | What it adds |
|-------|------|---------------|
| Pretraining | Raw web text (OpenWebText) | Language, world knowledge |
| **SFT** | Instruction-response pairs | **Format** — *"I answer in assistant format"* |
| RLHF / DPO | Pairwise preferences | **Quality** — *"I give helpful, safe answers"* |

VL07 quotes Ouyang et al. (2022, InstructGPT) as the quantitative
evidence that SFT alone is a large jump: human preference win rate
moves from ≈25 % (base) to ≈65 % (SFT). DPO / RLHF buys another
increment, but SFT carries the most weight. For our deliverable, SFT
alone is expected to recover most of the gap.

**Post-training ⊃ alignment.** VL07 slide 11 is explicit: post-training
is *everything* done after the pretrained base checkpoint; alignment is
the behavioural subset. For us, "post-training" and "alignment" are
effectively synonymous because we are not doing mid-training,
long-context adaptation, or RLVR (cf. the OLMo 3 diagram on VL07 slide
9, where post-training has many more branches).

---

## 2. SFT mechanics

### 2.1 Objective and loss

SFT is formally identical to pretraining with one change: the loss is
masked to the response tokens. Given a training example rendered as a
token sequence `[x_1, …, x_p, y_1, …, y_q]` where `x_{1:p}` are
instruction/system tokens and `y_{1:q}` are response tokens, the SFT
loss is

$$
\mathcal{L}_\text{SFT} = -\frac{1}{q}\sum_{t=1}^{q} \log P_\theta(y_t \mid x_{1:p}, y_{<t}).
$$

The model still sees the full prefix; only the gradient contribution
from the instruction positions is zeroed. In PyTorch this is done by
setting `labels[:p] = -100`, which is the sentinel value
`nn.CrossEntropyLoss` ignores (see VL07 slide 15; the convention comes
from HuggingFace `PreTrainedModel.forward`).

**Why mask?** VL07 slide 16 shows the empirical consequence. Without
masking, instruction tokens dominate gradient contribution in
Short-Q&A (≈40 %), Long-Context (≈70 %) and Multi-turn (≈60 %)
examples. The model then spends capacity learning to *generate*
instructions, which is the pretraining objective we already have —
wasted gradient. With masking, 100 % of gradient contribution comes
from response tokens, which is what we want the model to learn.

### 2.2 What changes between pretraining and SFT

The list below is exhaustive — nothing else is different at the
mathematical level:

1. Input format: concatenated `(instruction + response)` sequences
   instead of raw documents.
2. Labels: instruction positions set to −100 (masking).
3. Learning rate: typically 10× lower than pretraining peak (see §5).
4. Training duration: 1–3 epochs over a 10 k–100 k example dataset
   rather than a single pass over billions of tokens.

The optimiser, scheduler shape, precision, gradient clipping,
accumulation, and model architecture are unchanged. This is why SFT is
cheap and why the ParrotLLM training loop in `src/training/trainer.py`
can be re-used with only a new data loader and a label-masking
collator.

### 2.3 What SFT can and cannot do (VL07 §2)

| Can ✓ | Cannot ✗ |
|-------|----------|
| Follow instructions | Distinguish *good* from *great* answers |
| Answer in assistant format | Refuse harmful requests reliably |
| Stay on topic | Be calibrated about uncertainty |

The third column is the handover contract with Pair B: everything in
the "cannot" column is what DPO addresses by optimising on pairwise
preferences. We do not pursue calibration or refusal quality in SFT.

---

## 3. Data strategy

### 3.1 Source landscape (VL07 §2, Taori 2023, Wang 2023)

The canonical public SFT corpora cluster along two axes, scale and
quality (VL07 slide 18):

| Corpus | Size | Source | Quality |
|--------|------|--------|---------|
| Alpaca (Stanford) | 52 k | GPT-4 via Self-Instruct | Good |
| Dolly (Databricks) | 15 k | Human-written | High |
| OpenAssistant | 161 k | Community | Mixed-high |
| FLAN / Natural Instructions | ~1 M | Task-templated | Mid |
| WizardLM / evol-instruct | >1 M | GPT-4 evolved | Good-high |

Empirical guidance from the instruction-tuning literature (Zhou et al.
2023, *LIMA*, showed 1 k curated examples rival 52 k Alpaca on human
evaluation) is that **quality dominates quantity past ~1 k examples
for a ≤40 M model**. Our model is *far* below the scale at which
Alpaca's capacity starts to saturate, so dataset size is not our
binding constraint — data *format consistency* and
*benchmark-decontamination* are.

### 3.2 Recommended dataset choice

**Primary: Alpaca** (`tatsu-lab/alpaca` on Hugging Face Hub). VL07
slide 48 recommends it explicitly for PikoGPT because:
- it uses plain-text delimiters (no tokenizer vocab change),
- it is single-turn (matches the 1 024-token context limit), and
- it is well-studied, with published hyperparameters.

**Augmentation candidates** (if time permits, and only after the Alpaca
baseline is reproduced):
- A small Dolly slice (15 k, permissive licence) for distributional
  coverage — more factual short-answer density than Alpaca.
- OpenAssistant conversation trees, filtered to
  `lang == "en"` and single-turn, to diversify stylistic register.
- **Skip** the bigger synthetic corpora (UltraChat, WizardLM). At our
  model scale the marginal improvement per token is small and the
  contamination risk against the leaderboard benchmarks is higher.

### 3.3 Decontamination

Mandatory. The course leaderboard evaluates on LAMBADA, HellaSwag,
WinoGrande, OpenBookQA and *hidden* benchmarks (VL08 slide 29). If any
of these test splits leaks into SFT data, our reported scores are
invalid and, worse, we will not know. Re-use the
`src/data/preprocess.py` phase-1 SHA-1 hash machinery already in the
repo. Hash each (instruction + response) concatenated string at the
raw-text level, intersect with the benchmark test splits, drop hits.
Report the overlap count in the tech report.

### 3.4 Data mixing against catastrophic forgetting (VL07 §2)

Catastrophic forgetting (CF) is the standard failure mode of
full-parameter SFT at small scale: the model overfits to the
instruction distribution and loses pretraining knowledge, which shows
up as degraded Wikitext-103 perplexity and degraded world-knowledge
benchmarks. VL07 slide 25 lists five mitigations, which we apply in
the order of cost-effectiveness:

1. **Lower LR** (cheapest, biggest effect) — §5 below.
2. **Early stopping** on a held-out validation subset of SFT data.
3. **Mix pretraining tokens** into the SFT stream — e.g. 5–10 % of each
   batch drawn from the original OpenWebText bin files with standard
   next-token loss. Zhang et al. 2023 (*"What Makes a Good Instruction
   Fine-Tuning Dataset"*) and the OLMo 2/3 tech reports both do this.
4. **Weight regularisation** (L2 toward the base checkpoint's weights —
   "elastic weight consolidation" lite). Effective but fiddly.
5. **LoRA / PEFT** — by construction, the base weights are untouched,
   so CF on the base-model measurements is zero. See §7.

For the first run we do 1+2+3; 4 and 5 are fallbacks.

### 3.5 Instruction-data bootstrapping (Self-Instruct, Wang 2023)

Wang et al.'s Self-Instruct (VL07 slides 20–21) is how Alpaca was
generated: 175 seed tasks → GPT-4 expansion → filter/validate →
~52 k pairs. The pool-growth curve (slide 21) shows diminishing
returns after ~1 200 iterations (more duplicates). We do *not* need to
bootstrap anything for this project — Alpaca is already the output of
this pipeline — but it is worth understanding because (a) several
exam-style slides reference it, and (b) if we wanted to inject
PikoGPT-specific prompts (e.g. "explain a fintech concept to a
beginner") we would use exactly this pattern.

---

## 4. Chat template

### 4.1 Why templates matter (VL07 §3)

Without structural markers, the model cannot unambiguously tell *who is
speaking* or *where the response should end*. VL07 slide 28 shows the
failure mode: in a multi-turn dialog rendered as flat text, the model
generates continuations that conflate user and assistant turns.

### 4.2 Format options

| Format | Markers | Tokenizer impact |
|--------|---------|------------------|
| ChatML (OpenAI) | `<\|im_start\|>`, `<\|im_end\|>` | **New special tokens** added to vocab |
| LLaMA-2 (Meta) | `[INST]` / `[/INST]` + `<<SYS>>` | New special tokens |
| **Alpaca (Stanford)** | `### Instruction:` / `### Response:` | **None — plain text markers** |

All three work; the only *structural* requirement is that the template
used in training is byte-identical to the template used at inference
(VL07 slide 32, "Critical rule"). Diverging training and inference
templates silently degrades output quality because the model conditions
on prompt structure it has never seen before.

### 4.3 Choice for ParrotLLM

**Use the Alpaca template.** Rationale:

1. **No tokenizer change.** Our tokenizer is GPT-2 plus one added
   `<|pad|>` token, `vocab = 50 258`. The pretraining checkpoint has
   learnt embeddings for exactly this vocab. Adding ChatML's
   `<|im_start|>` and `<|im_end|>` would require growing the embedding
   matrix by 2 rows, randomly initialising them, and relying on SFT to
   teach the new rows useful representations. For a 40 M model with a
   small SFT budget, this is fragile; Alpaca's text-only markers
   sidestep it entirely.
2. **DPO compatibility.** Pair B's DPO pipeline inherits whatever chat
   format we ship. A plain-text format means they inherit the vocab
   unchanged too.
3. **Cyril's SFT guide** (`docs/post_training/sft_from_checkpoint_summary.md`)
   already assumes this direction — we continue without redirection.

The concrete training-time template:

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

For a variant with an optional `input` field (Alpaca's original
schema distinguishes "instruction" from "input context"):

```
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

At inference time, the prompt is rendered up to and including
`### Response:\n`, and generation stops at EOS or at a `###` token
sequence (whichever comes first). We store this template string verbatim
in `configs/default.yaml` under `tokenizer.chat_template` so there is
exactly one source of truth, shared by SFT training, DPO training, the
leaderboard runner, and the chat UI.

### 4.4 Loss-masking boundary

The mask boundary is the byte offset immediately after `### Response:\n`
in the rendered template. Tokens before this offset → label −100;
tokens at and after → their own ids. Two gotchas:

- The `\n` after `Response:` is part of the response region (we want
  the model to predict it).
- The tokenizer-level boundary can shift by one token if a whitespace
  merges across the boundary. Verify by printing a decoded example's
  `(token, label)` pairs for the first 30 positions. Do this once
  before the first real run; catch it in a `tests/test_sft_mask.py`.

---

## 5. Hyperparameters

The course gives one firm number ("use a small LR") and two soft
ones ("avoid catastrophic forgetting", "LoRA weights count toward
40 M"). The rest are standard instruction-tuning values cross-checked
against Alpaca's original config, the Raschka textbook, and the OLMo 2
tech report.

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Peak LR | **2e-5** | ≈1/10 of pretraining peak (3e-4). Alpaca used 2e-5 for LLaMA-7B; at our scale the safe range is 1e-5–5e-5. |
| Min LR | Peak × 0.1 = 2e-6 | Cosine floor. |
| Warmup | 100 steps (≈3 %) | Short because we're not starting from random weights. |
| Schedule | Cosine | Same as pretraining; WSD also fine. |
| Epochs | **2** (sweep {1, 2, 3}) | Alpaca original uses 3; we start at 2 because our base model is much smaller than LLaMA and more prone to overfitting. |
| Effective batch | 64 sequences (≈64 k tokens) | Matches pretraining. Use gradient accumulation if needed. |
| Sequence length | 1 024 | Hard constraint (context). Pack shorter examples with an EOS delimiter (optional, later). |
| Weight decay | **0.0 on new / 0.1 on existing** | No decay on any newly-initialised rows (none, if Alpaca); else 0.1 matching pretraining. |
| Gradient clip | 1.0 | Same as pretraining. |
| Precision | BF16 if available, else FP16+GradScaler | Same as pretraining. |
| Pretraining-mix ratio | 5–10 % tokens per batch from OpenWebText `.bin` | §3.4 mitigation 3. Disable on run 1, enable on run 2 to measure the delta. |

**Why we do *not* sweep β₁/β₂/ε of AdamW for SFT.** VL04 slide 28 (the
pretraining lecture) was explicit: "Do not tune β1, β2, or ε. Focus on
the learning rate." That guidance is even more binding in SFT because
our compute budget for sweeping is smaller. Keep optimiser state
unchanged from pretraining.

### 5.1 Expected training time

- 40 M model × 2 epochs × 52 k Alpaca examples × ≤512 avg tokens
  per example ≈ 5.3 × 10⁹ tokens = ~0.3 × pretraining token count.
- On a single RTX 5090 (Christof's box, available to the project) at
  ~300 k tok/sec for BF16 inference+backward on this model, this is
  ≈5 hours wall-clock. On 8×V100 the same job is ≈30–60 min. Neither
  is a binding constraint.

### 5.2 Logging / early-stopping signal

Track three numbers per validation step:

1. **SFT-val loss** on a 5 % held-out slice of the Alpaca training set
   — primary overfitting signal.
2. **Wikitext-103 PPL** — catastrophic-forgetting signal. Recompute
   every 200 steps using the existing `src/eval/perplexity.py`.
   If WT-103 PPL rises more than +5 % above the base checkpoint,
   drop the LR by 2× or trigger early stop.
3. **HellaSwag accuracy** on a 100-example subset, via the course's
   leaderboard runner — downstream signal. Checkpoint when this
   improves.

---

## 6. Catastrophic forgetting — operating bounds

VL07 slide 25 puts this on the slide-deck directly; our numerical
operating bound is:

- **Hard stop:** WT-103 test perplexity rising >10 % above base.
  Either the model is losing world knowledge or there is a data bug.
- **Yellow flag:** WT-103 rising 2–10 %. Acceptable for an SFT
  checkpoint so long as downstream benchmarks are improving; normal
  cost of alignment.
- **Green:** WT-103 unchanged or lower. Rare but ideal; typical
  when pretraining-mix is enabled at 10 %.

This is the project-specific decision rule that generalises VL07's
qualitative guidance into something measurable.

---

## 7. Full fine-tuning vs LoRA (VL07 §4)

### 7.1 The parameter-budget question

The fact sheet's 40 M cap applies to the *shipped* model (VL07 slide
48: "LoRA weights count as model weights"). Our base checkpoint is
35.76 M. An r=8 LoRA on the 4 attention projections (Q, K, V, O)
across 16 layers at d_model=320 adds `2 × 320 × 8 × 4 × 16 ≈ 328 k`
parameters — negligible, well under the 4.24 M headroom.

So **LoRA is permitted under the 40 M cap and saves no bits relative
to full FT at our scale.** The question is engineering, not budgetary.

### 7.2 When each is preferable

| Consideration | Full FT | LoRA (r=8) |
|---------------|---------|------------|
| Quality ceiling | Higher | ~90–98 % of full FT on format tasks |
| Training time | Baseline | ~0.7× (smaller backward pass) |
| VRAM for optimiser state | 2× weights (~80 MB + activations) | 2× LoRA only (~0.7 MB) |
| Catastrophic forgetting | Exists (§6) | Zero on base weights |
| Handoff to DPO | Base + modified weights | Base + adapter; DPO can start from merged or unmerged |
| Debugability | Standard | Extra layer of indirection |

**Recommendation:** run **full fine-tuning** as the primary method for
this project because (a) quality ceiling matters for the leaderboard,
(b) our model is small enough that the engineering overhead of LoRA is
not repaid, and (c) it simplifies Pair B's handoff — they inherit a
single-file checkpoint whose architecture is identical to the base.

Keep LoRA as a **fallback** if full FT exhibits catastrophic forgetting
we cannot mitigate in §6.

### 7.3 If we go LoRA

Defaults from Hu et al. 2021 and the PEFT library:

- Rank `r = 8` (the consensus default; VL07 slide 42).
- `alpha = 2r = 16`.
- Target modules: `{q_proj, k_proj, v_proj, o_proj}` (Q/K/V/O only).
- No dropout on the LoRA branch for our size (the regularising effect
  of low rank is already enough).
- Merge adapters into the base weights before shipping the checkpoint
  to Pair B, unless we explicitly decide to ship adapters separately.

**Important:** if we ship LoRA adapters *un-merged*, their weights
still count toward the 40 M cap per the fact sheet. Merged LoRA is
mathematically equivalent (no inference latency) and simpler to
account for.

---

## 8. Implementation plan

The implementation path below is the operational expansion of the
lecture material into the actual ParrotLLM repo. It extends Cyril's
existing `docs/post_training/sft_from_checkpoint_summary.md`.

### 8.1 Module layout

```
src/post_training/sft/
├── __init__.py
├── data.py          # load + normalise HF datasets → {instruction, response}
├── template.py      # Alpaca template render + mask-boundary helper
├── collator.py      # build labels with -100 on instruction tokens
├── trainer.py       # reuse src/training/trainer.py, override the loss path
└── __main__.py      # CLI entry wired from main.py --stage sft
```

Do **not** fork the training loop. Subclass or inject: the pretraining
loop is battle-tested (8 B-token run, distributed Optuna) and any
regression there breaks Pair B too.

### 8.2 Dataset pipeline

```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train")
# ds columns: instruction, input, output, text
# Normalise to internal schema:
def normalise(ex):
    return {
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "response": ex["output"],
    }
ds = ds.map(normalise, remove_columns=ds.column_names)
```

Decontaminate against LAMBADA / HellaSwag / WinoGrande / OpenBookQA
test splits using the phase-1 SHA-1 machinery from
`src/data/preprocess.py`. Log the overlap count.

Split 95 / 5 for train/val.

### 8.3 Template rendering and masking

```python
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)
ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def render(ex):
    prompt = (
        ALPACA_PROMPT_INPUT.format(**ex) if ex["input"]
        else ALPACA_PROMPT_NO_INPUT.format(**ex)
    )
    full = prompt + ex["response"] + tokenizer.eos_token
    return prompt, full

def encode(ex, tokenizer, max_len=1024):
    prompt, full = render(ex)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids   = tokenizer(full,   add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[:len(full_ids)]   # truncate mask to truncated sequence
    return {"input_ids": full_ids, "labels": labels}
```

### 8.4 Loss path

ParrotLLM's forward currently returns `(logits, loss)` where `loss` is
computed internally with `targets`. Add a second branch for
`labels`-style input (HF convention) so the collator's `-100` mask is
respected. Minimal diff: in `src/model/transformer.py::forward`, if
`labels` is given, compute `F.cross_entropy(logits.view(-1, V),
labels.view(-1), ignore_index=-100)`. Keep the existing `targets`
path for backward compatibility with the pretraining loop.

### 8.5 Pre-flight checks (adapted from VL04's "5-min sanity" guidance)

Before the full run, always:

1. **Decode one example's (token, label) pairs** — verify mask
   boundary is right.
2. **Forward pass on one batch** — confirm finite loss, gradients
   flow to all parameters.
3. **Overfit 1 batch for 200 steps** — loss must go to ≈0. If it
   doesn't, stop: it's not a hyperparameter issue, it's a data / mask
   / loss-path bug.
4. **Compare base vs. overfitted model on the same prompt** — outputs
   must differ qualitatively (the overfitted model should regurgitate
   the one response).

If these four pass, launch the real run.

### 8.6 Output: checkpoint and handoff

Save the SFT checkpoint in the same payload format as the pretraining
checkpoints (`config` + `model` dict). Add two fields:

- `training_stage: "sft"`
- `sft_metadata: {dataset, template_version, steps, val_loss, wt103_ppl}`

Ship to Pair B via:
1. Upload to the shared Hugging Face Hub dataset (training.hf_upload is
   already configured in the repo; see `configs/default.yaml`).
2. Notify Pair B in the weekly sync with the HF path and the
   leaderboard score delta vs. base.

### 8.7 Evaluation protocol

For the tech report we need three comparable numbers (base / SFT /
DPO). For SFT specifically:

- **Intrinsic:** val-loss, Wikitext-103 PPL, OpenWebText-eval PPL.
- **Extrinsic (leaderboard):** fork
  `https://github.com/unisg-ics-dsnlp/PikoGPT_Leaderboard`, put the
  SFT checkpoint in `submissions/ParrotLLM/runs/`, run
  `uv run python -m leaderboard.run_benchmarks --submission ParrotLLM
  --checkpoint runs/<ckpt>.pt --limit 100`, open a PR. The hidden
  benchmarks come back via GH. Baseline to beat (VL08 slide 29):
  Piko-Intelligence-Index = 300.
- **Qualitative:** 20 fixed prompts, side-by-side base vs SFT outputs
  rendered into `docs/post_training/sft_samples.md` for the poster.

---

## 9. Risks and mitigations

| Risk | Detection | Mitigation |
|------|-----------|-----------|
| Template drift between train/inference | Manual inspection of one decoded example; integration test | Single source of truth in `configs/default.yaml:tokenizer.chat_template` |
| Benchmark contamination in SFT data | Phase-1 hash intersection reports ≥1 % overlap | Drop matching rows before training |
| Catastrophic forgetting | WT-103 PPL rises >10 % vs base | Lower LR 2×; enable 10 % pretraining-mix; fall back to LoRA |
| Mask-boundary off-by-one | Overfit-1-batch test fails | Decode (token, label) pairs for 3 examples; fix the boundary |
| Wrong EOS / generation never stops | Leaderboard outputs run to max_tokens with garbage | Ensure `eos_token_id` is the tokenizer's EOS and the template ends with EOS during training |
| LoRA weights pushing us over 40 M | Parameter count script | Merge LoRA into base; prefer full FT (see §7.2) |
| Vocab mismatch with DPO pair | Handoff checkpoint's tokenizer-vocab assertion in DPO trainer | Use Alpaca format (no new tokens) → mismatch risk = 0 |

---

## 10. References

Mapping to VL07 slide numbers given where relevant.

- Ouyang, L. et al. (2022). *Training language models to follow
  instructions with human feedback.* NeurIPS. arXiv:2203.02155.
  **InstructGPT, the canonical 3-step alignment diagram.** (VL07 [1])
- Raschka, S. (2024). *Build a Large Language Model (From Scratch).*
  Manning. **Chapters 7 (SFT) and 8 (RLHF).** (VL07 [2])
- Taori, R. et al. (2023). *Stanford Alpaca: An Instruction-following
  LLaMA Model.* GitHub. **The dataset and prompt template we use.**
  (VL07 [3])
- Wang, Y. et al. (2023). *Self-Instruct: Aligning Language Models
  with Self-Generated Instructions.* ACL. arXiv:2212.10560.
  **Bootstrapping pipeline behind Alpaca.** (VL07 [4])
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large
  Language Models.* ICLR. arXiv:2106.09685. (VL07 [5])
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of
  Quantized LLMs.* NeurIPS. arXiv:2305.14314. (VL07 [6])
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment.*
  NeurIPS. arXiv:2305.11206. **Empirical argument for quality > quantity
  past ~1 k examples.** (not in VL07; core citation for §3.1)
- Groeneveld, D. et al. (2024). *OLMo: Accelerating the Science of
  Language Models.* arXiv:2402.00838. **Tech-report-quality write-up
  of an end-to-end base + SFT pipeline; practical reference for the
  evaluation protocol we mirror.** (OLMo 3 diagram on VL07 slide 9)

---

## 11. Open questions for the TAs

1. Are pretraining-mix batches (5–10 %) permitted during the SFT
   stage, or does the project expect SFT on instruction data alone?
   (Fact sheet is silent; VL07 slide 25 recommends the mix.)
2. If we ship a LoRA-adapted model, do we report `base.params +
   adapter.params` or the merged total? (VL07 slide 48 says the
   adapter counts; we want to confirm.)
3. Is the leaderboard runner tokenizer-sensitive — specifically, does
   it assume the default GPT-2 vocab of 50 257, or does it respect
   whatever tokenizer our submission ships? (Matters for the
   `<|pad|>` token we already added during pretraining.)

Flag these on Teams before the Week-9 exercise.
