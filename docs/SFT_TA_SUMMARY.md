# SFT Training Summary for TA Discussion

This document summarizes the supervised fine-tuning (SFT) work done so far, based on `docs/CHANGELOG.md`, `configs/posttraining/`, `data/posttraining/`, and `runs/posttraining/`.

## High-Level Status

The SFT pipeline works end to end: public datasets can be downloaded, normalized into a shared chat schema, packed, trained with assistant-only loss, checkpointed, and evaluated. The main issue is behavioral quality. The small ParrotLLM checkpoint can learn the chat format and produce assistant-like text, but it still fails simple factual or logical probes and often collapses into repetition.

The most useful probe so far is:

```text
### System:
You are ParrotLLM, a helpful assistant.

### User:
What is the capital of Paris?

### Assistant:
```

The question is intentionally flawed: Paris is a city, not a country, so a good model should say something like "Paris does not have a capital; Paris is the capital of France." The SFT models usually start with the right surface form but then loop or produce confused claims.

## Experiments at a Glance

| Experiment | Data size | Learning rate | Epochs | Replay ratio | Z-loss | Polish |
|---|---:|---:|---:|---:|---:|---|
| Smoke test | 16 examples | `1e-4` | very small smoke run | replay eval only | `0.0` | yes, `5e-5` smoke polish |
| First full local SFT | 18,775 examples | mainly `1e-4`, some `5e-5` | `1.0` | `0.0` | `0.0` | no |
| Full recipe preserved | 18,775 examples target mix | `5e-5`, `1e-4`, `2e-4` | `1.0` | `0.1` | `0.0` | yes, `0.25` epochs |
| Deep-burn narrow mix | 6,278 examples | `1e-5` | `3.0` | `0.3` | `0.0` | no |
| Deep-burn + z-loss | 6,278 examples | `1e-5` | `3.0` | `0.3` | `1e-4` | no |
| LIMA-style prepared mix | 1,000 examples | `1e-5` configured | `1.0` configured | `0.3` configured | `0.0` | no |
| Polish variant configured | 6,278 examples | `1e-5` configured | `1.0` + `1.0` polish | `0.3` configured | `0.0` | yes, 500-example subset |

The most important trend is that reducing LR and adding replay improved stability compared with the `1e-4` runs, but it did not solve repetition or factual/premise-correction failures. The LIMA-style and polish variants are configured/prepared, but no clearly corresponding completed non-smoke run was found in the current workspace.

## Base Model

- Base checkpoint used for recent SFT:
  - `runs/posttraining/base_import/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt`
- Model config used in SFT configs:
  - vocab size: `50258`
  - context length: `1024`
  - `d_model: 384`
  - `n_layers: 14`
  - `n_heads: 6`
  - `d_ff: 768`
  - dropout: `0.0151`
  - EOS/BOS: GPT-2 end-of-text token `50256`

## Data Used

### Original Full Public SFT Mix

This was the first full SFT dataset mix, preserved in earlier run snapshots such as:

- `runs/posttraining/sft/run_20260423_111010_sft_lr_0p0001/manifest_snapshot.json`

Prepared examples:

| Split | Examples | Packed sequences |
|---|---:|---:|
| train | 16,897 | 5,918 |
| dev | 939 | 331 |
| test | 939 | 334 |
| total | 18,775 | 6,583 |

Source composition:

| Source | Target | Kept | Why it was included |
|---|---:|---:|---|
| `wildchat_gpt4` | 5,000 | 5,000 | Realistic user prompt distribution and multi-turn chat behavior. |
| `oasst1_ready` | 4,500 | 3,275 | Human-reviewed dialogue branches; useful for cleaner assistant style. |
| `tulu_flan_v2` | 4,000 | 4,000 | Short exact-answer and benchmark-shaped tasks. |
| `tulu_persona_if` | 2,500 | 2,500 | Instruction following and constraint obedience. |
| `tulu_persona_reasoning` | 1,500 | 1,500 | Lightweight reasoning without long chain-of-thought traces. |
| `tulu_structured_outputs` | 1,500 | 1,500 | JSON, extraction, and strict-format behavior. |
| `pku_safe_rlhf_refusals` | 1,000 | 1,000 | Concise safe refusals for harmful prompts. |

Notes:

- `OASST1` underfilled because only 3,282 usable candidates were available after filtering and best-branch selection.
- Decontamination drops were `0` for all listed sources in this manifest.
- The full mix was chosen to cover chat realism, human-reviewed answers, exact answer tasks, instruction following, reasoning, structured output, and safety.

### Current Narrow Mix

The current `data/posttraining/sft_mix/manifest.json` is narrower than the original full mix. It keeps only `OASST1` and `Tulu FLAN`.

Prepared examples:

| Split | Examples | Packed sequences |
|---|---:|---:|
| train | 5,650 | 2,479 |
| dev | 314 | 131 |
| test | 314 | 135 |
| total | 6,278 | 2,745 |

Source composition:

| Source | Target | Kept | Why it was included |
|---|---:|---:|---|
| `oasst1_ready` | 4,000 | 3,278 | Human-reviewed dialogue trees for cleaner assistant behavior. |
| `tulu_flan_v2` | 3,000 | 3,000 | Short exact-answer/classification examples to protect factual QA behavior. |

This narrower mix matches the later "deep burn" attempt from the changelog: reduce noisy sources, lower LR, add replay, and train longer.

### Small LIMA-Style Mix

Config: `configs/posttraining/sft_exp_b_lima.yaml`

Prepared examples in `data/posttraining/sft_mix_lima/manifest.json`:

| Split | Examples | Packed sequences |
|---|---:|---:|
| train | 900 | 247 |
| dev | 50 | 14 |
| test | 50 | 15 |
| total | 1,000 | 276 |

Source composition:

| Source | Target | Kept | Purpose |
|---|---:|---:|---|
| `oasst1_ready` | 750 | 750 | High-quality human-reviewed chat. |
| `tulu_flan_v2` | 250 | 250 | Short exact-answer behavior. |

No matching `runs/posttraining/sft_lima/` directory was present, so this appears prepared/configured but not trained in the current workspace.

### Smoke-Test Mix

Used only to validate pipeline mechanics, not model quality.

Prepared examples in `data/posttraining/sft_mix_smoke_fast/manifest.json`:

| Source | Kept |
|---|---:|
| `wildchat_gpt4` | 4 |
| `oasst1_ready` | 4 |
| `tulu_flan_v2` | 4 |
| `pku_safe_rlhf_refusals` | 4 |

Total: 16 examples before split, 8 train examples, 2 train packed sequences.

## Data Processing Choices

All SFT examples are normalized into a shared message schema and rendered with this frozen template:

```text
### System:
...

### User:
...

### Assistant:
...
```

Important processing choices:

- assistant-only loss: user/system tokens are masked out, assistant answer tokens are trained
- EOS token is now included in the assistant loss mask, so the model can learn when to stop
- examples are filtered, deduplicated, decontaminated, split into train/dev/test, then packed
- decontamination references include Wikitext-103 test, OWT eval, HellaSwag, WinoGrande, OpenBookQA, and LAMBADA
- exact prompt-template symmetry matters strongly for this small model; inference must use the same headers/newlines as SFT

## Training Configurations Tried

### Smoke SFT

Config: `configs/posttraining/sft_smoke_fast.yaml`

Purpose: verify that download/prepare/train/polish/checkpoint logic works.

Observed summary from `data/posttraining/sft_mix_smoke_fast/sft_training_summary.json`:

| Field | Value |
|---|---|
| LR | `1e-4` |
| best run | `runs/posttraining/sft_smoke_fast/run_20260423_092238_sft_lr_0p0001` |
| best dev loss | `3.2104` |
| replay PPL | `23.94` |
| format score | `0.0` |
| polish | ran at `5e-5`, best dev loss `3.2084` |

### First Real Local Profile

From the changelog and early run snapshots.

| Field | Value |
|---|---|
| data | original 18,775-example full public mix |
| LR | mostly `1e-4`, plus earlier `5e-5` attempts |
| effective batch | 64 for the local feasibility profile (`train_batch_size: 8`, `gradient_accumulation_steps: 8`) |
| epochs | `1.0` |
| replay | disabled (`0.0`) |
| polish | disabled |
| max sequence length | `1024` |
| assistant-only loss | enabled |

Representative run:

- `runs/posttraining/sft/run_20260423_111010_sft_lr_0p0001`
- final eval at step 93:
  - dev loss: `3.9203`
  - replay PPL: `Infinity`
  - format score: `0.0`
  - composite score: `4.0203`

Interpretation: the model learned some SFT surface behavior, but the high LR / no replay setup likely caused forgetting or unstable behavior.

### Lower LR With Replay

Current default/golden profile in `configs/posttraining/sft.yaml`.

| Field | Value |
|---|---|
| data | current 6,278-example OASST1 + Tulu FLAN mix |
| LR | `1e-5` |
| epochs | `3.0` |
| train batch | `8` |
| gradient accumulation | `4` |
| replay ratio | `0.3` |
| replay source | `data/processed/train.bin` and `data/processed/val.bin` |
| polish | disabled |
| z-loss | `0.0` |
| save every | 10 optimizer steps |

Representative run:

- `runs/posttraining/sft/run_20260425_123144_sft_lr_1e-05`
- summary file says best checkpoint was step 0 with:
  - best dev loss: `2.9082`
  - replay PPL: `25.0064`
  - format score: `0.0`
- final eval at step 233 degraded:
  - dev loss: `4.0197`
  - replay PPL: `107.37`
  - format score: `0.0`

Interpretation: the best-scoring checkpoint was effectively before fine-tuning, while later training degraded replay perplexity heavily. This suggests the model may still be forgetting or overfitting despite low LR and replay.

### Z-Loss Variant

Config: `configs/posttraining/sft_exp_a_zloss.yaml`

Same as the lower-LR replay profile, but:

- `z_loss_coeff: 1e-4`

Representative run:

- `runs/posttraining/sft/run_20260427_221354_sft_lr_1e-05`
- latest visible step: 160
- checkpoint used for inference: `last_epoch_0000_step_0000160.pt`

Interpretation: z-loss was tried to stabilize logits, but the probe output still repeats and does not fix the factual/logical issue.

### Polish Variant

Config: `configs/posttraining/sft_exp_c_polish.yaml`

Same narrow source mix, but:

- `num_epochs: 1.0`
- `polish_epochs: 1.0`
- `polish_subset_size: 500`

This config exists, but no clearly separate polish run beyond smoke-test polish was found in `runs/posttraining/sft/`.

### Full Recipe Preserved for Higher Compute

Config: `configs/posttraining/sft_full_recipe.yaml`

| Field | Value |
|---|---|
| data | full 7-source public mix |
| LRs | `5e-5`, `1e-4`, `2e-4` |
| epochs | `1.0` |
| replay ratio | `0.1` |
| polish epochs | `0.25` |
| gradient accumulation | `8` |

This was preserved as the original quality-maximizing recipe but is slower and less practical on the local Apple Silicon setup.

## Important Technical Fixes During SFT

From the changelog:

- added assistant-only loss masking
- added chunked loss computation to reduce MPS memory pressure
- added MPS autocast support
- fixed EOS masking so EOS is trained as part of assistant output
- fixed inference to stop when EOS is generated
- aligned chat/inference template with SFT template
- fixed a `NameError` in evaluation where `tokenizer` was not defined in generation scope

These fixes mean the current failures are less likely to be pure pipeline bugs and more likely to be model capacity, data mix, objective, or optimization issues.

## Probe Outputs

All outputs below used greedy decoding (`temperature=0`) with `max_tokens=80` and the exact SFT prompt template.

### Base Pretrained Checkpoint

Checkpoint:

`runs/posttraining/base_import/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt`

Output:

```text
What is the capital of Paris?
### User:
What is the capital of Paris?
### Assistant:
What is the capital of Paris?
### User:
What is the capital of Paris?
### Assistant:
What is the capital of Paris?
### User:
What is the capital of Paris?
### Assistant:
What is the capital of Paris?
```

Interpretation: base model does not understand the assistant format and loops over the prompt structure.

### 1e-4 Full-Mix SFT

Checkpoint:

`runs/posttraining/sft/run_20260423_111010_sft_lr_0p0001/checkpoints/last_epoch_0000_step_0000093.pt`

Output:

```text
The capital of Paris is the Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus Juventus
```

Interpretation: SFT changes the model into answer mode, but it collapses into a repeated token.

### 5e-5 SFT

Checkpoint:

`runs/posttraining/sft/run_20260423_212230_sft_lr_5e-05/checkpoints/last_epoch_0000_step_0000150.pt`

Output:

```text
The capital of Paris is the capital of France, which is the capital of France.

The capital of France is the capital of France, which is the capital of France.

The capital of France is the capital of France, which is the capital of France.

The capital of France
```

Interpretation: closer to the relevant concept, but still loops and does not handle the flawed premise.

### 1e-5 Deep-Burn SFT With Replay

Checkpoint:

`runs/posttraining/sft/run_20260425_123144_sft_lr_1e-05/checkpoints/last_epoch_0000_step_0000233.pt`

Output:

```text
The capital of Paris is the capital of the city ofParis.

The capital of the city ofParis is the capital of the city ofParis.

The capital of the city ofParis is the capital of the city ofParis.

The capital of the city ofParis is the capital of
```

Interpretation: lower LR/replay reduces random token collapse, but the model still repeats and cannot resolve the incorrect premise.

### 1e-5 Deep-Burn SFT With Replay + Z-Loss

Checkpoint:

`runs/posttraining/sft/run_20260427_221354_sft_lr_1e-05/checkpoints/last_epoch_0000_step_0000160.pt`

Output:

```text
The capital ofParis is the capital of the city ofParis.

The capital of the city ofParis is the capital of the city ofParis.

The capital of the city ofParis is the capital of the city ofParis.

The capital of the city ofParis is the capital of
```

Interpretation: z-loss did not visibly solve the repetition/factual-reasoning issue on this probe.

## Current Hypotheses

1. The model is too small to reliably learn instruction format, factual knowledge, and premise correction at the same time from the current SFT setup.
2. The model may be overfitting the answer style while losing pretraining knowledge, even with replay.
3. The SFT objective may teach "produce an answer-looking continuation" more than "identify invalid premises."
4. The dataset mix may not contain enough short, direct correction examples such as "X is not a country/city/person; the correct relation is Y."
5. The format score is always `0.0`, so either the format metric is too strict/buggy or the model is still not generating the expected final structure.
6. Best checkpoints often occur at step 0, which suggests the composite selection metric may prefer the base model over actual SFT progress.

## Questions for TA

- Should we continue trying to SFT a ~35M model, or is the observed failure expected at this scale?
- Would a smaller, more targeted dataset of direct QA/correction examples be better than a broad instruction mixture?
- Should we add many synthetic examples specifically teaching invalid-premise correction and concise factual answers?
- Is replay ratio `0.3` enough, or should replay be increased further for such a small model?
- Should we freeze lower layers or use a smaller LR only on embeddings/LM head to reduce forgetting?
- Should checkpoint selection ignore step-0 checkpoints or use a probe-suite metric instead of dev loss?
- Is the always-zero format score a sign of an evaluation bug, or is the model genuinely failing the expected format?
- Would DPO/RLHF be premature until SFT can answer basic probes without repetition?

## Suggested Next Experiments

1. Create a tiny targeted SFT set of 500-2,000 examples focused on:
   - direct factual QA
   - invalid-premise correction
   - concise answers
   - explicit stop behavior
2. Run a very short LR sweep with `1e-6`, `3e-6`, and `1e-5`, using replay ratios `0.3`, `0.5`, and `0.7`.
3. Add probe-based checkpoint selection using a small deterministic prompt suite instead of relying only on dev loss.
4. Inspect/fix the format score metric because it remains `0.0` even after template fixes.
5. Try generation penalties during inference (`repetition_penalty`, no-repeat n-grams) only as diagnosis, not as a training fix.
6. Compare against the base model and a deliberately overfit 100-example SFT run to identify whether the model can learn the target behavior at all.
