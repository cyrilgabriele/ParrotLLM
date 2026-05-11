# Chat SFT + DPO — Design Spec

**Date:** 2026-05-11  
**Branch:** sft-dpo-gian  
**Goal:** A single one-shot pipeline that produces a chat-optimised checkpoint for the class demo.

---

## Problem

The current best submission checkpoint (`dpo_continuation_beta001`, sum=157.0) was trained on benchmark continuation pairs (MC letter completion). In a live chat demo, it is likely to produce benchmark-style outputs rather than natural conversational responses. The class audience will notice.

## Solution

Train a separate chat model from scratch (starting from the pretrained base) using:
1. **SFT** on conversational data (WildChat + OASST1 + TULU instruction-following)
2. **DPO** on human-preference pairs (Anthropic HH-RLHF)

This model is never submitted to the benchmark — it exists solely for the demo.

---

## Architecture

Unchanged from the rest of the project:

| Field | Value |
|---|---|
| d_model | 384 |
| n_layers | 14 |
| n_heads | 6 |
| d_ff | 768 |
| context_length | 1024 |
| vocab_size | 50258 |

---

## Config Files

### `configs/posttraining/sft_chat.yaml`

Cloned from `configs/posttraining/sft_chat_demo.yaml` with the following changes:

- `base_checkpoint`: `runs/posttraining/base_import/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt` (the only available base)
- `runs_dir`: `runs/posttraining/sft_chat`
- `learning_rates`: single value `[5.0e-5]` — no sweep; skip the LR search to save time
- `num_epochs`: 1.0, `polish_epochs`: 0.25 (same as sft_chat_demo)
- `prepared_dir`: `data/posttraining/sft_chat_mix` (isolated from benchmark SFT data)

Data sources (total ~19k examples):

| Source | Examples | Purpose |
|---|---|---|
| WildChat GPT-4 | 5000 | Realistic user prompt distribution |
| OASST1 human-reviewed | 4500 | Clean assistant behaviour |
| TULU persona IF | 2500 | Instruction following |
| TULU FLAN v2 | 4000 | Short-answer / classification |
| TULU persona reasoning | 1500 | Lightweight reasoning |
| TULU structured outputs | 1500 | JSON / format following |
| PKU safe RLHF refusals | 1000 | Refusal on harmful prompts |

### `configs/posttraining/dpo_chat.yaml`

Based on archived `configs/posttraining/_archive/dpo.yaml` with:

- `preference_format`: `hh_rlhf`
- `reference_checkpoint`: resolved at runtime by the pipeline script (best SFT checkpoint)
- `runs_dir`: `runs/posttraining/dpo_chat`
- `prepared_dir`: `data/posttraining/dpo_chat_pairs`
- `beta`: 0.1 (original well-tested value; no benchmark pressure to push lower)
- `learning_rate`: 5.0e-7
- `num_epochs`: 1.0

DPO sources:

| Source | Pairs | Subset |
|---|---|---|
| Anthropic/hh-rlhf | 6000 | helpful-base |
| Anthropic/hh-rlhf | 4000 | helpful-rejection-sampled |

### `configs/chat/chat_demo.yaml`

New chat config pointing `checkpoint_dir` at `runs/posttraining/dpo_chat` so the Gradio app picks up the result automatically.

---

## Pipeline Script

`scripts/chat_sft_dpo.sh` — idempotent, one-shot:

```
sft-prepare (sft_chat.yaml)
  → sft train (sft_chat.yaml)
    → resolve best SFT checkpoint (lowest loss best_loss_*.pt)
      → dpo-prepare (dpo_chat.yaml, reference_checkpoint injected)
        → dpo train (dpo_chat.yaml)
```

Each step is skipped if its output already exists (same pattern as `scripts/overnight_dpo_compare.sh`).

**Run command:**
```bash
nohup bash scripts/chat_sft_dpo.sh > runs/chat_sft_dpo.log 2>&1 &
```

**Estimated wall-clock:** 4–5h on Apple Silicon (MPS).

---

## Chat App Integration

The existing `src/chat/app.py` needs no changes. The Gradio checkpoint browser scans `chat_cfg.checkpoint_dir` recursively for `.pt` files. Pointing `chat_demo.yaml`'s `checkpoint_dir` at `runs/posttraining/dpo_chat` will surface the final checkpoint automatically.

**Launch command:**
```bash
uv run python main.py chat --config configs/chat/chat_demo.yaml
```

---

## What This Is Not

- Not a benchmark submission — the chat model is demo-only.
- Not replacing `dpo_continuation_beta001` — that stays as the submission checkpoint.
- No external model distillation involved (HH-RLHF is a human preference dataset, not model outputs).

---

## Success Criteria

- Pipeline completes without error
- DPO checkpoint exists at `runs/posttraining/dpo_chat/`
- Chat app loads the checkpoint and produces coherent multi-turn responses to open-ended questions
