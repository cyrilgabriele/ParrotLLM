# Chat SFT+DPO Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a one-shot `scripts/chat_sft_dpo.sh` that trains a chat-optimised ParrotLLM (SFT on conversational data → DPO on HH-RLHF preference pairs) for a class demo.

**Architecture:** Four new files — two training configs, one chat app config, one pipeline script. The script follows the established overnight-script pattern: idempotent steps, Python-based checkpoint injection, `uv run python main.py --stage <stage> --config <config>` invocations. No changes to existing model code, training code, or the Gradio app.

**Tech Stack:** Python / Pydantic (`configs/project_config.py` + `load_project_config`), uv, bash, PyTorch, Gradio (existing `src/chat/app.py`)

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `configs/posttraining/sft_chat.yaml` | Chat-focused SFT recipe (WildChat + OASST1 + TULU, single LR) |
| Create | `configs/posttraining/dpo_chat.yaml` | Chat DPO recipe (HH-RLHF hh_rlhf format, β=0.1) |
| Create | `configs/chat/chat_demo.yaml` | Gradio app config pointing at dpo_chat run dir |
| Create | `scripts/chat_sft_dpo.sh` | Pipeline: sft-prepare → sft → inject checkpoint → dpo-prepare → dpo |

---

## Task 1: Create `configs/posttraining/sft_chat.yaml`

**Files:**
- Create: `configs/posttraining/sft_chat.yaml`

- [ ] **Step 1: Verify the base checkpoint exists**

```bash
ls runs/posttraining/base_import/run_20260410_044337/checkpoints/
```

Expected output: `best_loss_3p2650_epoch_0000_step_0095500.pt`

- [ ] **Step 2: Write the config**

Create `configs/posttraining/sft_chat.yaml` with this exact content:

```yaml
model:
  vocab_size: 50258
  pad_token_id: 50257
  bos_token_id: 50256
  eos_token_id: 50256
  d_model: 384
  n_layers: 14
  n_heads: 6
  d_ff: 768
  context_length: 1024
  bias: false
  dropout: 0.0151
  rope_theta: 10000.0
  gradient_checkpointing: false

logging:
  console_level: INFO
  file_level: DEBUG
  components:
    posttraining: INFO
    training: INFO

chat:
  device: auto
  max_tokens: 256
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  system_prompt: "You are ParrotLLM, a helpful assistant."
  checkpoint_dir: runs

sft:
  device: auto
  base_checkpoint: runs/posttraining/base_import/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt
  cache_dir: data/posttraining/hf_cache
  raw_dir: data/posttraining/raw
  prepared_dir: data/posttraining/sft_chat_mix
  runs_dir: runs/posttraining/sft_chat
  checkpoint_dir: checkpoints
  system_prompt: "You are ParrotLLM, a helpful assistant."
  max_seq_length: 1024
  train_batch_size: 8
  eval_batch_size: 8
  gradient_accumulation_steps: 8
  learning_rates:
    - 5.0e-5
  min_lr_ratio: 0.1
  warmup_ratio: 0.03
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  z_loss_coeff: 0.0
  replay_ratio: 0.1
  replay_train_bin: data/processed/train.bin
  replay_val_bin: data/processed/val.bin
  num_epochs: 1.0
  polish_epochs: 0.25
  polish_subset_size: 4000
  save_every: 250
  eval_every: 100
  keep_last_checkpoints: 3
  keep_best_checkpoints: 2
  log_every: 10
  seed: 42
  compile: false
  format_score_weight: 0.1
  forgetting_penalty_weight: 0.05
  prompt_suite_path: configs/posttraining/dev_prompt_suite.jsonl
  sources:
    - name: wildchat_gpt4
      loader: wildchat
      path: allenai/WildChat
      split: train
      target_examples: 5000
      candidate_multiplier: 4
      language: en
      require_model_substring: gpt-4
      exclude_toxic: true
      exclude_redacted: true
      min_turns: 2
      max_turns: 6
      quality_weight: 1.0
      tags: [chat, multi_turn]
      rationale: Most realistic user prompt distribution for interactive chat behavior.
    - name: oasst1_ready
      loader: oasst1
      path: OpenAssistant/oasst1
      split: train
      target_examples: 4500
      candidate_multiplier: 3
      language: en
      use_best_branch: true
      require_tree_state: ready_for_export
      min_turns: 2
      max_turns: 4
      max_depth: 4
      quality_weight: 1.1
      tags: [chat, human_reviewed]
      rationale: Human-reviewed dialogue trees that teach cleaner assistant behavior.
    - name: tulu_flan_v2
      loader: tulu
      path: allenai/tulu-3-sft-mixture
      split: train
      target_examples: 4000
      candidate_multiplier: 5
      language: en
      source_matches: [flan_v2, flan]
      min_turns: 2
      max_turns: 2
      quality_weight: 1.0
      tags: [short_answer, classification]
      rationale: Protects exact-answer and benchmark-shaped task behavior.
    - name: tulu_persona_if
      loader: tulu
      path: allenai/tulu-3-sft-mixture
      split: train
      target_examples: 2500
      candidate_multiplier: 5
      language: en
      source_matches: [personahub_ifdata_manual_seed_v3_29980]
      min_turns: 2
      max_turns: 2
      quality_weight: 1.0
      tags: [instruction_following]
      rationale: Strong instruction-following and constraint obedience.
    - name: tulu_persona_reasoning
      loader: tulu
      path: allenai/tulu-3-sft-mixture
      split: train
      target_examples: 1500
      candidate_multiplier: 5
      language: en
      source_matches:
        - tulu_v3.9_open_math_2_gsm8k_50k
        - tulu_v3.9_personahub_math_interm_algebra_20k
      min_turns: 2
      max_turns: 2
      quality_weight: 0.95
      tags: [reasoning]
      rationale: Lightweight reasoning without over-indexing on long chain-of-thought.
    - name: tulu_structured_outputs
      loader: tulu
      path: allenai/tulu-3-sft-mixture
      split: train
      target_examples: 1500
      candidate_multiplier: 5
      language: en
      source_matches: [tablegpt, flan]
      min_turns: 2
      max_turns: 2
      quality_weight: 1.0
      tags: [json, extraction]
      rationale: Helps JSON validity and strict format following.
    - name: pku_safe_rlhf_refusals
      loader: pku_safe_rlhf_qa
      path: PKU-Alignment/PKU-SafeRLHF-QA
      split: train
      target_examples: 1000
      candidate_multiplier: 5
      language: en
      keep_harmful_only: true
      min_turns: 2
      max_turns: 2
      quality_weight: 1.0
      tags: [safety, refusal]
      rationale: Public safety QA dataset used to teach concise safe responses to harmful prompts.
  decontam_datasets:
    - name: wikitext103_test
      loader: local_disk
      path: data/wikitext-103-test
      field: text
      split: test
    - name: nlp26_owt_eval
      loader: local_disk
      path: data/owt-eval/NLP26/NLP26_OWT_eval/test
      field: text
      split: test
    - name: hellaswag
      loader: huggingface
      path: Rowan/hellaswag
      field: ctx
      split: validation
    - name: winogrande
      loader: huggingface
      path: allenai/winogrande
      subset: winogrande_xl
      field: sentence
      split: validation
    - name: openbookqa
      loader: huggingface
      path: allenai/openbookqa
      subset: main
      field: question_stem
      split: validation
    - name: lambada
      loader: huggingface
      path: EleutherAI/lambada_openai
      field: text
      split: test
```

- [ ] **Step 3: Verify the config parses**

```bash
uv run python -c "
from configs import load_project_config
cfg = load_project_config('configs/posttraining/sft_chat.yaml')
assert cfg.sft is not None
assert str(cfg.sft.runs_dir) == 'runs/posttraining/sft_chat'
assert cfg.sft.learning_rates == [5e-5]
assert len(cfg.sft.sources) == 7
print('sft_chat.yaml OK — sources:', len(cfg.sft.sources))
"
```

Expected output: `sft_chat.yaml OK — sources: 7`

- [ ] **Step 4: Commit**

```bash
git add configs/posttraining/sft_chat.yaml
git commit -m "config(sft): chat-focused SFT recipe for demo"
```

---

## Task 2: Create `configs/posttraining/dpo_chat.yaml`

**Files:**
- Create: `configs/posttraining/dpo_chat.yaml`

- [ ] **Step 1: Write the config**

Create `configs/posttraining/dpo_chat.yaml` with this exact content.
Note: `reference_checkpoint` contains a placeholder that the pipeline script will replace at runtime before calling DPO.

```yaml
model:
  vocab_size: 50258
  pad_token_id: 50257
  bos_token_id: 50256
  eos_token_id: 50256
  d_model: 384
  n_layers: 14
  n_heads: 6
  d_ff: 768
  context_length: 1024
  bias: false
  dropout: 0.0
  rope_theta: 10000.0
  gradient_checkpointing: false

logging:
  console_level: INFO
  file_level: DEBUG
  components:
    posttraining: INFO
    training: INFO

dpo:
  device: auto
  preference_format: hh_rlhf
  # Placeholder — overwritten by scripts/chat_sft_dpo.sh before dpo-prepare runs.
  reference_checkpoint: runs/posttraining/sft_chat/PLACEHOLDER.pt
  cache_dir: data/posttraining/hf_cache
  raw_dir: data/posttraining/dpo_raw
  prepared_dir: data/posttraining/dpo_chat_pairs
  runs_dir: runs/posttraining/dpo_chat
  system_prompt: "You are ParrotLLM, a helpful assistant."
  max_seq_length: 1024

  beta: 0.1
  learning_rate: 5.0e-7
  num_epochs: 1.0
  train_batch_size: 4
  gradient_accumulation_steps: 1
  warmup_ratio: 0.03
  min_lr_ratio: 0.1
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.999
  grad_clip: 1.0
  seed: 42

  save_every: 200
  eval_every: 100
  log_every: 1
  keep_last_checkpoints: 2
  keep_best_checkpoints: 2

  sources:
    - name: hh_rlhf_helpful_base
      path: Anthropic/hh-rlhf
      subset: helpful-base
      split: train
      target_pairs: 6000
      language: en
    - name: hh_rlhf_helpful_rejection_sampled
      path: Anthropic/hh-rlhf
      subset: helpful-rejection-sampled
      split: train
      target_pairs: 4000
      language: en

  decontam_datasets:
    - name: wikitext103_test
      loader: local_disk
      path: data/wikitext-103-test
      field: text
      split: test
    - name: nlp26_owt_eval
      loader: local_disk
      path: data/owt-eval/NLP26/NLP26_OWT_eval/test
      field: text
      split: test
    - name: hellaswag
      loader: huggingface
      path: Rowan/hellaswag
      field: ctx
      split: validation
    - name: winogrande
      loader: huggingface
      path: allenai/winogrande
      subset: winogrande_xl
      field: sentence
      split: validation
    - name: openbookqa
      loader: huggingface
      path: allenai/openbookqa
      subset: main
      field: question_stem
      split: validation
    - name: lambada
      loader: huggingface
      path: EleutherAI/lambada_openai
      field: text
      split: test
```

- [ ] **Step 2: Verify the config parses**

```bash
uv run python -c "
from configs import load_project_config
cfg = load_project_config('configs/posttraining/dpo_chat.yaml')
assert cfg.dpo is not None
assert cfg.dpo.preference_format == 'hh_rlhf'
assert cfg.dpo.beta == 0.1
assert str(cfg.dpo.runs_dir) == 'runs/posttraining/dpo_chat'
assert len(cfg.dpo.sources) == 2
print('dpo_chat.yaml OK — preference_format:', cfg.dpo.preference_format, 'beta:', cfg.dpo.beta)
"
```

Expected output: `dpo_chat.yaml OK — preference_format: hh_rlhf beta: 0.1`

- [ ] **Step 3: Commit**

```bash
git add configs/posttraining/dpo_chat.yaml
git commit -m "config(dpo): chat DPO recipe (HH-RLHF, beta=0.1) for demo"
```

---

## Task 3: Create `configs/chat/chat_demo.yaml`

**Files:**
- Create: `configs/chat/chat_demo.yaml`

- [ ] **Step 1: Write the config**

Create `configs/chat/chat_demo.yaml`:

```yaml
model:
  vocab_size: 50258
  pad_token_id: 50257
  bos_token_id: 50256
  eos_token_id: 50256
  d_model: 384
  n_layers: 14
  n_heads: 6
  d_ff: 768
  context_length: 1024
  bias: false
  dropout: 0.0
  rope_theta: 10000.0

logging:
  console_level: INFO
  file_level: DEBUG

chat:
  device: auto
  max_tokens: 256
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  system_prompt: "You are ParrotLLM, a helpful assistant."
  checkpoint_dir: runs/posttraining/dpo_chat
```

- [ ] **Step 2: Verify the config parses**

```bash
uv run python -c "
from configs import load_project_config
cfg = load_project_config('configs/chat/chat_demo.yaml')
assert cfg.chat is not None
assert str(cfg.chat.checkpoint_dir) == 'runs/posttraining/dpo_chat'
assert cfg.chat.max_tokens == 256
print('chat_demo.yaml OK — checkpoint_dir:', cfg.chat.checkpoint_dir)
"
```

Expected output: `chat_demo.yaml OK — checkpoint_dir: runs/posttraining/dpo_chat`

- [ ] **Step 3: Commit**

```bash
git add configs/chat/chat_demo.yaml
git commit -m "config(chat): demo chat config pointing at dpo_chat runs"
```

---

## Task 4: Create `scripts/chat_sft_dpo.sh`

**Files:**
- Create: `scripts/chat_sft_dpo.sh`

- [ ] **Step 1: Write the script**

Create `scripts/chat_sft_dpo.sh`:

```bash
#!/usr/bin/env bash
# One-shot chat training pipeline: SFT (chat data) → DPO (HH-RLHF).
# Idempotent: each step is skipped if its output already exists.
# Usage: nohup bash scripts/chat_sft_dpo.sh > runs/chat_sft_dpo.log 2>&1 &

set -euo pipefail

ROOT="/Users/gian1/CODE/HSG/FS26/NLP/ParrotLLM"
cd "$ROOT"

CONFIG_SFT="configs/posttraining/sft_chat.yaml"
CONFIG_DPO="configs/posttraining/dpo_chat.yaml"
LOG_DIR="runs/chat_sft_dpo_logs"
mkdir -p "$LOG_DIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/run.log"; }

log "=== Chat SFT+DPO pipeline started ==="

# ---- Step 1: SFT prepare ----
log "Step 1/5: SFT prepare"
if ls data/posttraining/sft_chat_mix/*.jsonl > /dev/null 2>&1; then
  log "  SFT mix already exists, skipping"
else
  uv run python main.py --stage sft-prepare --config "$CONFIG_SFT" \
    >> "$LOG_DIR/01_sft_prepare.log" 2>&1
fi
log "  SFT prepare done"

# ---- Step 2: SFT train ----
log "Step 2/5: SFT train"
SFT_CKPT_PROBE=$(find runs/posttraining/sft_chat -name "best_loss_*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$SFT_CKPT_PROBE" ]]; then
  log "  SFT checkpoint already exists: $SFT_CKPT_PROBE"
else
  uv run python main.py --stage sft --config "$CONFIG_SFT" \
    >> "$LOG_DIR/02_sft_train.log" 2>&1
fi

SFT_CKPT=$(find runs/posttraining/sft_chat -name "best_loss_*.pt" -type f 2>/dev/null \
  | sort | head -1)
if [[ -z "$SFT_CKPT" ]]; then
  log "FATAL: SFT produced no best_loss_*.pt checkpoint"
  exit 1
fi
log "  SFT best checkpoint: $SFT_CKPT"

# ---- Step 3: Inject SFT checkpoint into DPO config ----
log "Step 3/5: Wire SFT checkpoint into DPO config"
uv run python - <<PY
import re
from pathlib import Path
cfg_path = Path("$CONFIG_DPO")
src = cfg_path.read_text()
src = re.sub(
    r"^  reference_checkpoint:.*$",
    f"  reference_checkpoint: $SFT_CKPT",
    src,
    flags=re.M,
)
cfg_path.write_text(src)
print("DPO config updated:", "$SFT_CKPT")
PY
log "  DPO config wired"

# ---- Step 4: DPO prepare ----
log "Step 4/5: DPO prepare (HH-RLHF pairs)"
if [[ -f "data/posttraining/dpo_chat_pairs/train.jsonl" ]]; then
  log "  DPO pairs already exist, skipping"
else
  uv run python main.py --stage dpo-prepare --config "$CONFIG_DPO" \
    >> "$LOG_DIR/04_dpo_prepare.log" 2>&1
fi
log "  DPO prepare done"

# ---- Step 5: DPO train ----
log "Step 5/5: DPO train"
DPO_CKPT_PROBE=$(find runs/posttraining/dpo_chat -name "*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -n "$DPO_CKPT_PROBE" ]]; then
  log "  DPO checkpoint already exists: $DPO_CKPT_PROBE"
else
  uv run python main.py --stage dpo --config "$CONFIG_DPO" \
    >> "$LOG_DIR/05_dpo_train.log" 2>&1
fi

DPO_CKPT=$(find runs/posttraining/dpo_chat -name "best_loss_*.pt" -type f 2>/dev/null | head -1 || true)
if [[ -z "$DPO_CKPT" ]]; then
  DPO_CKPT=$(find runs/posttraining/dpo_chat -name "*.pt" -type f 2>/dev/null | head -1 || true)
fi
if [[ -z "$DPO_CKPT" ]]; then
  log "FATAL: DPO produced no checkpoint"
  exit 1
fi

log "=== Pipeline complete ==="
log "Final chat checkpoint: $DPO_CKPT"
log "Launch demo: uv run python main.py --stage chat --config configs/chat/chat_demo.yaml"
```

- [ ] **Step 2: Make executable and validate syntax**

```bash
chmod +x scripts/chat_sft_dpo.sh
bash -n scripts/chat_sft_dpo.sh
```

Expected: no output (bash -n only reports syntax errors)

- [ ] **Step 3: Commit**

```bash
git add scripts/chat_sft_dpo.sh
git commit -m "feat(scripts): one-shot chat SFT+DPO pipeline for demo"
```

---

## Task 5: Run the Pipeline

- [ ] **Step 1: Start the pipeline in the background**

```bash
nohup bash scripts/chat_sft_dpo.sh > runs/chat_sft_dpo.log 2>&1 &
echo $! > runs/chat_sft_dpo.pid
echo "Pipeline started, PID: $(cat runs/chat_sft_dpo.pid)"
```

- [ ] **Step 2: Tail the log to confirm it starts cleanly**

```bash
tail -f runs/chat_sft_dpo.log
```

Watch for the first few log lines:
```
[...] === Chat SFT+DPO pipeline started ===
[...] Step 1/5: SFT prepare
```
Hit Ctrl-C to stop tailing once you see it progressing. The process continues in the background.

- [ ] **Step 3: (After pipeline finishes) Verify DPO checkpoint exists**

```bash
find runs/posttraining/dpo_chat -name "best_loss_*.pt" | sort
```

Expected: at least one `.pt` file, e.g. `runs/posttraining/dpo_chat/run_YYYYMMDD_.../checkpoints/best_loss_0pXXXX_epoch_0000_step_NNNNN.pt`

- [ ] **Step 4: (After pipeline finishes) Launch and smoke-test the chat demo**

```bash
uv run python main.py --stage chat --config configs/chat/chat_demo.yaml
```

In the Gradio UI (opens at `http://127.0.0.1:7860`):
1. Select the `best_loss_*.pt` checkpoint from the dropdown and click Load
2. Type "What is a language model?" and verify the response is a coherent assistant-style answer (not a benchmark letter like "A)")
3. Type a follow-up question and verify multi-turn context is maintained

- [ ] **Step 5: Commit final state**

```bash
git add configs/posttraining/dpo_chat.yaml  # now has real reference_checkpoint injected
git commit -m "chore: update dpo_chat.yaml with resolved SFT checkpoint after pipeline run"
```
