# ParrotLLM

35.8M parameter decoder-only language model built from scratch for the PikoGPT Challenge (NLP Lab FS26).

## Quick Start

```bash
# 1. Clone and install (requires Python 3.14+ and uv)
git clone <repo-url> && cd ParrotLLM
uv sync

# 2. Download datasets
uv run python src/scripts/download_data.py

# 3. Preprocess (tokenize + filter + decontaminate)
uv run python main.py --stage preprocess --dataset-size small   # 10k docs, fast
uv run python main.py --stage preprocess --dataset-size full    # full OpenWebText

# 4. Train
uv run python main.py --stage train

# 5. Evaluate
uv run python main.py --stage eval --checkpoint checkpoints/step_5000.pt

# 6. Generate text
uv run python main.py --stage inference --checkpoint checkpoints/step_5000.pt \
    --prompt "The meaning of life is"

# 7. Chat UI
uv run python main.py --stage chat
```

## Setup for New Team Members

### Prerequisites

- **Python 3.14+**
- **uv** (dependency manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Step-by-Step

```bash
# Clone
git clone <repo-url>
cd ParrotLLM

# Install all dependencies (torch, transformers, datasets, wandb, gradio, etc.)
uv sync

# Verify everything works
uv run python -c "
from src.utils import load_config
from src.model import ParrotLLM
import torch
config = load_config('configs/default.yaml')
model = ParrotLLM(config)
print(f'Parameters: {model.count_parameters():,}')
x = torch.randint(0, 50257, (2, 128))
logits, loss = model(x, targets=x)
print(f'Logits: {logits.shape}, Loss: {loss.item():.2f}')
"
# Expected: Parameters: 35,763,840 | Logits: torch.Size([2, 128, 50257]) | Loss: ~9-11

# Download datasets (10k subset, wikitext-103 test, fasttext model, NLP26 eval)
uv run python src/scripts/download_data.py
```

That's it. No conda, no pip, no Docker. `uv sync` handles everything.

## Project Structure

```
ParrotLLM/
├── main.py                          # CLI entry point (all stages)
├── configs/
│   └── default.yaml                 # single source of truth for all hyperparams
├── src/
│   ├── utils.py                     # load_config, set_seed, get_device
│   ├── model/
│   │   └── transformer.py           # RMSNorm, RoPE, MHA, SwiGLU, ParrotLLM
│   ├── training/
│   │   └── trainer.py               # training loop, dataset, checkpointing
│   ├── eval/
│   │   ├── perplexity.py            # perplexity on Wikitext-103 / OWT val
│   │   └── inference.py             # text generation, leaderboard mode
│   ├── chat/
│   │   └── app.py                   # Gradio chat interface
│   ├── data/
│   │   └── preprocess.py            # tokenize, filter, decontaminate, save .bin
│   ├── scripts/
│   │   └── download_data.py         # download all datasets
│   └── notebooks/
│       └── 01_data_preprocessing.ipynb
├── data/                            # datasets (git-tracked metadata, not binaries)
│   ├── openwebtext-10k/             # 10k doc subset for dev
│   ├── wikitext-103-test/           # eval benchmark
│   ├── owt-eval/                    # NLP26 decontamination set
│   ├── lid.176.ftz                  # fasttext lang detection model
│   └── processed/                   # generated .bin files (gitignored)
├── checkpoints/                     # model checkpoints (gitignored)
├── docs/
│   └── ARCHITECTURE_DECISIONS.md    # paper-backed design rationale
├── pyproject.toml                   # dependencies & project metadata
└── uv.lock                          # locked dependency versions
```

## Architecture

```
Tokenizer:      GPT-2 (vocab=50,257)              [course constraint]
d_model:        320
Layers:         16
Heads:          8  (head_dim=40)
FFN:            SwiGLU, d_ff=854 (8/3 * 320)
Normalization:  RMSNorm, Pre-Norm
Positional:     RoPE (Rotary Position Embedding)
Weight Tying:   Yes (input embed = output head)
Context:        1024                               [course constraint]
Total params:   35,763,840  (~35.8M, budget: 40M)
```

### Why These Choices

| Decision | Choice | Reason | Paper |
|----------|--------|--------|-------|
| Deep & narrow | 16 layers, d=320 | Depth > width at small scale | MobileLLM (Meta, 2024) |
| SwiGLU | d_ff = 8/3 * d_model | Beats GELU at same param cost | Shazeer (2020) |
| RMSNorm + Pre-Norm | Before each sublayer | Same quality, faster, stable training | Zhang & Sennrich (2019) |
| RoPE | Rotary embeddings | Zero extra params, relative position | Su et al. (2021) |
| Weight tying | Shared embed/head | Saves 16M params, improves PPL | Press & Wolf (2017) |

Full analysis with paper references: [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md)

## Config-Driven Pipeline

Every stage reads from `configs/default.yaml`. Change hyperparameters there, not in code.

```yaml
# Key sections:
model:      # vocab, d_model, n_layers, n_heads, d_ff, context_length
training:   # batch_size, lr, warmup, max_steps, grad_accum, checkpointing
eval:       # batch_size, datasets, max_sequences
inference:  # temperature, top_k, top_p, max_tokens
chat:       # temperature, max_tokens, checkpoint_dir
```

## CLI Reference

All stages go through `main.py`:

```bash
# Preprocess
uv run python main.py --stage preprocess [--dataset-size small|full] [--lang en]

# Train
uv run python main.py --stage train [--config configs/default.yaml] [--checkpoint path.pt]

# Evaluate perplexity
uv run python main.py --stage eval --checkpoint checkpoints/step_5000.pt

# Generate text
uv run python main.py --stage inference --checkpoint checkpoints/step_5000.pt \
    [--prompt "text"] [--max-tokens 128] [--temperature 0.0]

# Leaderboard mode (stdout only, no logs)
uv run python main.py --stage inference --checkpoint checkpoints/step_5000.pt \
    --prompt "The answer is" --leaderboard

# Chat UI
uv run python main.py --stage chat
```

## Training Details

- **Optimizer**: AdamW (betas 0.9/0.95, weight decay 0.1, decay on 2D params only)
- **Schedule**: linear warmup (2000 steps) + cosine decay to 10% of peak LR
- **Mixed precision**: bfloat16 on Ampere+, float16+GradScaler on V100, float32 on CPU/MPS
- **torch.compile**: auto-enabled on CUDA for ~20-40% speedup
- **Gradient accumulation**: 4 micro-steps (effective batch = 256)
- **Gradient clipping**: max norm 1.0
- **Checkpoints**: every 5000 steps, eval every 500 steps
- **Logging**: wandb (auto-detected, falls back to console)

## What's Done (Week 1)

- [x] Data pipeline: download, language filter, decontamination, tokenization, binary serialization
- [x] Model: full LLaMA-style transformer (RMSNorm, RoPE, SwiGLU, weight tying)
- [x] Training loop: mixed precision, gradient accumulation, cosine LR, checkpointing, wandb
- [x] Evaluation: perplexity on Wikitext-103 and OWT val split
- [x] Inference: greedy + top-k/top-p sampling, leaderboard contract
- [x] Chat: Gradio web UI with checkpoint selection
- [x] Config: single YAML file drives every stage

## Datasources

| Dataset | Purpose | Size |
|---------|---------|------|
| [OpenWebText (full)](https://huggingface.co/datasets/Skylion007/openwebtext) | Training data | ~8M docs, ~8.9B tokens |
| [OpenWebText 10k](https://huggingface.co/datasets/stas/openwebtext-10k) | Fast dev iteration | 10k docs, ~11M tokens |
| [Wikitext-103 test](https://huggingface.co/datasets/Salesforce/wikitext) | Perplexity benchmark | Standard eval set |
| [NLP26 OWT eval](https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K) | Decontamination | Course eval split |
