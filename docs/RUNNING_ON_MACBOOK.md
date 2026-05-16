# Running ParrotLLM on a MacBook

How a teammate (or anyone else) checks out this branch on a Mac and runs
the trained 40M-parameter model. Tested on Apple-silicon (M1/M2/M3); also
works on Intel Macs in CPU mode (slower).

---

## 1. Prerequisites

- **macOS** (Apple-silicon or Intel)
- **Python 3.14+** — install via [pyenv](https://github.com/pyenv/pyenv) or `brew install python@3.14`
- **`uv`** (the dependency manager this repo uses):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **`git`** — comes with Xcode Command Line Tools (`xcode-select --install`)
- ~5 GB free disk for the venv + model + benchmarks

You do **not** need CUDA, Docker, or conda.

---

## 2. Clone and check out the branch

```bash
git clone https://github.com/cyrilgabriele/ParrotLLM.git
cd ParrotLLM
git checkout sft-christof
```

---

## 3. Install dependencies

```bash
uv sync
```

`uv` reads `pyproject.toml` + `uv.lock` and sets up a fully-pinned `.venv/`.
On Apple-silicon it installs the macOS PyTorch wheel with **MPS** (Metal
Performance Shaders) GPU acceleration automatically — no extra config.

Verify:

```bash
uv run python -c "
import torch
print('torch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
"
```

Expected on M1/M2/M3: `MPS available: True`.

---

## 4. Get the trained checkpoint

The trained `.pt` files are **not in git** (they're gitignored — too large).
You have two options:

### Option A — pull from the team's Hugging Face dataset repo

The team mirrors finished training runs to `ParrotLabs/Preprocessed`. The
winning checkpoint is `run_20260428_211931_sft/`. From the repo root:

```bash
mkdir -p runs/run_20260428_211931_sft/checkpoints
uv run python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('runs/run_20260428_211931_sft/checkpoints', exist_ok=True)
path = hf_hub_download(
    repo_id='ParrotLabs/Preprocessed',
    repo_type='dataset',
    filename='run_20260428_211931_sft/checkpoints/final_step_0001966_epoch_01_valloss_2p4231.pt',
    local_dir='runs',
)
print('downloaded to:', path)
"
```

Set `HF_TOKEN=...` (or put it in `.env`) if the repo is private.

### Option B — copy from a teammate's machine

```bash
# rsync from a machine that has the checkpoint
rsync -avh teammate@host:/path/to/ParrotLLM/runs/run_20260428_211931_sft/checkpoints/ \
  runs/run_20260428_211931_sft/checkpoints/
```

The single file you need: `final_step_0001966_epoch_01_valloss_2p4231.pt` (~160 MB).

---

## 5. Try it out — three options

### a) One-shot generation (sanity check)

```bash
uv run python main.py --stage inference \
  --checkpoint runs/run_20260428_211931_sft/checkpoints/final_step_0001966_epoch_01_valloss_2p4231.pt \
  --prompt "The capital of France is" \
  --max-tokens 30 --temperature 0.0
```

You should see a short greedy continuation like ` Paris.` (the model is
small — don't expect long coherent prose).

### b) Interactive chat

```bash
uv run python main.py --stage chat --config configs/default.yaml
```

Pick the V7 SFT checkpoint from the dropdown, chat in the Gradio web UI
that opens in your browser. Apple-silicon: chat runs on MPS, ~5 tok/sec.

### c) Run the public benchmarks

This reproduces the headline 33.6% public_avg locally.

First, fetch the four benchmark validation files from the leaderboard repo:

```bash
mkdir -p data/leaderboard_benchmarks
for B in hellaswag winogrande openbookqa lambada; do
  if [ "$B" = "lambada" ]; then SPLIT=test; else SPLIT=validation; fi
  curl -fsSL \
    "https://raw.githubusercontent.com/unisg-ics-dsnlp/PikoGPT_Leaderboard/main/leaderboard/benchmarks/$B/cleaned/$SPLIT.jsonl" \
    -o "data/leaderboard_benchmarks/${B}.jsonl"
done
```

Run the harness (single-process, ~5 minutes on M2 Pro):

```bash
PYTHONPATH=. uv run python tools/run_public_benchmarks.py \
  --checkpoint runs/run_20260428_211931_sft/checkpoints/final_step_0001966_epoch_01_valloss_2p4231.pt \
  --device auto \
  --limit 500 \
  --pmi
```

Expected output:

```
=== summary ===
  hellaswag     32.2%  (n=500, invalid=0)
  winogrande    54.2%  (n=500, invalid=0)
  openbookqa    25.0%  (n=500, invalid=0)
  lambada       22.2%  (n=500, invalid=0)
  public_avg    33.4%
```

(The official runner also reports ~33.6% — small variance on LAMBADA
between harness and runner.)

---

## 6. MacBook caveats

- **MPS backend, not CUDA.** All the production code paths handle this
  automatically via `src/utils.py::get_device`. If something breaks, pass
  `--device cpu` to fall back to CPU mode (slower but reliable).
- **`torch.compile` is disabled on MPS** automatically — it's a Linux/CUDA
  optimization. You'll see ~2× slower inference on Mac than on a 4090,
  which is fine for evaluation.
- **8B-token pretrain dataset (`data/processed/filter_c/`)** is not needed
  for inference or chat — only for further SFT training. The full
  preprocessed corpus is ~8 GB.
- **Training a new SFT run on a MacBook** is technically possible but
  slow (a single SFT epoch is ~6 hours on M2 Pro). Use a CUDA box for
  training; the MacBook for inference / chat / benchmarks only.

---

## 7. What's where in the repo

```
ParrotLLM/
├── main.py                                 # all stages (inference, chat, eval, sft, dpo, train, ...)
├── configs/
│   ├── default.yaml                        # base inference / chat config
│   └── post_training/sft_v7_8b.yaml        # winning SFT recipe
├── src/
│   ├── eval/
│   │   ├── inference.py                    # cloze MC scoring + LAMBADA rstrip + PMI calibration
│   │   └── perplexity.py                   # PPL on Wikitext-103 / OWT
│   ├── chat/app.py                         # Gradio chat UI
│   ├── post_training/sft/                  # SFT trainer
│   ├── post_training/dpo/                  # DPO trainer
│   └── model/transformer.py                # the 40M ParrotLLM
├── tools/
│   ├── run_public_benchmarks.py            # fast leaderboard harness
│   ├── soup_checkpoints.py                 # weight averaging
│   ├── build_auto_cloze.py                 # synthetic data generator
│   └── overnight_pipeline_v8.sh            # full night-time training+eval flow
├── docs/
│   ├── RUNNING_ON_MACBOOK.md               # this file
│   ├── post_training/v7_v8_results.md      # results & insights
│   └── post_training/SFT.md                # SFT design notes
├── runs/                                    # gitignored; model checkpoints land here
└── OVERNIGHT_REPORT.md                      # detailed log of the night that produced the v7 ckpt
```

If anything's missing or fails, see `docs/post_training/v7_v8_results.md`
for the full context, then ping Christof on the team channel.
