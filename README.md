# ParrotLLM

ParrotLLM is a course-scale decoder-only language model built from scratch for
the PikoGPT Challenge in NLP Lab FS26. The final pretrained model uses a tuned
39.97M-parameter GPT-style architecture and was trained under a strict 40M
parameter budget.

The full project write-up is in
[ParrotLLM/techreport.tex](ParrotLLM/techreport.tex). It documents the data
pipeline, architecture search, four pretraining runs, inference contract, and
post-training branch results.

## Quick Start

```bash
# 1. Clone and install (requires Python 3.14+ and uv)
git clone <repo-url>
cd ParrotLLM
uv sync

# 2. Download datasets
uv run python src/scripts/download_data.py

# 3. Preprocess
uv run python main.py --stage preprocess --config configs/default.yaml

# 4. Train
uv run python main.py --stage train --config configs/default.yaml

# 5. Evaluate
uv run python main.py --stage eval --config configs/default.yaml \
    --checkpoint checkpoints/step_5000.pt

# 6. Generate text
uv run python main.py --stage inference --config configs/default.yaml \
    --checkpoint checkpoints/step_5000.pt \
    --prompt "The meaning of life is"

# 7. Mock inference (downloads Hugging Face GPT-2; no ParrotLLM checkpoint)
uv run python main.py --stage inference --mock-testing \
    --prompt "The meaning of life is"

# 8. Chat UI
uv run python main.py --stage chat --config configs/default.yaml

# 9. Two-team demo (Cyril & Christof vs. Gian & Tilman)
bash tools/download_demo_checkpoints.sh
uv run python main.py --stage chat --config configs/chat/chat_demo.yaml

# 10. Training dashboard
uv run python main.py --stage dashboard --config configs/default.yaml
```

## Two-team demo

The original team split into two halves for the final submission — both
trained their own ~40M-param model on top of the shared pretraining
base. The chat UI can load either checkpoint and switch between them
live from the sidebar so the two outputs can be compared side-by-side
on the same prompt.

| Label              | Submission             | Source                                                                                  |
|--------------------|------------------------|-----------------------------------------------------------------------------------------|
| Cyril & Christof   | `ParrotLLM_llarotpm`   | PR #4 — release asset on `steinerchristof/PikoGPT_Leaderboard` (tag `parrotllm-may05`)  |
| Gian & Tilman      | `PikoGPT_ParrotLabs`   | PR #13 — committed blob on `TilmanHaferbeck/PikoGPT_Leaderboard@parrotllm_submission`   |

On a fresh machine:

```bash
git clone <repo-url> && cd ParrotLLM
uv sync                                     # install deps
bash tools/download_demo_checkpoints.sh     # ~120 MB total into runs/demo/
uv run python main.py --stage chat --config configs/chat/chat_demo.yaml
```

The sidebar exposes the two checkpoints as labeled radio buttons
("Cyril & Christof" loads on startup; click "Gian & Tilman" to swap).
The named labels and paths are config-driven via `chat.demo_checkpoints`
in `configs/chat/chat_demo.yaml`, so the demo can be re-targeted
without touching code.

## Setup

### Prerequisites

- Python 3.14+
- `uv` dependency manager

Install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify The Install

```bash
uv sync

uv run python -c "
from configs import load_project_config
from src.model import ParrotLLM
import torch

project = load_project_config('configs/default.yaml')
config = project.model_dump(mode='python')
model = ParrotLLM(config)
print(f'Parameters: {model.count_parameters():,}')
x = torch.randint(0, config['model']['vocab_size'], (2, 128))
logits, loss = model(x, targets=x)
print(f'Logits: {logits.shape}, Loss: {loss.item():.2f}')
"
```

Expected parameter count for the current default model:
`39,966,592`.

## Project Structure

```text
ParrotLLM/
├── main.py                         # CLI entry point for all stages
├── configs/
│   ├── default.yaml                # main config
│   ├── big_run/                    # final large pretraining configs
│   ├── preprocessing/              # data variant configs
│   ├── training/                   # smoke/legacy train configs
│   ├── tuning/                     # Optuna architecture/HP tuning configs
│   ├── eval/                       # eval smoke configs
│   ├── inference/                  # inference smoke configs
│   └── chat/                       # chat smoke configs
├── src/
│   ├── data/                       # preprocessing pipeline
│   ├── model/                      # transformer implementation
│   ├── training/                   # training loop and Optuna tuning
│   ├── eval/                       # perplexity and inference
│   ├── chat/                       # Gradio chat UI
│   ├── dashboard/                  # Gradio/TUI training dashboard
│   ├── scripts/                    # data/model utility scripts
│   └── notebooks/                  # exploratory notebooks
├── tests/                          # unit and integration tests
├── docs/
│   ├── architecture/               # architecture notes and diagrams
│   ├── gpu_cluster/                # cluster training notes
│   └── poster/                     # poster assets/results
├── ParrotLLM/
│   ├── techreport.tex              # technical report
│   └── figures/                    # report figures
├── results/                        # tuning/evaluation outputs
├── runs/                           # local run outputs and checkpoints
├── pyproject.toml                  # project metadata and dependencies
└── uv.lock                         # locked dependency versions
```

Large binaries, processed data, and checkpoints are local artifacts and are not
expected to be fully tracked in git.

## Architecture

The current default architecture is the final tuned shape reported in the
technical report and encoded in [configs/default.yaml](configs/default.yaml).

```text
Tokenizer:      GPT-2 tokenizer + dedicated pad token
Vocabulary:     50,258
d_model:        384
Layers:         14
Heads:          6 (head_dim=64)
FFN:            SwiGLU, d_ff=768
Normalization:  RMSNorm, QK-Norm, Peri-LN-style dual normalization
Positional:     RoPE
Weight tying:   yes
Context:        1024
Biases:         disabled in linear layers
Total params:   39,966,592
```

Architecture selection moved beyond the initial MobileLLM-style draft. The
final shape came from staged Optuna searches at roughly 8.75M, 17.5M, and 40M
parameters, using proxy training to select a wider, shallower model than the
initial deep-and-narrow design.

See [docs/architecture/ARCHITECTURE_DECISIONS.md](docs/architecture/ARCHITECTURE_DECISIONS.md)
and [ParrotLLM/techreport.tex](ParrotLLM/techreport.tex) for the full rationale.

## Data Pipeline

Preprocessing runs through `main.py --stage preprocess` and is controlled by
YAML config. The final pipeline includes:

- raw-text decontamination against evaluation sets
- sanitization of HTML, boilerplate, control characters, and URLs
- English language filtering with fastText
- optional AG-News topic filtering/resampling
- code/artifact filtering
- heuristic quality filtering
- MinHash-style near-deduplication
- ellipsis-density filtering
- GPT-2 tokenization with a dedicated pad token
- binary train/validation output

The report compares six controlled data variants. Experiment C, a balanced
World/Business/Sci-Tech mixture with Sports removed, was selected as the final
pretraining data recipe.

## Training And Tuning

All main runtime settings live in YAML configs rather than being hard-coded.
The main CLI stages are:

```bash
uv run python main.py --stage preprocess --config configs/default.yaml
uv run python main.py --stage tune --config configs/tuning/tune.yaml
uv run python main.py --stage train --config configs/default.yaml
uv run python main.py --stage train --config configs/default.yaml \
    --resume-training --checkpoint path/to/checkpoint.pt
uv run python main.py --stage eval --config configs/default.yaml \
    --checkpoint path/to/checkpoint.pt
uv run python main.py --stage inference --config configs/default.yaml \
    --checkpoint path/to/checkpoint.pt --prompt "Prompt text"
uv run python main.py --stage chat --config configs/default.yaml
uv run python main.py --stage dashboard --config configs/default.yaml
```

Training uses AdamW, mixed precision where available, gradient clipping,
checkpoint retention, periodic validation, optional `torch.compile`, optional
Hugging Face run uploads, and console/file logging. The dashboard can be run as
a Gradio app or as a terminal UI with `--tui`.

## Pretraining Results

Final pretraining used the chair-provided 8xV100 hardware. The team ran four
pretraining attempts:

| Run | Dataset | Size | Outcome |
| --- | --- | --- | --- |
| 1 | Experiment A | 800M-token setup | short baseline run |
| 2 | Experiment C | 800M-token setup | short comparison run |
| 3 | Experiment C | 8B-token setup | early-stopped with the initial tuned HPs |
| 4 | Experiment C | 8B-token setup | final time-limited run |

The final pretrained checkpoint reported in the technical report is:

```text
runs/big_run/exp_c_8b/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt
```

Reported pre-post-training perplexities:

| Dataset | Perplexity |
| --- | ---: |
| Wikitext-103 | 51.8 |
| OpenWebText validation | 24.17 |

## Post-Training Branches

Post-training is not represented as a single linear history on the current
branch. The technical report tracks three post-training branches:

| Branch | Owners / role | Summary |
| --- | --- | --- |
| `sft-tilman` | Tilman+Gian SFT branch | Initial benchmark-focused SFT setup and first successful SFT pass from the pretrained base. |
| `sft-dpo-gian` | Tilman+Gian DPO branch | Continuation-pair DPO on top of benchmark-targeted SFT; strongest documented public result. |
| `sft-christof` | Christof+Cyril branch | Course-style SFT variants, DPO implementation, and inference repairs; final selected result was SFT v7 plus PMI/cloze inference. |

Documented public benchmark ledger from the report:

| Model/source | Limit | HellaSwag | WinoGrande | OpenBookQA | LAMBADA | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained base, local sweep | 500 | 32.6 | 50.0 | 24.0 | 11.2 | 29.45 |
| Tilman+Gian DPO, manifest | 200 | 23.5 | 59.5 | 35.5 | 38.5 | 39.25 |
| Christof+Cyril SFT v7 + PMI | 500 | 32.2 | 54.0 | 25.0 | 23.2 | 33.60 |
| Local SFT, May 5 | 500 | 31.6 | 52.2 | 24.8 | 14.0 | 30.65 |
| Local DPO, May 6 | 500 | 31.2 | 52.2 | 24.4 | 13.0 | 30.20 |

The limits and inference contracts differ across some entries, so these numbers
are an artifact ledger rather than a strict paired comparison.

## Inference

Inference supports normal generation and a leaderboard mode:

```bash
uv run python main.py --stage inference --config configs/default.yaml \
    --checkpoint path/to/checkpoint.pt \
    --prompt "The answer is" \
    --leaderboard
```

Use `--mock-testing` to validate the CLI without a ParrotLLM checkpoint. It
loads `openai-community/gpt2` from Hugging Face on demand.

The report describes additional branch-level inference fixes used for
benchmark submissions, including option-text cloze scoring, WinoGrande blank
substitution, LAMBADA whitespace handling, PMI calibration experiments, and a
constrained-letter fallback.

## Data Sources

| Dataset | Purpose |
| --- | --- |
| [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) | pretraining data source |
| [OpenWebText 10k](https://huggingface.co/datasets/stas/openwebtext-10k) | fast development subset |
| [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) | perplexity benchmark and decontamination target |
| NLP26 OWT eval split | course evaluation/decontamination split |
| AG-News classifier labels | topic filtering and controlled data mixtures |

## Development

Run tests with:

```bash
uv run pytest
```

Useful smoke configs live under `configs/*/*dummy.yaml` and
`configs/*/*smoketest.yaml`.
