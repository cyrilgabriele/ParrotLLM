# parrotlabs_parrotllm

Team submission for the HSG NLP FS26 PikoGPT public-benchmark leaderboard.

## Team

- **Name:** ParrotLabs
- **Submission folder:** `Submissions/parrotlabs_parrotllm/`
- **Source repo:** `cyrilgabriele/ParrotLLM` (branch `sft-dpo-gian` for this submission)

## Model

| Property | Value |
|---|---|
| Architecture | Decoder-only transformer (GPT-style) |
| Tokenizer | `openai-community/gpt2` (50k BPE) |
| Vocab size | 50258 (50k + pad/eos/bos) |
| `d_model` | 384 |
| `n_layers` | 14 |
| `n_heads` | 6 |
| `d_ff` | 768 |
| Context length | 1024 |
| Positional encoding | RoPE (theta=10000) |
| Activation | SwiGLU (gated MLP) |
| Norm | RMSNorm (pre-norm) |
| Weight tying | `lm_head.weight` ↔ `tok_emb.weight` |
| Trainable params (unique) | 39.97M |
| State-dict params (raw) | 59.27M |
| Bias on linears | False |

## Submission checkpoint

- **Path:** `Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt` (~458 MB)
- **SHA-256:** `1c131cd13b088e875e0705f5a428fffac394005d8f61c947421c2be8c87bf888`
- **Provenance:** continuation-pair DPO at step 2900 (training loss 0.4400)
- **Reference SFT checkpoint:** Alpaca-template SFT, lr 5e-7, early-stopped at step 300 (loss 0.985 from base 6.79)
- **DPO hyperparameters:** β=0.1, lr=2.0e-6, 1 epoch over 24.5k continuation pairs

### Downloading the checkpoint

The checkpoint is 458 MB (over GitHub's 100 MB per-file limit) and is hosted publicly on
Hugging Face at [`ParrotLabs/parrotlabs_parrotllm`](https://huggingface.co/ParrotLabs/parrotlabs_parrotllm).

Fetch it into the path the leaderboard runner expects:

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='ParrotLabs/parrotlabs_parrotllm',
    filename='parrotlabs_final.pt',
    local_dir='Submissions/parrotlabs_parrotllm/runs',
)
print(p)
"
```

Or via the Hugging Face CLI:

```bash
hf download ParrotLabs/parrotlabs_parrotllm parrotlabs_final.pt \
  --local-dir Submissions/parrotlabs_parrotllm/runs
```

Verify the download with:

```bash
shasum -a 256 Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt
# expected: 1c131cd13b088e875e0705f5a428fffac394005d8f61c947421c2be8c87bf888
```

## Public-bench results (in-process harness, limit=200, temperature=0)

| Benchmark | Score |
|---|---|
| HellaSwag | 22.00% |
| WinoGrande | 58.50% |
| OpenBookQA | 36.50% |
| LAMBADA | 36.50% |
| **Average** | **38.38%** |

These numbers reproduce in `runs/overnight_sft_dpo_bench/10_submission_pmi_off.json`.

## Reproduction

From the project root (`ParrotLLM/`):

```bash
# 1) SFT prepare + train (Alpaca template, lr 5e-7, early-stop on composite_score)
uv run python main.py --stage sft-prepare --config configs/posttraining/sft_benchmark.yaml
uv run python main.py --stage sft         --config configs/posttraining/sft_benchmark.yaml

# 2) DPO prepare + train (continuation pairs, β=0.1, lr 2e-6, 1 epoch)
uv run python main.py --stage dpo-prepare --config configs/posttraining/dpo_continuation.yaml
uv run python main.py --stage dpo         --config configs/posttraining/dpo_continuation.yaml

# 3) Bench (subprocess runner, full leaderboard contract)
cd external/PikoGPT_Leaderboard
uv run python -m leaderboard.run_benchmarks \
  --submission parrotlabs_parrotllm \
  --checkpoint runs/parrotlabs_final.pt \
  --bench hellaswag winogrande openbookqa lambada \
  --limit 200 \
  --python /path/to/ParrotLLM/.venv/bin/python
```

A faster in-process harness is available at `tools/run_bench_inproc.py`
(produces bit-identical accuracy when PMI is off, ~40s vs ~50min):

```bash
uv run python tools/run_bench_inproc.py \
  --checkpoint Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt \
  --bench all --limit 200 --pmi off \
  --output runs/overnight_sft_dpo_bench/10_submission_pmi_off.json
```

## Inference contract

Implemented in `Submissions/parrotlabs_parrotllm/main.py` and `src/inference.py`:

- **Prompt routing.** Inputs are inspected to decide one of three paths:
  1. `mc` — last non-blank line is exactly `Answer:` and ≥2 `^[A-Z]) ...` option lines.
  2. `lambada` — narrative passage ending in a trailing space, ≥80 chars, no MC shape.
  3. `chat` — everything else (SFT/DPO Alpaca template wrap is applied here).
- **Alpaca wrap (chat path only).** `### Instruction:\n<sys>\n\n<user>\n\n### Response:\n` —
  matches the system prompt and template used at training time.
- **MC path: cloze scoring.** For each option, score length-normalized
  log P(" <option>" | full prompt) and pick the argmax. WinoGrande uses option
  substitution into `_` and scores the post-blank tail. HellaSwag/OpenBookQA score
  the option text directly after the `Answer:` line.
- **PMI calibration: OFF by default.** PMI hurt our setup by ~2 pp average across
  configurations (helps HellaSwag +1.5–3 pp but over-corrects WinoGrande −7 pp /
  OpenBookQA −3 pp). Kept as opt-in via `--pmi on` for ablation. See
  [`runs/overnight_sft_dpo_bench/summary.md`](../../runs/overnight_sft_dpo_bench/summary.md)
  Round 3 for the empirical comparison.
- **LAMBADA / chat path: greedy decode with KV cache.** Argmax at temperature=0,
  early-exit on EOS, left-windowed when prompt+continuation exceeds context.
- **Constrained-letter fallback.** If cloze scoring raises, fall back to a 1-token
  greedy generate masked to ids that decode to one of the option letters — so
  the model always emits a valid letter even on malformed inputs.

## Stdout contract (leaderboard mode)

In `--leaderboard` mode `main.py` writes ONLY the generated continuation to stdout:
no banners, no logs, no checkpoint-load chatter. MC outputs a single uppercase
letter; LAMBADA/chat outputs the decoded continuation. Tested against the
leaderboard subprocess runner; 0 invalid outputs across 50-example slice.

## Key configs and logs

- SFT recipe: [`configs/posttraining/sft_benchmark.yaml`](../../configs/posttraining/sft_benchmark.yaml)
- DPO recipe: [`configs/posttraining/dpo_continuation.yaml`](../../configs/posttraining/dpo_continuation.yaml)
- Overnight pipeline summary: [`runs/overnight_sft_dpo_bench/summary.md`](../../runs/overnight_sft_dpo_bench/summary.md)
- SFT training log: `runs/overnight_sft_dpo_bench/03_sft_train.log`
- DPO training log: `runs/overnight_sft_dpo_bench/06_dpo_train.log`
- In-process bench (limit 200, PMI off): `runs/overnight_sft_dpo_bench/10_submission_pmi_off.json`
- Subprocess runner results: `external/PikoGPT_Leaderboard/Results/parrotlabs_parrotllm/parrotlabs_final/`

## What we shipped vs. what we tried

| Idea | Status | Reason |
|---|---|---|
| Alpaca-template SFT | shipped | best single-stage result (+5 pp avg vs. base) |
| Continuation-pair DPO (β=0.1) | shipped | improved every public bench over SFT |
| KV-cache decode | shipped | bit-identical scores, faster chat-path latency |
| Length-normalized DPO | retained as opt-in | lower train loss but −1 pp avg public bench |
| PMI calibration | retained as opt-in | −2 pp avg on our setup |
| WT-103 perplexity tripwire | retained as opt-in | safety net for future SFT runs |

The submitted configuration is the empirically best one; the alternatives are
disabled by default but kept reachable so the team can re-test if the bench
distribution changes.
