# Architecture Decisions - Backed by Papers

Every design choice below is justified by published research. Papers are cited with arxiv IDs for easy lookup.

---

## 1. Depth vs Width: Go Deep and Narrow

### The Evidence

**MobileLLM** (Liu et al., Meta, ICML 2024 - [arXiv:2402.14905](https://arxiv.org/abs/2402.14905))

This is the single most relevant paper for our project. Meta specifically studied sub-billion parameter models and found:

> "Deeper and thinner models generally outperform their wider counterparts."

Their ablation at ~125M params (Table 9 from the paper):

| Layers | Hidden Dim | Avg Zero-Shot Score |
|--------|-----------|---------------------|
| 4 | 1280 | 43.3% |
| 12 | 768 | 43.9% |
| **30** | **512** | **44.8%** |
| 62 | 384 | 44.7% |

At ~350M params:

| Layers | Hidden Dim | Avg Zero-Shot Score |
|--------|-----------|---------------------|
| 5 | 2048 | 47.1% |
| 15 | 1280 | 48.7% |
| **32** | **896** | **49.8%** |
| 66 | 640 | 49.5% |

**Key finding:** There's a sweet spot. Going too narrow (62 layers / d=384) starts to hurt again. The optimum was around 30 layers for 125M params. Scaled to our 40M budget, the ratio suggests **more layers with moderate width is better than fewer layers with large width**.

**Kaplan et al.** (OpenAI, 2020 - [arXiv:2001.08361](https://arxiv.org/abs/2001.08361))

Found that architecture shape matters weakly compared to total parameter count:

> "The loss depends only weakly on the aspect ratio d_model/n_layers."

However, they noted models with d_model/n_layers ratio outside the 20-200 range underperformed. MobileLLM's later work challenged this at small scale, showing depth matters MORE when parameters are scarce.

### What this means for us (40M params)

MobileLLM's findings suggest we should **not** copy GPT-2 small's 12-layer design scaled down. Instead, lean toward more layers (14-20) with a moderate hidden dim (256-320). But we're constrained by our vocab size (50,257 vs MobileLLM's 32,000), which eats more of our parameter budget in the embedding table.

### Decision

**Favor depth over width**, but stay within the sweet spot - not excessively narrow.

---

## 2. Weight Tying: Mandatory at Our Scale

### The Evidence

**Press & Wolf** (2017 - [arXiv:1608.05859](https://arxiv.org/abs/1608.05859))

> "Tying input and output embeddings leads to a significant reduction in perplexity across a variety of neural network language models."

Additionally, weight tying reduces model size without harming (and actually improving) performance. The tied embedding evolves in a way that benefits both input and output tasks.

**MobileLLM** (2024) confirmed this is critical at small scale:

> "For a 125M-parameter model, the embedding layers account for over 20% of parameters. Sharing the input and output embeddings reduces the number of parameters by 16M, approximately 11.8% of total parameters."

For **our 40M model with vocab 50,257**, the math is even more dramatic:

| d_model | Embedding params | % of 40M budget | Savings from tying |
|---------|-----------------|-----------------|-------------------|
| 512 | 25.7M | 64% | 25.7M freed |
| 384 | 19.3M | 48% | 19.3M freed |
| 320 | 16.1M | 40% | 16.1M freed |
| 256 | 12.9M | 32% | 12.9M freed |

Without weight tying, a d_model=384 model would spend 48% of its params just on the embedding table, then need another 48% for the output projection. That leaves almost nothing for the actual transformer layers.

**Pythia** (Biderman et al., 2023) did NOT use weight tying, but their smallest model was 70M - nearly double our budget. At 40M, we don't have that luxury.

### Decision

**Weight tying is non-negotiable.** Every major small model uses it.

---

## 3. Positional Encoding: RoPE vs Learned

### The Evidence

**RoFormer** (Su et al., 2021 - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864))

RoPE showed:
- ~30% faster convergence compared to learned absolute positional embeddings at billion-parameter scale
- Better performance on long text classification benchmarks
- Encodes **relative** position (distance between tokens matters) vs absolute position (each position is independent)
- Long-term decay property: distant tokens naturally have weaker attention, which is linguistically intuitive
- Zero additional parameters

> "Across a large suite of setups including regular, linear, and local self-attention, RoPE either matches or surpasses all other methods for injecting positional information into transformers."

**Pythia-70M** used RoPE. **MobileLLM** used RoPE. **LLaMA** (all sizes) uses RoPE.

**However:** GPT-2 used learned absolute embeddings and they work fine at context=1024. RoPE's advantages are more pronounced at longer contexts. At 1024 tokens, the difference is smaller.

### Risk Assessment

RoPE is more complex to implement. A subtle bug in the rotation matrices will silently degrade performance. Learned embeddings are 3 lines of code (`nn.Embedding(1024, d_model)`).

### Decision

**RoPE if the team is confident implementing it correctly. Learned absolute if not.** The performance gap at context=1024 is small. Correctness matters more.

---

## 4. Activation Function: SwiGLU vs GELU

### The Evidence

**Shazeer** (Google, 2020 - [arXiv:2002.05202](https://arxiv.org/abs/2002.05202))

Tested GLU variants as FFN replacements in T5. To keep parameter count equal, he reduced FFN hidden dimension by factor 2/3 (since SwiGLU uses 3 matrices instead of 2).

Results: **GEGLU and SwiGLU produced the best perplexities** across pretraining and downstream tasks. Both consistently outperformed standard GELU and ReLU FFN layers.

Adopted by: **LLaMA** (all versions), **PaLM** (Google), **MobileLLM**, **Mistral**.

**The tradeoff at 40M params:**

| Activation | FFN matrices | FFN hidden (for d=320) | Params per FFN layer |
|-----------|-------------|----------------------|---------------------|
| GELU | 2 (up + down) | 1280 (4x) | 2 x 320 x 1280 = 819K |
| SwiGLU | 3 (gate + up + down) | 854 (8/3x) | 3 x 320 x 854 = 820K |

The parameter cost is the **same** (that's the whole point of the 2/3 reduction). SwiGLU just distributes them differently: narrower FFN but with a gating mechanism.

### Decision

**SwiGLU.** Same parameter cost, proven better performance, used by every modern model. The 2/3 FFN reduction compensates for the extra matrix.

---

## 5. Normalization: RMSNorm + Pre-Norm

### The Evidence

**Zhang & Sennrich** (2019 - [arXiv:1910.07467](https://arxiv.org/abs/1910.07467))

RMSNorm hypothesis: the re-centering step in LayerNorm (subtracting the mean) is unnecessary. Only the re-scaling (dividing by RMS) matters.

Results:
- **Comparable performance** to LayerNorm across all tested architectures
- **7-64% speedup** depending on architecture
- For transformers specifically: **11-34% speedup**
- Now used in LLaMA, Mistral, MobileLLM

**Xiong et al.** (2020 - [arXiv:2002.04745](https://arxiv.org/abs/2002.04745))

Studied Pre-Norm vs Post-Norm placement:

> "In the Post-LN Transformer, the expected gradients of the parameters near the output layer are large, making training unstable when using a large learning rate."

> "In the Pre-LN Transformer, the gradients are well-behaved at initialization."

Practical impact:
- Pre-Norm: **smooth, stable training curves**, no loss spikes
- Post-Norm: loss spikes and unstable gradient norms
- Pre-Norm allows **removing or reducing warmup schedules entirely**
- Pre-Norm reaches comparable results with **significantly less training time and hyper-parameter tuning**

### Decision

**RMSNorm + Pre-Norm.** Faster, more stable training, less hyperparameter sensitivity. If implementation simplicity is paramount, `nn.LayerNorm` with Pre-Norm placement is also fine - the norm type matters less than the placement.

---

## 6. Training Data Amount: Chinchilla and Beyond

### The Evidence

**Hoffmann et al.** (DeepMind, 2022 - [arXiv:2203.15556](https://arxiv.org/abs/2203.15556))

The Chinchilla scaling law: for compute-optimal training, use **~20 tokens per parameter**.

For 40M params: 40M x 20 = **800M tokens minimum**.

**But:** Chinchilla optimizes for a fixed compute budget. If you care about final model quality (not compute efficiency), you should **overtrain** well beyond 20:1.

**LLaMA** (Touvron et al., 2023) trained their 7B model on 1T+ tokens = **~143 tokens per parameter**. Their finding: smaller models benefit enormously from seeing more data.

**Pythia-70M** was trained on **300B tokens** = ~4,300 tokens per parameter. Massively beyond Chinchilla-optimal, but this was intentional for research purposes.

### What this means for us

Our OpenWebText subset is the constraint. We should use **as much data as we can** within our compute budget (<=24h on 8xV100; we plan for 23h to stay safe). The Chinchilla minimum of 800M tokens is a floor, not a ceiling. If the dataset allows, aim for multiple epochs to effectively see more tokens.

### Decision

**Train on as many tokens as compute allows.** Chinchilla's 20:1 is a minimum. More data = better model, especially at small scale.

---

## 7. Attention: Standard MHA

### The Evidence

**MobileLLM** used **GQA** (Grouped Query Attention) to save parameters. But their smallest model was 125M - triple our budget. At 40M params, the parameter savings from GQA are marginal and the implementation complexity isn't worth it.

**Pythia-70M** (closest to our scale) used standard **MHA with 8 heads**.

GQA's primary benefit is reducing KV-cache memory at inference. For our demo chat interface serving one user, this is irrelevant.

### Decision

**Standard Multi-Head Attention.** Simpler, well-understood, sufficient at our scale.

---

## Reference Architectures from Literature

### Models closest to our 40M param budget:

| | Pythia-70M | TinyStories-28M | MobileLLM-125M | GPT-2 Small |
|--|-----------|----------------|----------------|------------|
| Params | 70M | ~28M | 125M | 124M |
| d_model | 512 | ~512 | 576 | 768 |
| Layers | 6 | ~8 | **30** | 12 |
| Heads | 8 | ~8 | 9 (GQA) | 12 |
| d_ff | 2048 | ~2048 | ~1536 (SwiGLU) | 3072 |
| Vocab | 50,304 | ~10K | 32,000 | 50,257 |
| Context | 2,048 | ~1024 | 2,048 | 1,024 |
| Positional | RoPE | Learned | RoPE | Learned |
| Norm | LayerNorm | LayerNorm | RMSNorm | LayerNorm |
| Activation | GELU | GELU | SwiGLU | GELU |
| Emb. Tying | No | Yes | Yes | Yes |

Source: [Pythia HuggingFace](https://huggingface.co/EleutherAI/pythia-70m), [MobileLLM arXiv](https://ar5iv.labs.arxiv.org/html/2402.14905), [TinyStories arXiv](https://arxiv.org/abs/2305.07759)

---

## Our Configuration (Paper-Backed)

Given our constraint of **40M params, vocab=50,257, context=1024**:

```
Tokenizer:       GPT-2 (fixed)                    [Constraint]
Weight Tying:    Yes                               [Press & Wolf 2017, MobileLLM 2024]
Positional:      RoPE (or Learned as fallback)     [Su et al. 2021, Pythia]
Norm:            RMSNorm, Pre-Norm                 [Zhang & Sennrich 2019, Xiong et al. 2020]
Activation:      SwiGLU                            [Shazeer 2020, LLaMA, MobileLLM]
Attention:       Standard MHA                      [Pythia-70M]
```

### Concrete Config: d_model=320, 16 layers

```
Embedding (tied):  50,257 * 320                    = 16,082,240

Per transformer layer:
  Q projection:    320 * 320                       =    102,400
  K projection:    320 * 320                       =    102,400
  V projection:    320 * 320                       =    102,400
  O projection:    320 * 320                       =    102,400
  SwiGLU gate:     320 * 854                       =    273,280
  SwiGLU up:       320 * 854                       =    273,280
  SwiGLU down:     854 * 320                       =    273,280
  RMSNorm (x2):    320 * 2                         =        640
  ─────────────────────────────────────────────────────────────
  Subtotal per layer:                              =  1,230,080

16 layers:         1,230,080 * 16                  = 19,681,280
Final RMSNorm:     320                             =        320

TOTAL:             16,082,240 + 19,681,280 + 320   = 35,763,840  (~35.8M)
Headroom:          40M - 35.8M                     =  ~4.2M spare
```

### Alternative Configs

**Option B: Deeper (pushing MobileLLM's finding)**
```
d_model=256, layers=24, heads=8 (head_dim=32)
FFN=682 (8/3 * 256, SwiGLU)
Embedding: 50,257 * 256 = 12.9M (tied)
Per layer: 786K
24 layers: 18.9M
Total: ~31.8M params (8.2M headroom)
```

**Option C: Safe/Classic (Pythia-inspired)**
```
d_model=384, layers=8, heads=6 (head_dim=64)
FFN=1536 (4x, GELU)
Embedding: 50,257 * 384 = 19.3M (tied)
Per layer: 1.77M
8 layers: 14.2M
Total: ~33.5M params (6.5M headroom)
```

### Which option the papers favor

- **MobileLLM** would favor **Option A or B** (depth over width)
- **Kaplan et al.** would say "it doesn't matter much, just get the total params right"
- **Pythia** would favor **Option C** (proven, simple, classic proportions)

### Recommendation

**Option A (d=320, 16 layers)** as the primary config. It incorporates MobileLLM's depth-over-width finding without going to extremes, uses modern components (SwiGLU, RMSNorm), and stays well within budget. If it doesn't work, fall back to **Option C** which is the safest proven design.

---

## Scaling Law Analysis for Our Compute Budget

### Our Constraints
- **Model:** 40M parameters (N = 4 x 10^7)
- **Compute:** <=24h on 8xV100 (plan for **23h** to stay within quota) = **184 GPU-hours**
- **Data:** OpenWebText subset (fixed)

### How many tokens can we process?

**V100 specs** ([NVIDIA datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf)):
- Peak FP16 Tensor Core: ~125 TFLOPS
- Realistic MFU (Model FLOPs Utilization) for a small model: **30-40%**

**The formula** ([EleutherAI Transformer Math 101](https://blog.eleuther.ai/transformer-math/)):
```
C = 6 * N * D

Where:
  C = total compute (FLOPs)
  N = parameters
  D = tokens processed
  6 = 2 for forward + 4 for backward pass
```

**Practical throughput estimate:**

For a tiny 40M model on 8xV100, estimated ~50K-150K tokens/second:

| Throughput | Tokens in 23h | Tokens/param ratio |
|-----------|--------------|-------------------|
| 50K tok/s (conservative) | **4.14B** | 116x |
| 100K tok/s (moderate) | **8.28B** | 232x |
| 150K tok/s (optimistic) | **12.42B** | 347x |

### Chinchilla comparison

**Chinchilla says:** 20 tokens per parameter = **800M tokens** for 40M params.

We can process **~6-17x more** than Chinchilla-optimal even with the tighter 23h budget. This is good news.

### Predicted loss (Chinchilla scaling law)

Using: **L(N, D) = E + A/N^alpha + B/D^beta**

With fitted constants from [Hoffmann et al.](https://arxiv.org/abs/2203.15556):

| Scenario | Tokens (D) | Tokens/Param | Predicted Loss | ~Perplexity |
|----------|-----------|-------------|---------------|-------------|
| Chinchilla-optimal | 800M | 20x | ~4.75 | ~116 |
| Conservative (23h) | 4.14B | 116x | ~4.06 | ~58 |
| Moderate (23h) | 8.28B | 232x | ~3.87 | ~48 |
| Optimistic (23h) | 12.42B | 347x | ~3.78 | ~44 |

> **Caveat:** These predictions extrapolate Chinchilla's scaling laws (calibrated at 70M-16B) down to 40M params. Exact numbers may differ.

### Data repetition - how many epochs are safe?

**Muennighoff et al.** (2023 - [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)) - "Scaling Data-Constrained Language Models"

> Up to **4 epochs** of repetition degrades quality minimally. Beyond 4 epochs, returns diminish significantly and quality starts degrading.

| Subset size | Max useful epochs | Effective tokens |
|-------------|-------------------|-----------------|
| 1B tokens | 4 | ~4B |
| 2B tokens | 4 | ~8B |
| 4B tokens | 3-4 | ~12-16B |
| 8B tokens | 2-3 | ~16-24B |

### Scaling law verdict

**LLaMA's approach fits us best.** We have more compute than Chinchilla requires. Strategy: **overtrain aggressively** on as many tokens as compute allows, up to ~4 epochs of the dataset.

---

## Paper Reference List

| Paper | Authors | Year | Key Finding for Us |
|-------|---------|------|-------------------|
| [MobileLLM](https://arxiv.org/abs/2402.14905) | Liu et al. (Meta) | 2024 | Depth > width at small scale |
| [GLU Variants](https://arxiv.org/abs/2002.05202) | Shazeer (Google) | 2020 | SwiGLU beats GELU |
| [RoFormer](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | RoPE beats learned positions |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | Zhang & Sennrich | 2019 | RMSNorm = LayerNorm quality, 10-30% faster |
| [Pre-Norm](https://arxiv.org/abs/2002.04745) | Xiong et al. | 2020 | Pre-Norm is more stable, needs less warmup |
| [Weight Tying](https://arxiv.org/abs/1608.05859) | Press & Wolf | 2017 | Tying embeddings improves perplexity |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | Hoffmann et al. (DeepMind) | 2022 | ~20 tokens/param minimum, more is better |
| [Scaling Laws](https://arxiv.org/abs/2001.08361) | Kaplan et al. (OpenAI) | 2020 | Shape matters less than total params |
| [Pythia](https://arxiv.org/abs/2304.01373) | Biderman et al. (EleutherAI) | 2023 | Reference configs for 70M-12B models |
| [TinyStories](https://arxiv.org/abs/2305.07759) | Eldan & Li (Microsoft) | 2023 | Small models can be surprisingly capable |
| [LLaMA](https://arxiv.org/abs/2302.13971) | Touvron et al. (Meta) | 2023 | Overtrain beyond Chinchilla for better quality |
| [Data-Constrained](https://arxiv.org/abs/2305.16264) | Muennighoff et al. | 2023 | Up to 4 epochs safe for data repetition |
| [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) | EleutherAI | 2023 | FLOPs estimation: C = 6ND |
