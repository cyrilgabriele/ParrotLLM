# ParrotLLM

40M parameter decoder-only LLM built from scratch for the PikoGPT Challenge (NLP Lab FS26).

## Architecture

```
Tokenizer:      GPT-2 (vocab=50,257)              [fixed constraint]
d_model:        320
Layers:         16
Heads:          8  (head_dim=40)
FFN:            SwiGLU, d_ff=854 (8/3 * 320)
Normalization:  RMSNorm, Pre-Norm
Positional:     RoPE
Weight Tying:   Yes (input embed = output head)
Context:        1024                               [fixed constraint]
```

### Parameter Calculation

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

### Why These Choices

| Decision | Choice | Reason | Paper |
|----------|--------|--------|-------|
| Deep & narrow | 16 layers, d=320 | Depth > width at small scale | MobileLLM (Meta, 2024) |
| SwiGLU | d_ff = 8/3 * d_model | Beats GELU at same param cost | Shazeer (2020) |
| RMSNorm + Pre-Norm | Before each sublayer | Same quality, 10-30% faster, stable training | Zhang & Sennrich (2019), Xiong et al. (2020) |
| RoPE | Rotary embeddings | Zero extra params, encodes relative position | Su et al. (2021) |
| Weight tying | Shared embed/head | Saves 16M params, improves perplexity | Press & Wolf (2017) |

> Full analysis with paper references: [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md)

## Datasources

- Full OpenWebText: https://huggingface.co/datasets/Skylion007/openwebtext
- 10k OpenWebText: https://huggingface.co/datasets/stas/openwebtext-10k
