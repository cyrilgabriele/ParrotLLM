# Dataset Preprocessing Experiment Plan

## 1. Token Budget

**Chinchilla** (Hoffmann et al., 2022, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) shows that
compute-optimal training requires:

$$D_{optimal} = 20 \times N$$

This model is ~35M parameters, so:

$$D_{optimal} = 20 \times 35\text{M} = 700\text{M tokens}$$

For **comparing dataset variants cheaply**, the full budget is not needed. **Kaplan et al.** (2020,
[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)) demonstrate that scaling trends are visible and
stable across many orders of magnitude — a fraction of the optimal budget is sufficient to rank
variants. ~15–20% of the Chinchilla budget (≈ 100–150M clean output tokens) will separate good from
bad preprocessing choices with high confidence.

**Because web text loses ~40–55% of raw documents to quality filters** (documented in the Gopher
pipeline, Rae et al., 2021, [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)), the
`--target-tokens` input must be roughly double the desired output:

| Goal | `--target-tokens` |
|---|---|
| Comparison run (~150M clean tokens) | `300000000` |
| Full Chinchilla-optimal final run (~700M clean tokens) -> we go for ~800M to have some buffer | `1400000000` |

---

## 2. Why Topic Filtering Is Worth Testing

**phi-1** (Gunasekar et al., 2023, [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)) is the
strongest peer-reviewed evidence: a 1.3B model trained on 7B carefully curated tokens outperforms
models 10× its size trained on unfiltered web data. The key mechanism is raising the **average
information density** of each token seen during training — which is exactly what topic filtering does.

**LLaMA** (Touvron et al., 2023, [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) reinforces
this: LLaMA-13B, trained longer on filtered public data, beats GPT-3 (175B) on most benchmarks. The
paper explicitly credits data curation, not model size.

---

## 3. Experiment Matrix

All variants use `--subset-seed 42` so they draw from the identical shuffled OpenWebText slice —
differences in final perplexity are attributable solely to preprocessing, not sampling noise.

---

### Variant A — Baseline (no topic filter)

Motivation: establishes the raw web-text baseline. Gopher
([arXiv:2112.11446](https://arxiv.org/abs/2112.11446)) used a similar unfiltered-but-quality-filtered
pipeline as its starting point.

```bash
python main.py --stage preprocess --config configs/preprocess_var_a.yaml

```

---

### Variant B — All 4 topics, uniform distribution

Motivation: tests whether any topic filtering helps at all, before introducing skew. The domain
breadth matches the Gopher/MassiveText approach of keeping diverse but filtered text.

```bash
python main.py --stage preprocess --config configs/preprocess_var_b.yaml

```

---

### Variant C — Knowledge-focused (no Sports)

Motivation: Sports text is factually dense but domain-narrow and linguistically repetitive (game
reports, scores). Removing it raises the average reasoning and factual content per token — consistent
with the phi-1 finding that "textbook quality" data disproportionately improves model quality.

```bash
python main.py --stage preprocess --config configs/preprocess_var_c.yaml

```

---

### Variant D — Sci/Tech-heavy

Motivation: Kaplan et al. ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)) show that
information-dense text (technical writing, scientific prose) produces lower loss at the same token
budget than low-density text. This tests whether doubling down on technical/scientific content
accelerates convergence.

```bash
python main.py --stage preprocess --config configs/preprocess_var_d.yaml

```

---

## 4. What Each Comparison Answers

| Comparison | Question | Key literature |
|---|---|---|
| A vs. B | Does topic filtering help vs. raw filtered web? | phi-1 [2306.11644](https://arxiv.org/abs/2306.11644) |
| B vs. C | Does dropping Sports improve general LM quality? | phi-1 [2306.11644](https://arxiv.org/abs/2306.11644) |
| C vs. D | Does Sci/Tech skew accelerate loss convergence? | Kaplan et al. [2001.08361](https://arxiv.org/abs/2001.08361) |
| Winner vs. full run | Does the same advantage hold at Chinchilla scale? | Chinchilla [2203.15556](https://arxiv.org/abs/2203.15556) |

---

## 5. Evaluation Protocol

Train each variant for **exactly 20,000 steps** (the same fixed compute budget). Evaluate exclusively
on **Wikitext-103** — not the variant's own `val.bin` — so the evaluation target is identical across
all runs. Report validation perplexity $e^{L}$.

The variant with the lowest Wikitext-103 perplexity at step 20,000 is then used for the full
`--target-tokens 1400000000` Chinchilla-optimal training run. LLaMA
([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) demonstrated that identifying the best data mix
early and then scaling it is more cost-effective than running all variants at full scale.

---

## 6. Literature References

| Paper | Authors | Year | Link |
|---|---|---|---|
| Scaling Laws for Neural Language Models | Kaplan et al. (OpenAI) | 2020 | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) |
| Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al. (DeepMind) | 2022 | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |
| Scaling Language Models: Gopher | Rae et al. (DeepMind) | 2021 | [arXiv:2112.11446](https://arxiv.org/abs/2112.11446) |
| LLaMA: Open and Efficient Foundation Language Models | Touvron et al. (Meta) | 2023 | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |
| Textbooks Are All You Need (phi-1) | Gunasekar et al. (Microsoft) | 2023 | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) |
