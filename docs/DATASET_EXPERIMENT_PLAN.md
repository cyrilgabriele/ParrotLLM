# Dataset Preprocessing Experiment Plan

## 1. Goal

Use small-scale runs to identify the best data recipe before committing the 24-hour external-GPU run.
Because your final evaluation mixes:

- language modeling on Wikitext-103 and OpenWebText test,
- leaderboard tasks such as LAMBADA, HellaSwag, Winogrande, and OpenBookQA,
- and additional hidden benchmarks,

the safest strategy is to test both:

- broad web-text coverage, and
- higher-density factual/technical mixes.

The experiment matrix below is designed to answer that with controlled preprocessing variants.

---

## 2. Core Principle: Match on Clean Output Tokens

Compare dataset variants at the same final clean-token budget, not at the same raw `target_tokens`.
Different filters retain different fractions of OpenWebText, so equal raw download budgets are not a fair
comparison.

Recommended workflow:

1. Calibration pass:
   Set each variant to `target_tokens: 100000000`, preprocess once, and record `Total tokens kept`.
2. Breadth screen:
   Rebuild each variant to land at roughly `120M-150M` clean tokens.
3. Scaling-law pass:
   Take the best 2 variants and build them at about `50M`, `150M`, and `400M` clean tokens.
4. Final run:
   Train the winning recipe at about `700M-800M` clean tokens for the external GPU run.

As a starting heuristic, `target_tokens: 300000000` is reasonable for the breadth screen, but the
calibration pass should be used to tighten that per variant.

---

## 3. Concrete YAML Variants

All configs below use the same default cleaning backbone:

- decontamination enabled,
- language filtering enabled,
- heuristic code filter enabled,
- heuristic quality filter enabled unless explicitly disabled,
- fuzzy dedup enabled,
- ellipsis filter enabled,
- minimum post-tokenization document length set explicitly to `64`.

### Variant A — `general_clean`

File: `configs/preprocess_var_a.yaml`

Purpose:
- Broad baseline.
- Best hedge for hidden benchmarks and general downstream robustness.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_a.yaml
```

### Variant B — `topic_uniform`

File: `configs/preprocess_var_b.yaml`

Purpose:
- Tests whether any AG-News topic filtering helps while preserving balanced topic diversity.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_b.yaml
```

### Variant C — `knowledge_balanced_resampled`

File: `configs/preprocess_var_c.yaml`

Purpose:
- Drops Sports and forces a near-uniform factual/technical mix across World, Business, and Sci/Tech.
- Useful if class balancing itself helps.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_c.yaml
```

### Variant D — `tech_heavy`

File: `configs/preprocess_var_d.yaml`

Purpose:
- Tests whether a stronger Sci/Tech skew improves learning efficiency.
- High-upside for perplexity, but riskier for broad commonsense coverage.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_d.yaml
```

### Variant E — `knowledge_balanced`

File: `configs/preprocess_var_e.yaml`

Purpose:
- Drops Sports, but does not resample.
- Preserves the natural World/Business/Sci-Tech mix of the sampled OWT slice.
- This is the cleanest way to test whether removing Sports helps without also introducing synthetic class proportions.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_e.yaml
```

### Variant F — `loose_clean`

File: `configs/preprocess_var_f.yaml`

Purpose:
- Same broad baseline as Variant A, but disables the heuristic quality filter.
- Answers whether Phase 5 is improving the corpus or discarding too much useful web text.

Run:

```bash
uv run python main.py --stage preprocess --config configs/preprocess_var_f.yaml
```

---

## 4. Recommended Priority Order

If you cannot run all six variants, use this order:

1. `general_clean` (`preprocess_var_a.yaml`)
2. `knowledge_balanced` (`preprocess_var_e.yaml`)
3. `tech_heavy` (`preprocess_var_d.yaml`)
4. `loose_clean` (`preprocess_var_f.yaml`)
5. `knowledge_balanced_resampled` (`preprocess_var_c.yaml`)
6. `topic_uniform` (`preprocess_var_b.yaml`)

Reasoning:

- A is the strongest broad baseline.
- E isolates the most plausible improvement: remove Sports, keep natural distribution.
- D tests the upside of more information-dense technical text.
- F tests whether your current heuristic quality filter is too aggressive.
- C and B are still useful, but resampling-based topic balancing is a weaker first-order hypothesis.

---

## 5. Recommended Experiment Matrix

### Phase 0: Calibration

Goal:
- Estimate retention rate per preprocessing recipe.

Procedure:
- For each variant A-F, temporarily set `target_tokens: 100000000`.
- Run preprocessing once.
- Record:
  - input documents,
  - final kept documents,
  - `Total tokens kept`,
  - tokens removed by code filter, quality filter, dedup, and ellipsis filter.

Deliverable:
- A table with per-variant retention ratio:
  `clean_tokens / target_tokens`.

### Phase 1: Breadth Screen

Goal:
- Rank dataset variants cheaply but meaningfully.

Procedure:
- For each variant A-F, choose `target_tokens` so final clean output lands around `120M-150M`.
- Train every run with identical model size, optimizer, seed, and training-token budget.
- Evaluate on:
  - Wikitext-103 test perplexity,
  - the provided OWT held-out test set,
  - if feasible, one quick proxy leaderboard sweep.

Primary comparison set:

| Variant | Alias | Main question |
|---|---|---|
| A | general_clean | Best broad baseline? |
| E | knowledge_balanced | Does dropping Sports help without resampling? |
| D | tech_heavy | Does more Sci/Tech improve efficiency? |
| F | loose_clean | Is the quality filter too strict? |
| C | knowledge_balanced_resampled | Does class balancing help beyond simple Sports removal? |
| B | topic_uniform | Does generic topic balancing help at all? |

### Phase 2: Scaling-Law Pass

Goal:
- Pick the winner for the 24-hour run.

Procedure:
- Take the best 2 variants from Phase 1.
- Build matched clean datasets at approximately:
  - `50M`,
  - `150M`,
  - `400M` clean tokens.
- Fit the loss-vs-data trend and choose the variant with the better scaling behavior, not just the best smallest-run result.

### Phase 3: Final External-GPU Run

Goal:
- Maximize final benchmark performance under the 24-hour budget.

Procedure:
- Use the best-scaling variant.
- Build a final clean corpus around `700M-800M` tokens.
- Keep decontamination enabled and verify that both:
  - `data/wikitext-103-test`, and
  - `data/owt-eval/NLP26/NLP26_OWT_eval/test`
  are present before preprocessing.

---

## 6. Expected Outcomes

Working hypothesis:

- `general_clean` or `knowledge_balanced` is the most likely overall winner.
- `tech_heavy` may win on loss/perplexity but is less certain on broad commonsense tasks.
- `loose_clean` is a high-value sanity check because overly aggressive filtering can hurt small-model performance by shrinking coverage too much.

---

## 7. Repo-Specific Notes

- In this codebase, `target_tokens` takes precedence over `dataset_size`, so the variant files intentionally keep `dataset_size: dummy` as a fallback only.
- The effective post-tokenization length filter is `minimum_tokens_per_doc: 64`; that is now set explicitly in all preprocessing variants for clarity.
- Keep the codebase as the source of truth when comments and implementation differ.
