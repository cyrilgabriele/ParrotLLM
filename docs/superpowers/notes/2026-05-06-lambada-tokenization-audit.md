# LAMBADA prompt tokenization audit

**Date:** 2026-05-06
**Branch:** sft-dpo-gian
**Audited code:** `Submissions/PikoGPPT_ParrotLabs/main.py:render_prompt_for_inference`,
`Submissions/PikoGPPT_ParrotLabs/src/inference.py:is_lambada_shape`

## Why this audit

The leaderboard's LAMBADA test prompts (`external/PikoGPT_Leaderboard/leaderboard/benchmarks/lambada/cleaned/test.jsonl`)
end in a deliberate trailing space — see `test_meta.json`:

> "Prompt is the passage WITHOUT the last word, plus a trailing space. Gold is the last word (answer_text)."

GPT-2 BPE encodes a trailing `" "` as token id `220` (a standalone-space
token). That token is unusual in natural running-text tokenization, where
the space normally gets absorbed into the leading-space variant of the
following word (e.g. `" Dag"`, `" insurance"`). The concern: if our SFT
pipeline trained on prompts that *don't* end in a bare space-token, then at
inference the model is OOD vs. its pretraining distribution and the
next-word prediction degrades.

## What was audited

`tools/audit_lambada_prompt.py` samples 5 LAMBADA test examples (seed 0),
runs each through `render_prompt_for_inference(template="alpaca",
leaderboard=True)`, and prints:

- whether the wrapped (inference-bound) prompt still ends in a space,
- the last 5 token IDs of the wrapped prompt vs. the raw prompt,
- the BPE encoding of `" " + answer` vs. `answer`.

## Finding

**The submission deliberately strips the trailing space.** The relevant line:

```python
# Submissions/PikoGPPT_ParrotLabs/main.py:120
if is_lambada_shape(raw_prompt):
    return RenderedPrompt(kind="lambada", text=raw_prompt.rstrip())
```

For all 5 sampled examples, the wrapped prompt ends one character shorter
than the raw prompt and the last 5 token IDs differ at the boundary (the
raw version ends in token `220` (`" "`); the wrapped version ends in the
previous content token, e.g. `" is"`, `" the"`).

Example 0 (idx=3155, gold=`"Irina"`):

- raw last 5 ids: `[503, 284, 2298, 510, 220]` → `[" out", " to", " pick", " up", " "]`
- wrapped last 5 ids: `[1182, 503, 284, 2298, 510]` → `[" head", " out", " to", " pick", " up"]`
- `" Irina"` BPE: `[5686, 1437]` (`[" Ir", "ina"]`)
- `"Irina"` BPE: `[40, 22267]` (`["I", "rina"]`)

## Verdict: path is correct, no fix needed

The `rstrip()` is **intentional and correct** for our setup:

1. **It matches lm-eval-harness LAMBADA convention.** lm-eval-harness scores
   `log P(" " + target | context_without_trailing_space)`. The leading space
   becomes the first token of the *target*, not the last token of the
   *prefix*. By stripping, we put the trailing space exactly where lm-eval
   puts it: at the boundary where it gets absorbed into the leading-space
   variant of the predicted word.

2. **It keeps the prefix on natural BPE.** The bare token id `220` is rare
   in pretraining text — it appears mostly at intra-paragraph whitespace
   boundaries, not as the final token of a multi-sentence narrative. By
   stripping, we present the model a prefix whose final token (e.g. `" is"`,
   `" not"`, `","`) is the kind of last-token it sees billions of times in
   pretraining. The predicted next token is then `" Dag"`, `" insurance"`,
   `" maximum"` — also natural BPE.

3. **The runner's parser handles either output shape.** `parse_lambada_word`
   in `external/.../run_benchmarks.py` does `gen.lstrip()` then takes the
   first whitespace-delimited word and normalizes it. Both paths
   (preserved-space → first-token-no-leading-space, stripped-space →
   first-token-with-leading-space) round-trip to the same normalized word
   under that parser. So the rstrip doesn't break the leaderboard contract.

4. **We have no LAMBADA-shape SFT data.** Per the overnight summary:
   `cbt`, `bookcorpus` (and similar narrative-continuation sources) were
   dropped because `datasets` 4.5 removed legacy script loaders. So the
   model's only "predict the last word given a passage" exposure comes
   from pretraining on natural BPE text. The stripped-prefix path
   matches that distribution; the preserved-space path does not.

## What this implies for benches

LAMBADA scores reported in `runs/overnight_sft_dpo_bench/summary.md` (SFT
34.00%, DPO 36.50%) reflect the stripped-space path. They are not biased
by an inference-time tokenization mismatch — the inference path matches
both lm-eval convention and natural BPE. Any further LAMBADA gains have
to come from the underlying continuation distribution (more pretraining
data, more LAMBADA-shape SFT data once we re-add CBT/BookCorpus, or DPO
with continuation-pair preferences scoped to narrative text).

## No fix applied

The audit script is checked in at `tools/audit_lambada_prompt.py` for
future re-runs (e.g. if the wrapper changes). The submission code in
`main.py:render_prompt_for_inference` is unchanged.
