"""Audit how the leaderboard submission tokenizes LAMBADA prompts.

Why this matters: LAMBADA cleaned/test.jsonl prompts deliberately end in a
trailing space. BPE (GPT-2) tokenization treats trailing whitespace as part
of the NEXT token; if the wrapper or inference path strips that space, the
model sees a different token at the boundary than it would during BPE-faithful
LM scoring (e.g. the prompt becomes a token-shorter context that ends on
"any" instead of "any "), and the next-word prediction is OOD vs. how the
model was pretrained / how lm-eval-harness scores LAMBADA.

We don't fix anything here — just print enough diagnostics to verify whether
the inference path preserves or strips the trailing space. The findings are
written to docs/superpowers/notes/2026-05-06-lambada-tokenization-audit.md.

Usage:
    uv run python tools/audit_lambada_prompt.py
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "Submissions" / "PikoGPPT_ParrotLabs"
LAMBADA_PATH = (
    PROJECT_ROOT
    / "external"
    / "PikoGPT_Leaderboard"
    / "leaderboard"
    / "benchmarks"
    / "lambada"
    / "cleaned"
    / "test.jsonl"
)


def _load_submission_main():
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    inference_path = SUBMISSION_DIR / "src" / "inference.py"
    if "src.inference" not in sys.modules:
        spec = importlib.util.spec_from_file_location("src.inference", inference_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["src.inference"] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("submission_main_audit", SUBMISSION_DIR / "main.py")
    main_mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_main_audit"] = main_mod
    spec.loader.exec_module(main_mod)
    return main_mod


def show(label: str, s: str) -> None:
    last30 = s[-30:].replace("\n", "\\n")
    print(f"  {label:>22} (len={len(s):4d}, ends_in_space={s.endswith(' ')!s:5}): ...{last30!r}")


def main() -> None:
    from transformers import GPT2TokenizerFast

    main_mod = _load_submission_main()
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", use_fast=True)

    examples = []
    with LAMBADA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    rng = random.Random(0)
    sample_idxs = rng.sample(range(len(examples)), 5)
    print(f"LAMBADA test.jsonl has {len(examples)} examples; sampling 5 (seeded 0): {sample_idxs}\n")

    for k, idx in enumerate(sample_idxs):
        ex = examples[idx]
        raw = ex["prompt"]
        gold = ex["answer_text"]

        rendered = main_mod.render_prompt_for_inference(
            raw_prompt=raw,
            template="alpaca",
            system_prompt=main_mod.DEFAULT_SYSTEM_PROMPT,
            leaderboard=True,
        )
        wrapped = rendered.text

        raw_ids = tok.encode(raw, add_special_tokens=False)
        wrapped_ids = tok.encode(wrapped, add_special_tokens=False)
        gold_with_space_ids = tok.encode(" " + gold, add_special_tokens=False)
        gold_no_space_ids = tok.encode(gold, add_special_tokens=False)

        print(f"=== Example {k} (idx={idx}, id={ex['id']}) ===")
        print(f"  gold answer_text: {gold!r}")
        print(f"  rendered.kind={rendered.kind!r}")
        show("raw prompt", raw)
        show("wrapped (inference)", wrapped)
        print(f"  raw last 5 token IDs:     {raw_ids[-5:]}  decoded={[tok.decode([t]) for t in raw_ids[-5:]]}")
        print(f"  wrapped last 5 token IDs: {wrapped_ids[-5:]}  decoded={[tok.decode([t]) for t in wrapped_ids[-5:]]}")
        print(f"  ' {gold}' tokens: {gold_with_space_ids}  decoded={[tok.decode([t]) for t in gold_with_space_ids]}")
        print(f"  '{gold}'  tokens: {gold_no_space_ids}  decoded={[tok.decode([t]) for t in gold_no_space_ids]}")

        # The acid test: does the wrapped prompt's tokenization make the gold
        # word a "with leading space" token (matching lm-eval-harness LAMBADA
        # convention), or did stripping the trailing space turn the boundary
        # into "no leading space"?
        if wrapped.endswith(" "):
            verdict = "PRESERVED trailing space — gold word follows ' '-prefix BPE convention"
        else:
            # If wrapped ended without a trailing space but raw had one, then
            # the wrapper stripped it. The next emitted token would be a "with
            # leading space" token regardless (because BPE encodes the space
            # as part of the next token), but the model's *prefix tokenization*
            # would differ from how it would have looked during pretraining
            # of the same passage.
            verdict = "STRIPPED trailing space — boundary token differs from raw passage"
        print(f"  >> verdict: {verdict}\n")


if __name__ == "__main__":
    main()
