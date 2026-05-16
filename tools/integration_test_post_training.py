"""End-to-end integration test for the Pretrain → SFT → DPO pipeline.

Loads the three checkpoints in the chain and runs a curated prompt set
through each at temperature 0 (deterministic, comparable). Reports per-
stage aggregate metrics so you can see the chain *as a chain* rather
than each stage in isolation.

Run: uv run python -m tools.integration_test_post_training

What it asserts visually:
- Pretrain is incoherent on Alpaca-formatted prompts (it's never seen the
  template). This is the correct baseline; a "fluent" pretrain would
  indicate the SFT didn't actually move the model.
- SFT terminates on EOS within a short budget — the response-only loss
  taught the model where a turn ends.
- DPO ≈ SFT in template/EOS behavior (DPO inherits both) but should be
  at least as good on the preference-aligned criterion (here: gives an
  answer rather than refusing-by-looping). DPO is NOT expected to add
  factual knowledge — the preference set was orca_dpo_pairs, not a
  fact-recall corpus.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch

from src.eval.inference import generate, load_model_from_checkpoint
from src.post_training.sft import format_sft_prompt
from src.utils import build_tokenizer


# ── The exact chain. Pretrain path read from runs/sft_v2_run.log line 23. ──
PIPELINE = [
    ("pretrain",
     r"runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt"),
    ("sft",
     r"runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt"),
    ("dpo",
     r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt"),
]


@dataclass(frozen=True)
class Probe:
    name: str           # short label for the table
    instruction: str    # passed to format_sft_prompt
    must_contain: tuple[str, ...] = ()   # case-insensitive substrings expected


PROBES: tuple[Probe, ...] = (
    # Capitals of major countries — the most common factual triplet on the web.
    Probe("cap_france",  "What is the capital of France?",            ("paris",)),
    Probe("cap_germany", "What is the capital of Germany?",           ("berlin",)),
    Probe("cap_japan",   "What is the capital of Japan?",             ("tokyo",)),
    Probe("cap_uk",      "What is the capital of the United Kingdom?",("london",)),
    Probe("cap_usa",     "What is the capital of the United States?", ("washington",)),
    Probe("cap_italy",   "What is the capital of Italy?",             ("rome",)),
    # Very-common attribute questions
    Probe("sky_color",   "What color is the sky?",                    ("blue",)),
    Probe("grass_color", "What color is grass?",                      ("green",)),
    Probe("days_week",   "How many days are there in a week?",        ("7", "seven")),
    Probe("months_year", "How many months are in a year?",            ("12", "twelve")),
    # Definitions of common nouns
    Probe("def_dog",     "What is a dog?",                            ("animal", "mammal", "pet", "domest")),
    Probe("def_water",   "What is water?",                            ("liquid", "h2o", "drink")),
)


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of n-grams that occur more than once.

    1.0 → fully degenerate (every n-gram repeats), 0.0 → no repetition.
    Tuned for catching `the capital of france is the capital of france`-style
    loops without flagging natural-language repetition (e.g. "the the")."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n + 1:
        return 0.0
    grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


@dataclass
class Result:
    stage: str
    probe: str
    text: str               # decoded post-EOS-truncated response
    raw_text: str           # full decoded gen incl. <|endoftext|> if present
    tokens_to_eos: int      # number of tokens before first EOS (or len if none)
    eos_emitted: bool
    rep4: float             # repetition_score(text)
    contains_expected: bool


def _truncate_at_marker(text: str) -> str:
    """Mirror the chat UI's post-decode truncation rules."""
    for marker in ("\n###", "###", "<|endoftext|>"):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx]
    return text


def _run_one(model, tok, ctx_len, device, stage: str, probe: Probe,
             max_new: int = 80) -> Result:
    if stage == "pretrain":
        # Pretrain has never seen the Alpaca template, so prompt it as raw
        # text continuation — that's what it was trained on. This gives
        # pretrain its fairest possible shot at producing something
        # coherent for the comparison.
        prompt = probe.instruction + "\n"
        eos_id = None  # pretrain doesn't reliably emit EOS at semantic ends
    else:
        prompt = format_sft_prompt(probe.instruction)
        eos_id = tok.eos_token_id

    in_ids = tok.encode(prompt)
    x = torch.tensor([in_ids], dtype=torch.long, device=device)
    out = generate(model, x, max_new_tokens=max_new,
                   temperature=0.0, top_k=50, top_p=0.9,
                   context_length=ctx_len, eos_token_id=eos_id)
    gen_ids = out[0, len(in_ids):].tolist()
    eos_pos = gen_ids.index(tok.eos_token_id) if tok.eos_token_id in gen_ids else -1
    raw = tok.decode(gen_ids)
    text = _truncate_at_marker(raw).strip()

    contains = bool(probe.must_contain) and any(
        s.lower() in text.lower() for s in probe.must_contain
    )
    return Result(
        stage=stage, probe=probe.name, text=text, raw_text=raw,
        tokens_to_eos=eos_pos if eos_pos >= 0 else len(gen_ids),
        eos_emitted=eos_pos >= 0,
        rep4=repetition_score(text),
        contains_expected=contains,
    )


def _stage_summary(results: list[Result]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    eos_rate = sum(r.eos_emitted for r in results) / n
    avg_len = sum(r.tokens_to_eos for r in results) / n
    avg_rep = sum(r.rep4 for r in results) / n
    probes_with_truth = [r for r in results if r.text and r.contains_expected is not None]
    truth_probes = [r for r in results if any(
        p.must_contain for p in PROBES if p.name == r.probe
    )]
    contains_rate = (
        sum(r.contains_expected for r in truth_probes) / len(truth_probes)
        if truth_probes else float("nan")
    )
    return {
        "n": n,
        "eos_rate": eos_rate,
        "avg_tokens_to_eos": avg_len,
        "avg_rep4": avg_rep,
        "contains_expected_rate": contains_rate,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer()
    print(f"Device: {device}")
    print(f"Tokenizer: vocab={len(tok)}, eos_id={tok.eos_token_id}, "
          f"pad_id={tok.pad_token_id}\n")

    by_stage: dict[str, list[Result]] = {}
    for stage, ckpt_path in PIPELINE:
        print(f"--- Loading {stage:<8} ← {ckpt_path}")
        model, cfg = load_model_from_checkpoint(ckpt_path, device)
        ctx = cfg["model"]["context_length"]
        results: list[Result] = []
        for probe in PROBES:
            results.append(_run_one(model, tok, ctx, device, stage, probe))
        by_stage[stage] = results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── per-prompt comparison ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PER-PROMPT COMPARISON (temp=0, max_new=80, post-EOS truncated)")
    print("=" * 100)
    for probe in PROBES:
        print(f"\n* [{probe.name}] {probe.instruction}")
        if probe.must_contain:
            print(f"  expected substring (any of): {probe.must_contain}")
        for stage, _ in PIPELINE:
            r = next(x for x in by_stage[stage] if x.probe == probe.name)
            flag = "OK" if (not probe.must_contain or r.contains_expected) else "--"
            eos_flag = "EOS" if r.eos_emitted else "—  "
            print(f"  {stage:<8} [{eos_flag} @ {r.tokens_to_eos:>2}t, "
                  f"rep4={r.rep4:.2f}] {flag} {r.text!r}")

    # ── per-stage aggregate ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STAGE AGGREGATES")
    print("=" * 100)
    print(f"{'stage':<10} {'n':>3}  {'EOS rate':>10}  {'avg tok/EOS':>12}  "
          f"{'avg rep4':>10}  {'fact-hit':>10}")
    for stage, _ in PIPELINE:
        s = _stage_summary(by_stage[stage])
        ch = (f"{s['contains_expected_rate']:.0%}"
              if not _isnan(s['contains_expected_rate']) else "n/a")
        print(f"{stage:<10} {s['n']:>3}  {s['eos_rate']:>9.0%}  "
              f"{s['avg_tokens_to_eos']:>12.1f}  {s['avg_rep4']:>10.2f}  "
              f"{ch:>10}")

    # ── chain assertions ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("CHAIN ASSERTIONS (diagnostic, non-fatal)")
    print("=" * 100)
    pre, sft, dpo = (_stage_summary(by_stage[s]) for s in ("pretrain", "sft", "dpo"))
    _check("SFT terminates on EOS more often than pretrain on Alpaca prompts",
           sft["eos_rate"] > pre["eos_rate"])
    _check("SFT generates shorter responses than pretrain (learned to stop)",
           sft["avg_tokens_to_eos"] < pre["avg_tokens_to_eos"])
    _check("DPO inherits SFT's EOS termination",
           dpo["eos_rate"] >= sft["eos_rate"] - 0.1)
    _check("DPO inherits SFT's repetition profile (no major regression)",
           dpo["avg_rep4"] <= sft["avg_rep4"] + 0.1)

    return 0


def _isnan(x: float) -> bool:
    return x != x


def _check(label: str, ok: bool) -> None:
    mark = "PASS" if ok else "WARN"
    print(f"  [{mark}] {label}")


if __name__ == "__main__":
    raise SystemExit(main())
