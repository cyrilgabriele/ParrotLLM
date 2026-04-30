"""SFT-stage scorecard — only the four metrics that map to VL07 SFT objectives.

The course is explicit about what SFT is responsible for and what it is not
(slide 12: follow instructions / answer in assistant format; slide 24:
quality / refusal / calibration are RLHF-DPO's job; slide 25: do not destroy
pretraining knowledge). This script measures *only* the SFT-scope behaviors,
modeled on IFEval (Zhou et al. 2023, arXiv:2311.07911) for instruction
following.

Four metrics + one composite:
  1. IFEval-strict pass rate    — slide 12 "follow instructions"
  2. Template-bleed rate         — slide 12 "answer in assistant format"
  3. EOS emission rate           — slide 12 "answer in assistant format" (clear end)
  4. Replay perplexity           — slide 25 "no catastrophic forgetting"

Everything else (distinct-N, length stats, per-format breakdowns, LM-judge
quality scoring) is deliberately omitted because it either crosses into
DPO scope (slide 24) or doesn't map to a stated SFT objective.

Usage:
    uv run python scripts/sft_scorecard.py \\
        --checkpoint <ckpt.pt> [--checkpoint <ckpt2.pt> ...] \\
        --baseline <base_ckpt.pt> \\
        --suite configs/posttraining/dev_prompt_suite_ifeval_lite.jsonl \\
        [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import math
import string
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.eval.inference import generate  # noqa: E402
from src.model import ParrotLLM  # noqa: E402
from src.posttraining.eval import DEFAULT_FORBIDDEN_SUBSTRINGS  # noqa: E402
from src.posttraining.templates import (  # noqa: E402
    build_generation_prompt,
    strip_generated_assistant_text,
)
from src.utils import build_tokenizer  # noqa: E402


# Replay perplexity reference for our base pretraining checkpoint
# (best_loss_3p2650_..._step_0095500.pt, measured on data/processed/val.bin).
# Used as the anchor for the forgetting penalty when no --baseline is provided.
BASE_REPLAY_PPL_DEFAULT = 26.5


# ── IFEval-strict scoring ─────────────────────────────────────────────────

def _normalize_exact(text: str) -> str:
    return text.strip().strip(string.whitespace + string.punctuation).lower()


def _all_constraints_pass(case: dict, response: str, raw: str) -> bool:
    """IFEval-strict: every constraint declared by the case must hold.

    Mirrors the rules in src.posttraining.eval._score_case but returns a
    single boolean (passed all) rather than a multiplied float."""
    normalized = response.strip()

    # 1. No template bleed in the cleaned response
    forbidden = list(DEFAULT_FORBIDDEN_SUBSTRINGS) + [
        str(s) for s in (case.get("forbidden_contains") or [])
    ]
    if any(s and s in normalized for s in forbidden):
        return False

    # 2. No prompt echo (only when case opts in)
    if case.get("forbid_prompt_echo", False):
        prompt_text = " ".join(
            str(m.get("content", ""))
            for m in case.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "user"
        ).strip()
        if prompt_text and prompt_text.lower() in raw.lower():
            return False

    # 3. Format-specific compliance
    fmt = str(case.get("expected_format", "freeform")).strip().lower()
    if fmt == "json":
        try:
            json.loads(normalized)
        except json.JSONDecodeError:
            return False
    elif fmt == "mcq_letter":
        if normalized not in {"A", "B", "C", "D"}:
            return False
    elif fmt == "refusal":
        markers = (
            "cannot help",
            "can't help",
            "won't help",
            "cannot assist",
            "can't assist",
        )
        if not any(m in normalized.lower() for m in markers):
            return False
    if fmt == "short_answer" and "max_words" in case:
        if len(normalized.split()) > int(case["max_words"]):
            return False

    # 4. Gold answer (exact match after normalization)
    gold = case.get("gold")
    if gold is not None and _normalize_exact(normalized) != _normalize_exact(str(gold)):
        return False

    # 5. Must contain (every needle)
    for needle in case.get("must_contain") or []:
        if str(needle) not in normalized:
            return False

    return True


# ── Per-checkpoint evaluation ─────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: Path,
    cases: list[dict],
    *,
    device: torch.device,
    tokenizer,
    max_tokens: int,
    system_prompt: str,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ParrotLLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ctx_len = int(ckpt["config"]["model"]["context_length"])
    eos_id = tokenizer.eos_token_id

    n = len(cases)
    strict_pass = 0
    eos_emitted = 0
    bleed_in_raw = 0

    for case in cases:
        msgs = case.get("messages") or [{"role": "user", "content": case.get("prompt", "")}]
        prompt_text = build_generation_prompt(msgs, system_prompt=system_prompt)
        ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        max_t = int(case.get("max_tokens", max_tokens))

        out = generate(
            model, ids, max_t,
            temperature=0.0, top_k=0, top_p=1.0,
            context_length=ctx_len, eos_token_id=eos_id,
        )
        new = out[0, ids.shape[1]:].tolist()

        if eos_id in new:
            eos_emitted += 1
            new_no_eos = new[:new.index(eos_id)]
        else:
            new_no_eos = new
        raw = tokenizer.decode(new_no_eos)
        response = strip_generated_assistant_text(raw)

        if any(s in raw for s in DEFAULT_FORBIDDEN_SUBSTRINGS):
            bleed_in_raw += 1
        if _all_constraints_pass(case, response, raw):
            strict_pass += 1

    return {
        "n_prompts": n,
        "ifeval_strict": strict_pass / max(1, n),
        "template_bleed": bleed_in_raw / max(1, n),
        "eos_rate": eos_emitted / max(1, n),
    }


def replay_ppl_from_run(ckpt_path: Path) -> float | None:
    """Pull the best replay_ppl from this checkpoint's parent run.

    SFT runs evaluate replay_ppl on data/processed/val.bin (raw OWT) at every
    eval point. We use the minimum value seen as the run's representative
    forgetting score."""
    run_dir = ckpt_path.parent.parent
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    best = None
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") != "eval" and rec.get("event") != "eval":
            continue
        v = rec.get("replay_ppl")
        if isinstance(v, (int, float)) and math.isfinite(v):
            if best is None or v < best:
                best = float(v)
    return best


def composite_score(metrics: dict, base_replay_ppl: float) -> float:
    """Single number for ranking, with weights drawn from VL07.

    SFT objectives in slide 12 are co-equal: instruction-following AND
    assistant format. Slide 25 makes catastrophic forgetting a constraint
    rather than a goal, so it appears as a penalty.

      0.5 · IFEval-strict
    + 0.5 · format_score          where format_score = 0.5·eos_rate + 0.5·(1−template_bleed)
    − 0.5 · forgetting_penalty    where forgetting_penalty = max(0, (replay_ppl − base) / base)
    """
    fmt = 0.5 * metrics["eos_rate"] + 0.5 * (1.0 - metrics["template_bleed"])
    rp = metrics.get("replay_ppl")
    if isinstance(rp, (int, float)) and math.isfinite(rp) and base_replay_ppl > 0:
        forgetting = max(0.0, (rp - base_replay_ppl) / base_replay_ppl)
    else:
        forgetting = 0.0
    return 0.5 * metrics["ifeval_strict"] + 0.5 * fmt - 0.5 * forgetting


# ── Markdown rendering ────────────────────────────────────────────────────

def _label(p: Path) -> str:
    return p.parent.parent.name


def render_markdown(reports: list[tuple[str, dict]]) -> str:
    lines = ["# SFT scorecard\n"]
    cols = ["metric"] + [lbl for lbl, _ in reports]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    def row(name: str, fmt: str, key: str):
        cells = [name]
        for _, m in reports:
            v = m.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("n/a")
            else:
                cells.append(fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")

    row("IFEval-strict pass rate (slide 12: follow instructions)", "{:.1%}", "ifeval_strict")
    row("Template-bleed rate (slide 12: assistant format — lower is better)", "{:.1%}", "template_bleed")
    row("EOS rate (slide 12: assistant format — higher is better)", "{:.1%}", "eos_rate")
    row("Replay perplexity (slide 25: forgetting; base ≈ 26.5)", "{:.2f}", "replay_ppl")
    lines.append("|  |" + "|" * len(reports))
    row("**SFT composite** (0.5·instr + 0.5·format − 0.5·forget)", "**{:.4f}**", "composite")

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Checkpoint to score (repeatable for side-by-side).")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional baseline checkpoint; included as the first column.")
    parser.add_argument("--suite", type=Path,
                        default=PROJECT_ROOT / "configs/posttraining/dev_prompt_suite_ifeval_lite.jsonl")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--system-prompt", type=str,
                        default="You are ParrotLLM, a helpful assistant.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    cases = []
    with args.suite.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    tokenizer = build_tokenizer()

    targets: list[Path] = []
    if args.baseline is not None:
        targets.append(args.baseline)
    targets += list(args.checkpoint)

    reports: list[tuple[str, dict]] = []
    seen: set[Path] = set()
    base_replay_ref = BASE_REPLAY_PPL_DEFAULT
    for ckpt_path in targets:
        if ckpt_path in seen:
            continue
        seen.add(ckpt_path)
        print(f"Scoring {ckpt_path} ...", flush=True, file=sys.stderr)
        m = evaluate_checkpoint(
            ckpt_path, cases,
            device=device, tokenizer=tokenizer,
            max_tokens=args.max_tokens, system_prompt=args.system_prompt,
        )
        rp = replay_ppl_from_run(ckpt_path)
        m["replay_ppl"] = rp if rp is not None else float("nan")
        if ckpt_path == args.baseline and rp is not None:
            base_replay_ref = rp
        m["composite"] = composite_score(m, base_replay_ref)
        reports.append((_label(ckpt_path), m))

    md = render_markdown(reports)
    print()
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
