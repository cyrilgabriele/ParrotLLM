"""In-process benchmark harness for ParrotLLM leaderboard checkpoints.

Replaces ``external/PikoGPT_Leaderboard/leaderboard/run_benchmarks.py``'s
subprocess-per-example pattern with a single Python process that loads the
model + tokenizer once and iterates inference in-loop. Uses the SAME inference
helpers as the leaderboard submission (cloze MC scoring, LAMBADA continuation
generation), so when PMI is OFF the per-example output exactly matches the
subprocess runner. With PMI ON (default for `--leaderboard`-style scoring),
per-option scoring uses the unconditional-baseline correction implemented in
``Submissions/parrotlabs_parrotllm/src/inference.py``.

This harness is for fast local iteration only — the leaderboard contract
remains the subprocess runner. Do NOT modify run_benchmarks.py from here.

Output: a JSON file per --bench (or one merged file when --bench=all) with
``{benchmark, total, correct, invalid, accuracy_pct, wall_time_s}``.

CLI:
    uv run python tools/run_bench_inproc.py \
        --checkpoint <path> \
        --bench {all|hellaswag|winogrande|openbookqa|lambada} \
        --limit N \
        --output runs/.../<label>.json \
        [--device {auto|mps|cpu|cuda}] [--seed 0] [--pmi {auto,on,off}]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "Submissions" / "parrotlabs_parrotllm"
BENCH_ROOT = PROJECT_ROOT / "external" / "PikoGPT_Leaderboard" / "leaderboard" / "benchmarks"

DEFAULT_BENCH_PATHS = {
    "hellaswag": BENCH_ROOT / "hellaswag" / "cleaned" / "validation.jsonl",
    "winogrande": BENCH_ROOT / "winogrande" / "cleaned" / "validation.jsonl",
    "openbookqa": BENCH_ROOT / "openbookqa" / "cleaned" / "validation.jsonl",
    "lambada": BENCH_ROOT / "lambada" / "cleaned" / "test.jsonl",
}

# The same allowed-letter sets as run_benchmarks.py for parsing fallbacks.
MC_ALLOWED_LETTERS = {
    "hellaswag": {"A", "B", "C", "D"},
    "winogrande": {"A", "B"},
    "openbookqa": {"A", "B", "C", "D"},
}


# ── Submission module loader ─────────────────────────────────────────────────
def _load_submission_modules():
    """Import the submission's ``src.inference`` and ``main`` modules.

    The submission's ``main.py`` does ``from src.inference import ...``. If the
    project root's ``src/`` package was already imported during this Python
    process (e.g. by tests), Python caches that ``src`` and the submission's
    import would fail. Pre-register the submission's ``src.inference`` under
    the name ``src.inference`` so the import resolves to the submission file.
    """
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))

    inference_path = SUBMISSION_DIR / "src" / "inference.py"
    if "src.inference" not in sys.modules:
        inf_spec = importlib.util.spec_from_file_location("src.inference", inference_path)
        inf_module = importlib.util.module_from_spec(inf_spec)
        sys.modules["src.inference"] = inf_module
        inf_spec.loader.exec_module(inf_module)
    inference_mod = sys.modules["src.inference"]

    main_path = SUBMISSION_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("submission_main_inproc", main_path)
    main_mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_main_inproc"] = main_mod
    spec.loader.exec_module(main_mod)
    return inference_mod, main_mod


# ── IO ───────────────────────────────────────────────────────────────────────
def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ── Parsing helpers (must match run_benchmarks.py exactly) ───────────────────
def parse_mc_letter(gen: str, allowed: set[str]) -> Optional[str]:
    s = gen.lstrip()
    if not s:
        return None
    ch = s[0].upper()
    return ch if ch in allowed else None


def normalize_lambada(s: str) -> str:
    s = s.strip()
    s = s.strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    return s.lower()


def parse_lambada_word(gen: str) -> str:
    s = gen.lstrip()
    if not s:
        return ""
    word = re.split(r"\s+", s, maxsplit=1)[0]
    return normalize_lambada(word)


# ── Per-example inference (mirrors main.py --leaderboard, in-process) ────────
@torch.no_grad()
def run_example(
    *,
    raw_prompt: str,
    bench: str,
    model,
    tokenizer,
    device: torch.device,
    context_length: int,
    eos_id: Optional[int],
    inference_mod,
    main_mod,
    pmi_enabled: bool,
    mc_max_tokens: int,
    lambada_max_tokens: int,
) -> str:
    """Return the same STDOUT string the subprocess submission would emit
    in ``--leaderboard`` mode for ``raw_prompt``.

    The flow mirrors ``main.main`` line-for-line:
      1. ``render_prompt_for_inference(..., leaderboard=True)``.
      2. If MC: ``_score_mc(...)`` → letter; on exception, constrained-letter
         argmax fallback.
      3. Else (LAMBADA / chat): tokenized greedy decode with KV cache, EOS-trim.
    """
    rendered = main_mod.render_prompt_for_inference(
        raw_prompt=raw_prompt,
        template="alpaca",  # matches the submission default
        system_prompt=main_mod.DEFAULT_SYSTEM_PROMPT,
        leaderboard=True,
    )

    if rendered.kind == "mc":
        try:
            best_idx = main_mod._score_mc(
                model=model,
                tokenizer=tokenizer,
                rendered=rendered,
                device=device,
                context_length=context_length,
                pmi=pmi_enabled,
            )
            return chr(ord("A") + best_idx)
        except Exception:
            n_opts = max(2, len(rendered.mc_options))
            allowed = inference_mod.letter_token_ids(
                tokenizer, [chr(ord("A") + k) for k in range(n_opts)]
            )
            input_ids = tokenizer.encode(rendered.text, add_special_tokens=False)
            if not input_ids:
                input_ids = [int(eos_id) if eos_id is not None else 0]
            idx_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            out = main_mod.generate(
                model,
                idx_t,
                max_new_tokens=1,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                context_length=context_length,
                allowed_first_token_ids=allowed,
            )
            generated = tokenizer.decode(
                out[0, len(input_ids):].tolist(),
                clean_up_tokenization_spaces=False,
            ).lstrip()
            return generated[:1].upper() if generated else "A"

    # LAMBADA / chat path
    input_ids = tokenizer.encode(rendered.text, add_special_tokens=False)
    if not input_ids:
        if eos_id is None:
            return ""
        input_ids = [int(eos_id)]
    idx_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    max_new = lambada_max_tokens if rendered.kind == "lambada" else mc_max_tokens
    output = main_mod.generate(
        model,
        idx_t,
        max_new_tokens=max(0, int(max_new)),
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        context_length=context_length,
        eos_token_id=int(eos_id) if eos_id is not None else None,
    )
    generated_ids = output[0, len(input_ids):].tolist()
    if eos_id is not None and generated_ids and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)


# ── Per-bench evaluators ─────────────────────────────────────────────────────
def eval_mc_bench(
    *,
    bench: str,
    data_path: Path,
    limit: Optional[int],
    model,
    tokenizer,
    device,
    context_length,
    eos_id,
    inference_mod,
    main_mod,
    pmi_enabled: bool,
    mc_max_tokens: int,
    lambada_max_tokens: int,
    verbose: bool = False,
) -> dict:
    allowed = MC_ALLOWED_LETTERS[bench]
    total = correct = invalid = 0
    t0 = time.perf_counter()
    for i, ex in enumerate(read_jsonl(data_path)):
        if limit is not None and i >= limit:
            break
        prompt = ex["prompt"]
        gold = str(ex["answer_letter"]).upper()
        try:
            gen = run_example(
                raw_prompt=prompt,
                bench=bench,
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_length=context_length,
                eos_id=eos_id,
                inference_mod=inference_mod,
                main_mod=main_mod,
                pmi_enabled=pmi_enabled,
                mc_max_tokens=mc_max_tokens,
                lambada_max_tokens=lambada_max_tokens,
            )
        except Exception as e:
            total += 1
            invalid += 1
            if verbose:
                print(f"[{bench} {i}] ERROR: {e!r}")
            continue
        pred = parse_mc_letter(gen, allowed)
        total += 1
        if pred is None:
            invalid += 1
        elif pred == gold:
            correct += 1
        if verbose:
            print(f"[{bench} {i}] gold={gold} pred={pred} raw={gen!r}")
    wall = time.perf_counter() - t0
    acc = (correct / total * 100.0) if total else 0.0
    return {
        "benchmark": bench,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy_pct": acc,
        "wall_time_s": wall,
    }


def eval_lambada_bench(
    *,
    data_path: Path,
    limit: Optional[int],
    model,
    tokenizer,
    device,
    context_length,
    eos_id,
    inference_mod,
    main_mod,
    pmi_enabled: bool,
    mc_max_tokens: int,
    lambada_max_tokens: int,
    verbose: bool = False,
) -> dict:
    total = correct = invalid = 0
    t0 = time.perf_counter()
    for i, ex in enumerate(read_jsonl(data_path)):
        if limit is not None and i >= limit:
            break
        prompt = ex["prompt"]
        gold = ex["answer_text"]
        gold_n = normalize_lambada(gold)
        try:
            gen = run_example(
                raw_prompt=prompt,
                bench="lambada",
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_length=context_length,
                eos_id=eos_id,
                inference_mod=inference_mod,
                main_mod=main_mod,
                pmi_enabled=pmi_enabled,
                mc_max_tokens=mc_max_tokens,
                lambada_max_tokens=lambada_max_tokens,
            )
        except Exception as e:
            total += 1
            invalid += 1
            if verbose:
                print(f"[lambada {i}] ERROR: {e!r}")
            continue
        pred_n = parse_lambada_word(gen)
        total += 1
        if pred_n == gold_n:
            correct += 1
        if verbose:
            print(f"[lambada {i}] gold={gold_n} pred={pred_n} raw={gen!r}")
    wall = time.perf_counter() - t0
    acc = (correct / total * 100.0) if total else 0.0
    return {
        "benchmark": "lambada",
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy_pct": acc,
        "wall_time_s": wall,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
PUBLIC_BENCHES = ("hellaswag", "winogrande", "openbookqa", "lambada")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="In-process leaderboard benchmark harness.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--bench",
        nargs="+",
        choices=("all", *PUBLIC_BENCHES),
        default=["all"],
        help="One or more benchmark names, or 'all' for the four public benches.",
    )
    p.add_argument("--limit", type=int, default=None, help="Per-bench example cap.")
    p.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pmi",
        choices=("auto", "on", "off"),
        default="auto",
        help="PMI calibration for MC cloze. auto = OFF — matches the deployed "
        "--leaderboard inference path (PMI measured -2pp avg on our setup; "
        "see runs/overnight_sft_dpo_bench/summary.md, Round 3).",
    )
    p.add_argument("--mc-max-tokens", type=int, default=3)
    p.add_argument("--lambada-max-tokens", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_benches(bench_args: list[str]) -> list[str]:
    if "all" in bench_args:
        return list(PUBLIC_BENCHES)
    # preserve order, drop duplicates
    seen, out = set(), []
    for b in bench_args:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    inference_mod, main_mod = _load_submission_modules()

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = main_mod.get_device(args.device)
    print(f"[harness] device={device}  checkpoint={ckpt_path.name}")

    model, config = main_mod.load_model(ckpt_path, device)
    tokenizer = main_mod.load_tokenizer(SUBMISSION_DIR)
    context_length = int(config["model"].get("context_length", 1024))
    eos_id = config["model"].get("eos_token_id")
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    # auto -> off, matching the deployed --leaderboard inference path.
    pmi_enabled = (args.pmi == "on") if args.pmi != "auto" else False
    print(f"[harness] pmi={'on' if pmi_enabled else 'off'}  context_length={context_length}")

    benches = resolve_benches(args.bench)

    results: list[dict] = []
    overall_t0 = time.perf_counter()
    for bench in benches:
        data_path = DEFAULT_BENCH_PATHS[bench]
        if not data_path.is_file():
            raise FileNotFoundError(f"Benchmark data missing: {data_path}")
        print(f"\n--- {bench} ({data_path.name}, limit={args.limit}) ---")
        if bench == "lambada":
            r = eval_lambada_bench(
                data_path=data_path,
                limit=args.limit,
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_length=context_length,
                eos_id=eos_id,
                inference_mod=inference_mod,
                main_mod=main_mod,
                pmi_enabled=pmi_enabled,
                mc_max_tokens=args.mc_max_tokens,
                lambada_max_tokens=args.lambada_max_tokens,
                verbose=args.verbose,
            )
        else:
            r = eval_mc_bench(
                bench=bench,
                data_path=data_path,
                limit=args.limit,
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_length=context_length,
                eos_id=eos_id,
                inference_mod=inference_mod,
                main_mod=main_mod,
                pmi_enabled=pmi_enabled,
                mc_max_tokens=args.mc_max_tokens,
                lambada_max_tokens=args.lambada_max_tokens,
                verbose=args.verbose,
            )
        print(
            f"[{bench}] {r['correct']}/{r['total']} = {r['accuracy_pct']:.2f}% "
            f"(invalid={r['invalid']}, {r['wall_time_s']:.1f}s)"
        )
        results.append(r)
    overall_wall = time.perf_counter() - overall_t0

    payload = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "seed": int(args.seed),
        "pmi": "on" if pmi_enabled else "off",
        "limit": args.limit,
        "mc_max_tokens": int(args.mc_max_tokens),
        "lambada_max_tokens": int(args.lambada_max_tokens),
        "wall_time_s": overall_wall,
        "benchmarks": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[harness] wrote {args.output} (total wall {overall_wall:.1f}s)")


if __name__ == "__main__":
    main()
