"""Run LAMBADA / HellaSwag / WinoGrande / OpenBookQA against the full
pretrain → SFT → DPO chain and save a comparison table.

This is the local sanity-check version of the official PikoGPT
leaderboard runner (factsheet §4.4). Use it to decide which checkpoint
to submit and to populate the tech-report comparison table required by
the rubric. Each benchmark is scored by candidate-NLL ranking, mirroring
the leaderboard's likelihood-based grading style.

Run: uv run python -m tools.benchmark_pipeline [--n 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.scripts.sft_benchmark import (
    SCORERS, _load,
)
from src.utils import build_tokenizer, get_device


PIPELINE = [
    ("pretrain",
     r"runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt"),
    ("sft",
     r"runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt"),
    ("dpo",
     r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt"),
]

BENCHMARKS = ["lambada", "hellaswag", "winogrande", "openbookqa"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="examples per benchmark (default 200)")
    parser.add_argument("--out", type=str,
                        default="runs/leaderboard_comparison.json")
    args = parser.parse_args()

    device = get_device("auto")
    tok = build_tokenizer()
    print(f"Device: {device}, n_per_benchmark={args.n}\n")

    rows: dict[str, dict[str, float]] = {}
    timings: dict[str, dict[str, float]] = {}

    for stage, ckpt in PIPELINE:
        print(f"=== {stage} ← {ckpt}")
        model, mc = _load(ckpt, device)
        ctx = int(mc["context_length"])
        rows[stage] = {}
        timings[stage] = {}

        for bm in BENCHMARKS:
            t0 = time.time()
            print(f"  scoring {bm} ...", flush=True)
            r = SCORERS[bm](model, tok, args.n, ctx_len=ctx, device=device)
            dt = time.time() - t0
            rows[stage][bm] = r["accuracy"]
            timings[stage][bm] = dt
            print(f"    {r['benchmark']:<11} acc={r['accuracy']*100:6.2f}% "
                  f"(n={r['n']}, {dt:5.1f}s)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    # ── Summary table ────────────────────────────────────────────────
    print("=" * 78)
    print("LEADERBOARD ACCURACY COMPARISON (% correct)")
    print("=" * 78)
    print(f"{'benchmark':<12} " + "  ".join(f"{s:>10}" for s, _ in PIPELINE)
          + "    delta(DPO-pre)  delta(DPO-SFT)")
    print("-" * 78)
    for bm in BENCHMARKS:
        pre = rows["pretrain"][bm]
        sft = rows["sft"][bm]
        dpo = rows["dpo"][bm]
        d_pre = (dpo - pre) * 100
        d_sft = (dpo - sft) * 100
        line = (f"{bm:<12} "
                + f"  {pre*100:8.2f}% "
                + f"  {sft*100:8.2f}% "
                + f"  {dpo*100:8.2f}% "
                + f"     {d_pre:+6.2f}pp"
                + f"        {d_sft:+6.2f}pp")
        print(line)

    # Mean across benchmarks
    print("-" * 78)
    means = {s: sum(rows[s][b] for b in BENCHMARKS) / len(BENCHMARKS)
             for s, _ in PIPELINE}
    print(f"{'mean':<12} "
          + f"  {means['pretrain']*100:8.2f}% "
          + f"  {means['sft']*100:8.2f}% "
          + f"  {means['dpo']*100:8.2f}% "
          + f"     {(means['dpo']-means['pretrain'])*100:+6.2f}pp"
          + f"        {(means['dpo']-means['sft'])*100:+6.2f}pp")

    # ── Save JSON ────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_per_benchmark": args.n,
        "checkpoints": dict(PIPELINE),
        "accuracy": rows,
        "timing_seconds": timings,
        "mean_accuracy": means,
    }, indent=2))
    print(f"\nSaved → {out_path}")

    # ── Verdict per the lecture rubric ──────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT (factsheet §4.3 — leaderboard grades on these benchmarks)")
    print("=" * 78)
    best_stage = max(means, key=lambda s: means[s])
    print(f"  Best mean accuracy: {best_stage} ({means[best_stage]*100:.2f}%)")
    if best_stage == "pretrain":
        print(f"  → Submit PRETRAIN to the leaderboard. SFT/DPO degraded benchmarks")
        print(f"    (catastrophic forgetting, VL07 slide 25). Use DPO only for the")
        print(f"    chat demo since chat is graded on usability, not accuracy.")
    elif best_stage == "sft":
        print(f"  → Submit SFT to the leaderboard. DPO regressed on benchmarks.")
        print(f"    Likely too aggressive a beta or overfit on preferences.")
    else:
        print(f"  → Submit DPO to the leaderboard. The full chain is your best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
