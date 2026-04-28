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
    ("pre_500M",
     r"runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt"),
    ("pre_8B",
     r"runs/big_run/exp_c_8b/run_20260410_044337/checkpoints/best_loss_3p2650_epoch_0000_step_0095500.pt"),
    ("sft_v3",
     r"runs/run_20260427_214613_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7318.pt"),
    ("sft_v5_8B",
     r"runs/run_20260427_230323_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p4115.pt"),
    ("dpo_v4",
     r"runs/run_20260427_215602_dpo/checkpoints/best_step_0000360_epoch_00_valloss_0p6640.pt"),
    ("dpo_v5_8B",
     r"runs/run_20260427_231456_dpo/checkpoints/best_step_0000360_epoch_00_valloss_0p6449.pt"),
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
    stages = [s for s, _ in PIPELINE]
    print("=" * 90)
    print("LEADERBOARD ACCURACY COMPARISON (% correct)")
    print("=" * 90)
    print(f"{'benchmark':<12} " + "  ".join(f"{s:>10}" for s in stages))
    print("-" * 90)
    for bm in BENCHMARKS:
        cells = "  ".join(f"{rows[s][bm]*100:8.2f}%" for s in stages)
        print(f"{bm:<12}  {cells}")

    print("-" * 90)
    means = {s: sum(rows[s][b] for b in BENCHMARKS) / len(BENCHMARKS) for s in stages}
    cells = "  ".join(f"{means[s]*100:8.2f}%" for s in stages)
    print(f"{'mean':<12}  {cells}")

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
        print(f"    (catastrophic forgetting, VL07 slide 25). Use best DPO for chat demo.")
    elif best_stage == "sft":
        print(f"  → Submit SFT. DPO did not improve benchmarks beyond noise.")
    else:
        print(f"  → Submit {best_stage}. The full chain is your best on benchmarks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
