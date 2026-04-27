"""Perplexity sweep on Wikitext-103 + OpenWebText val for all three
checkpoints in the chain. Pillar 1 of the rubric (factsheet §4.3).

Run: uv run python -m tools.perplexity_sweep
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from src.eval.perplexity import compute_perplexity
from src.scripts.sft_benchmark import _load
from src.utils import build_tokenizer, get_device


PIPELINE = [
    ("pretrain",
     r"runs/big_run/exp_c/run_20260408_124138/checkpoints/best_loss_3p5437_epoch_0001_step_0003000.pt"),
    ("sft",
     r"runs/run_20260426_203420_sft/checkpoints/best_step_0001500_epoch_01_valloss_2p7198.pt"),
    ("dpo",
     r"runs/run_20260426_210502_dpo/checkpoints/best_step_0000200_epoch_00_valloss_0p0368.pt"),
]


def main() -> int:
    device = get_device("auto")
    tok = build_tokenizer()
    print(f"Device: {device}")

    # Pre-tokenize once: shared across all three checkpoints.
    print("Tokenizing Wikitext-103 test ...", flush=True)
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    wt_text = "\n\n".join(wt["text"])
    wt_ids = torch.tensor(tok.encode(wt_text), dtype=torch.long)
    print(f"  Wikitext-103 test: {len(wt_ids):,} tokens")

    owt_path = Path("data/processed/val.bin")
    if owt_path.exists():
        owt_ids = torch.from_numpy(np.memmap(owt_path, dtype=np.uint16, mode="r").astype(np.int64))
        print(f"  OWT val: {len(owt_ids):,} tokens")
    else:
        owt_ids = None
        print(f"  OWT val: file missing at {owt_path} — skipping")

    rows: dict[str, dict[str, float]] = {}
    for stage, ckpt in PIPELINE:
        print(f"\n=== {stage} ← {ckpt}")
        model, mc = _load(ckpt, device)
        ctx = int(mc["context_length"])

        t0 = time.time()
        wt_ppl = compute_perplexity(model, wt_ids, ctx, device,
                                    batch_size=8, max_sequences=512)
        print(f"  Wikitext-103 PPL: {wt_ppl:8.2f}   ({time.time()-t0:5.1f}s)")
        rows.setdefault(stage, {})["wikitext103"] = wt_ppl

        if owt_ids is not None:
            t0 = time.time()
            owt_ppl = compute_perplexity(model, owt_ids, ctx, device,
                                         batch_size=8, max_sequences=512)
            print(f"  OWT val      PPL: {owt_ppl:8.2f}   ({time.time()-t0:5.1f}s)")
            rows[stage]["owt_val"] = owt_ppl

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PERPLEXITY (lower = better) — Pillar 1 of factsheet §4.3")
    print("=" * 78)
    print(f"{'dataset':<14} " + "  ".join(f"{s:>10}" for s, _ in PIPELINE)
          + "    Δ(DPO-pre)")
    print("-" * 78)
    datasets = ["wikitext103", "owt_val"]
    for d in datasets:
        if all(d in rows[s] for s, _ in PIPELINE):
            pre = rows["pretrain"][d]
            sft = rows["sft"][d]
            dpo = rows["dpo"][d]
            delta = dpo - pre
            sign = "+" if delta >= 0 else ""
            print(f"{d:<14}   {pre:8.2f}     {sft:8.2f}     {dpo:8.2f}     {sign}{delta:7.2f}")

    out = Path("runs/perplexity_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoints": dict(PIPELINE),
        "perplexity": rows,
    }, indent=2))
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
