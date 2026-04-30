"""Generate sampled outputs from an SFT checkpoint for visual quality review.

Usage:
  uv run python scripts/sft_inspect_samples.py \
      --checkpoint runs/posttraining/sft/<run>/checkpoints/best_*.pt \
      --suite configs/posttraining/dev_prompt_suite_ifeval_lite.jsonl \
      --output runs/posttraining/sft/<run>/inspection.md

Prints a markdown table with prompt | greedy answer | sampled answer | gold.
Reads architecture from the checkpoint (does not need a config).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so 'from src...' imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.eval.inference import generate
from src.model import ParrotLLM
from src.posttraining.templates import build_generation_prompt, strip_generated_assistant_text
from src.utils import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-prompt", type=str, default="You are ParrotLLM, a helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ParrotLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ctx_len = int(cfg["model"]["context_length"])

    tokenizer = build_tokenizer()

    cases = []
    with args.suite.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    rows = ["| # | prompt | greedy | sample (T=0.7) | gold |", "|---|---|---|---|---|"]
    for i, case in enumerate(cases):
        msgs = case.get("messages") or [{"role": "user", "content": case.get("prompt", "")}]
        prompt = build_generation_prompt(msgs, system_prompt=args.system_prompt)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        max_t = int(case.get("max_tokens", args.max_tokens))

        out_g = generate(model, ids, max_t, temperature=0.0, top_k=0, top_p=1.0,
                         context_length=ctx_len, eos_token_id=tokenizer.eos_token_id)
        out_s = generate(model, ids, max_t, temperature=0.7, top_k=50, top_p=0.9,
                         context_length=ctx_len, eos_token_id=tokenizer.eos_token_id)

        def _decode(o):
            text = tokenizer.decode(o[0, ids.shape[1]:].tolist())
            return strip_generated_assistant_text(text).replace("\n", " ⏎ ").strip()[:120]

        user_text = msgs[-1]["content"][:80].replace("\n", " ⏎ ")
        rows.append(
            f"| {i:02d} | {user_text} | {_decode(out_g)} | {_decode(out_s)} | "
            f"{case.get('gold','')} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
