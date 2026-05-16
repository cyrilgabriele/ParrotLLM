"""Quick sanity-check: run both demo checkpoints on a battery of prompts
with the chat_demo.yaml sampling defaults. Prints Q/A pairs to stdout.

Use to validate sampling quality before the live demo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.inference import generate_stream, load_model_from_checkpoint  # noqa: E402
from src.post_training.sft import format_sft_prompt  # noqa: E402
from src.utils import build_tokenizer  # noqa: E402


PROMPTS = [
    # The 6 demo examples shown in the UI sidebar
    "Continue this story: The old lighthouse keeper walked down to the shore and",
    "Write a short poem about autumn rain.",
    "Write a friendly email opening to a new colleague.",
    "Summarize in one sentence: The cat sat lazily on the warm windowsill while the rain fell outside.",
    "Translate to German: Good morning, how are you?",
    "What is the capital of France?",
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/chat/chat_demo.yaml"


def run_checkpoint(label: str, ckpt_path: str, sampling: dict, prompts: list[str]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}\n{label}  ({ckpt_path})\n{'='*72}")
    model, cfg = load_model_from_checkpoint(ckpt_path, device)
    tokenizer = build_tokenizer()
    eos_id = tokenizer.eos_token_id
    ctx = cfg["model"]["context_length"]
    for q in prompts:
        prompt = format_sft_prompt(q)
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out_ids: list[int] = []
        t0 = time.perf_counter()
        for tok in generate_stream(
            model, idx, max_new_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_k=sampling["top_k"],
            top_p=sampling["top_p"],
            repetition_penalty=sampling["repetition_penalty"],
            no_repeat_ngram_size=sampling.get("no_repeat_ngram_size", 3),
            eos_token_id=eos_id,
            context_length=ctx,
        ):
            out_ids.append(tok)
            if tok == eos_id:
                break
        dt = time.perf_counter() - t0
        # Strip the SFT response template suffix if present
        answer = tokenizer.decode(out_ids).split("###")[0].rstrip()
        print(f"\nQ: {q}")
        print(f"A: {answer}")
        print(f"   ({len(out_ids)} tok in {dt:.2f}s = {len(out_ids)/dt:.1f} tok/s)")


def main() -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["chat"]
    sampling = {
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "repetition_penalty": cfg["repetition_penalty"],
        "no_repeat_ngram_size": 3,
    }
    print("Sampling config:")
    for k, v in sampling.items():
        print(f"  {k}: {v}")
    print(f"GPU available: {torch.cuda.is_available()}")

    torch.manual_seed(42)
    run_checkpoint("Cyril & Christof", "runs/demo/cyril_christof.pt", sampling, PROMPTS)
    torch.manual_seed(42)
    run_checkpoint("Gian & Tilman", "runs/demo/gian_tilman.pt", sampling, PROMPTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
