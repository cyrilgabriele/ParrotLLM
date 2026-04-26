"""Local likelihood-based benchmark scorer.

Implements multiple-choice scoring on the four PikoGPT leaderboard
benchmarks (HellaSwag, WinoGrande, OpenBookQA, LAMBADA-style completion).
For each example, score every candidate by the model's negative log-
likelihood of the candidate tokens given the context, pick the lowest-NLL
candidate, compare to the gold label.

This is a sanity check — NOT a replacement for the official PikoGPT
Leaderboard runner. Use it to quickly compare base / SFT / DPO checkpoints
without round-tripping through the leaderboard repo.

Usage:
    uv run python -m src.scripts.sft_benchmark \\
        --checkpoint runs/run_20260426_202050_sft/checkpoints/best_step_0000600_epoch_00_valloss_2p7040.pt \\
        --benchmarks hellaswag winogrande openbookqa \\
        --n 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils import build_tokenizer, get_device


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _load(checkpoint_path: str, device: torch.device):
    from src.model import ParrotLLM
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if "model" not in cfg:
        cfg = {"model": cfg}
    model = ParrotLLM(cfg).to(device)
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model.eval()
    return model, cfg["model"]


@torch.no_grad()
def _score_candidate_nll(model, tokenizer, ctx: str, candidate: str,
                         *, ctx_len: int, device: torch.device) -> float:
    """Return the average NLL per candidate token, given the context.

    Uses the legacy `targets` forward path (next-token CE on raw text).
    Score lower = more probable continuation.
    """
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not cand_ids:
        return float("inf")

    full = ctx_ids + cand_ids
    if len(full) > ctx_len:
        # Drop from the LEFT of context; never truncate the candidate.
        overflow = len(full) - ctx_len
        ctx_ids = ctx_ids[overflow:]
        full = ctx_ids + cand_ids
    if len(full) < 2:
        return float("inf")

    idx = torch.tensor([full], dtype=torch.long, device=device)
    inputs = idx[:, :-1]
    targets = idx[:, 1:]
    logits, _ = model(inputs, return_logits=True)
    log_probs = F.log_softmax(logits, dim=-1)

    # Score only the candidate positions: targets at positions
    # [len(ctx_ids)-1 ... len(full)-2] correspond to predicting the
    # candidate tokens (which sit at full positions len(ctx_ids)..end).
    cand_start = len(ctx_ids) - 1
    cand_target_logp = log_probs[0, cand_start:, :].gather(
        -1, targets[0, cand_start:].unsqueeze(-1)
    ).squeeze(-1)
    nll = -cand_target_logp.mean().item()
    return nll


def score_hellaswag(model, tokenizer, n: int, *, ctx_len: int, device) -> dict:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0
    total = 0
    for i, row in enumerate(ds):
        if i >= n:
            break
        ctx = f"{row['ctx_a']} {row['ctx_b']}"
        endings = row["endings"]
        nlls = [_score_candidate_nll(model, tokenizer, ctx, " " + e,
                                     ctx_len=ctx_len, device=device) for e in endings]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        if pred == int(row["label"]):
            correct += 1
        total += 1
    return {"benchmark": "HellaSwag", "n": total, "accuracy": correct / max(1, total)}


def score_winogrande(model, tokenizer, n: int, *, ctx_len: int, device) -> dict:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                      trust_remote_code=True)
    correct = 0
    total = 0
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Sentence has '_' as the blank.
        sentence = row["sentence"]
        opt1, opt2 = row["option1"], row["option2"]
        # Score the FULL filled sentence under each option.
        s1 = sentence.replace("_", opt1)
        s2 = sentence.replace("_", opt2)
        nll1 = _score_candidate_nll(model, tokenizer, "", s1, ctx_len=ctx_len, device=device)
        nll2 = _score_candidate_nll(model, tokenizer, "", s2, ctx_len=ctx_len, device=device)
        pred = "1" if nll1 < nll2 else "2"
        if pred == row["answer"]:
            correct += 1
        total += 1
    return {"benchmark": "WinoGrande", "n": total, "accuracy": correct / max(1, total)}


def score_openbookqa(model, tokenizer, n: int, *, ctx_len: int, device) -> dict:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    correct = 0
    total = 0
    LABELS = ["A", "B", "C", "D"]
    for i, row in enumerate(ds):
        if i >= n:
            break
        stem = row["question_stem"].rstrip("?.").rstrip()
        choices = row["choices"]["text"]
        ctx = f"Question: {stem}?\nAnswer:"
        nlls = [_score_candidate_nll(model, tokenizer, ctx, " " + c,
                                     ctx_len=ctx_len, device=device) for c in choices]
        pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
        gold = LABELS.index(row["answerKey"])
        if pred == gold:
            correct += 1
        total += 1
    return {"benchmark": "OpenBookQA", "n": total, "accuracy": correct / max(1, total)}


def score_lambada(model, tokenizer, n: int, *, ctx_len: int, device) -> dict:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    correct = 0
    total = 0
    for i, row in enumerate(ds):
        if i >= n:
            break
        text = row["text"].strip()
        # Last word is the target. Greedy-decode 1 token from the prefix.
        words = text.split()
        if len(words) < 2:
            continue
        prefix = " ".join(words[:-1])
        gold_word = words[-1]
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        # Truncate to context len-1 so we can predict 1 more.
        if len(ids) > ctx_len - 1:
            ids = ids[-(ctx_len - 1):]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx, return_logits=True)
        # Greedy continue ~5 tokens, then check if the gold word is a prefix.
        gen_ids = []
        cur = idx
        for _ in range(8):
            with torch.no_grad():
                lg, _ = model(cur, return_logits=True)
            nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(int(nxt.item()))
            cur = torch.cat([cur, nxt], dim=1)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
        produced = tokenizer.decode(gen_ids).strip()
        if produced.split()[:1] == [gold_word]:
            correct += 1
        total += 1
    return {"benchmark": "LAMBADA", "n": total, "accuracy": correct / max(1, total)}


SCORERS = {
    "hellaswag": score_hellaswag,
    "winogrande": score_winogrande,
    "openbookqa": score_openbookqa,
    "lambada": score_lambada,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["hellaswag", "winogrande", "openbookqa"],
                        choices=list(SCORERS))
    parser.add_argument("--n", type=int, default=200,
                        help="examples per benchmark (default 200; small for speed)")
    args = parser.parse_args()

    device = get_device("auto")
    tokenizer = build_tokenizer()

    print(f"Loading {Path(args.checkpoint).name}")
    model, mc = _load(args.checkpoint, device)
    ctx_len = int(mc["context_length"])
    print(f"  context_length={ctx_len}, params={sum(p.numel() for p in model.parameters()):,}")
    print()

    results = []
    for name in args.benchmarks:
        print(f"Scoring {name} (n={args.n}) ...", flush=True)
        r = SCORERS[name](model, tokenizer, args.n, ctx_len=ctx_len, device=device)
        results.append(r)
        print(f"  {r['benchmark']}: {r['accuracy']*100:.2f}% (n={r['n']})")

    print()
    print(f"{'Benchmark':<14} {'Accuracy':<10} {'n':<5}")
    for r in results:
        print(f"{r['benchmark']:<14} {r['accuracy']*100:>7.2f}%   {r['n']:<5}")


if __name__ == "__main__":
    main()
