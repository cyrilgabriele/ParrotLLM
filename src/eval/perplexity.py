"""Evaluate perplexity on Wikitext-103 and OWT validation split."""

import math

import numpy as np
import torch
from transformers import AutoTokenizer

from src.model import ParrotLLM
from src.utils import get_device, load_config, set_seed


def compute_perplexity(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    context_length: int,
    device: torch.device,
    batch_size: int = 32,
    max_sequences: int = 512,
) -> float:
    model.eval()
    n_tokens = len(token_ids)
    stride = context_length
    losses = []

    starts = list(range(0, n_tokens - context_length, stride))[:max_sequences]

    for batch_start in range(0, len(starts), batch_size):
        batch_offsets = starts[batch_start : batch_start + batch_size]
        xs, ys = [], []
        for s in batch_offsets:
            chunk = token_ids[s : s + context_length + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        with torch.no_grad():
            _, loss = model(x, targets=y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def eval_wikitext(model, config, device):
    from datasets import load_dataset

    mc = config["model"]
    ec = config.get("eval", {})
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    ppl = compute_perplexity(
        model, token_ids, mc["context_length"], device,
        ec.get("batch_size", 32), ec.get("max_sequences", 512),
    )
    return ppl


def eval_owt_val(model, config, device):
    mc = config["model"]
    ec = config.get("eval", {})
    val_path = ec.get("datasets", [{}])[-1].get("path", "data/val.bin")

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    token_ids = torch.from_numpy(data.astype(np.int64))

    ppl = compute_perplexity(
        model, token_ids, mc["context_length"], device,
        ec.get("batch_size", 32), ec.get("max_sequences", 512),
    )
    return ppl


def run_eval(args) -> None:
    config = load_config(args.config)
    ec = config.get("eval", {})
    set_seed(config["training"]["seed"])
    device = get_device(ec.get("device", args.device))

    assert args.checkpoint, "--checkpoint required for eval"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)

    model = ParrotLLM(ckpt_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded checkpoint: {args.checkpoint}")
    print(f"[eval] parameters: {model.count_parameters():,}")

    # Wikitext-103
    try:
        wt_ppl = eval_wikitext(model, ckpt_config, device)
        print(f"[eval] Wikitext-103 perplexity: {wt_ppl:.2f}")
    except Exception as e:
        print(f"[eval] Wikitext-103 skipped: {e}")

    # OWT val
    try:
        owt_ppl = eval_owt_val(model, ckpt_config, device)
        print(f"[eval] OWT val perplexity: {owt_ppl:.2f}")
    except Exception as e:
        print(f"[eval] OWT val skipped: {e}")
