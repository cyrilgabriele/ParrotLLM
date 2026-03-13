"""Evaluate perplexity on Wikitext-103 and OWT validation split."""

import math
from typing import Dict

import numpy as np
import torch

from configs import EvalDatasetConfig, ProjectConfig
from src.model import ParrotLLM
from src.utils import build_tokenizer, maybe_load_hf_token
from datasets import load_dataset


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


def eval_wikitext(
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    dataset_cfg: EvalDatasetConfig | None,
    batch_size: int,
    max_sequences: int,
    hf_token: str | None,
):
    mc = config["model"]
    tokenizer = build_tokenizer()

    subset = dataset_cfg.subset if dataset_cfg and dataset_cfg.subset else "wikitext-103-raw-v1"
    split = dataset_cfg.split if dataset_cfg and dataset_cfg.split else "test"

    load_kwargs = {"split": split}
    if hf_token:
        load_kwargs["use_auth_token"] = hf_token
    ds = load_dataset("wikitext", subset, **load_kwargs)
    text = "\n\n".join(ds["text"])
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    ppl = compute_perplexity(
        model,
        token_ids,
        mc["context_length"],
        device,
        batch_size,
        max_sequences,
    )
    return ppl


def eval_owt_val(
    model, config, device, dataset_cfg: EvalDatasetConfig | None, batch_size: int, max_sequences: int
):
    mc = config["model"]
    val_path = dataset_cfg.path if dataset_cfg and dataset_cfg.path else "data/processed/val.bin"

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    token_ids = torch.from_numpy(data.astype(np.int64))

    ppl = compute_perplexity(
        model,
        token_ids,
        mc["context_length"],
        device,
        batch_size,
        max_sequences,
    )
    return ppl


def run_eval(
    project_config: ProjectConfig,
    full_config: dict,
    *,
    checkpoint: str,
    device: torch.device,
    hf_token: str | None = None,
) -> None:
    eval_cfg = project_config.eval
    if eval_cfg is None:
        raise ValueError("Eval configuration missing; cannot run eval stage.")

    datasets = {ds.name: ds for ds in eval_cfg.datasets}

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", full_config)

    model = ParrotLLM(ckpt_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded checkpoint: {checkpoint}")
    print(f"[eval] parameters: {model.count_parameters():,}")

    # Wikitext-103
    try:
        wt_dataset = datasets.get("wikitext")
        wt_ppl = eval_wikitext(
            model,
            ckpt_config,
            device,
            wt_dataset,
            eval_cfg.batch_size,
            eval_cfg.max_sequences,
            hf_token,
        )
        print(f"[eval] Wikitext-103 perplexity: {wt_ppl:.2f}")
    except Exception as e:
        print(f"[eval] Wikitext-103 skipped: {e}")

    # OWT val
    try:
        owt_dataset = datasets.get("owt_val")
        owt_ppl = eval_owt_val(
            model,
            ckpt_config,
            device,
            owt_dataset,
            eval_cfg.batch_size,
            eval_cfg.max_sequences,
        )
        print(f"[eval] OWT val perplexity: {owt_ppl:.2f}")
    except Exception as e:
        print(f"[eval] OWT val skipped: {e}")
