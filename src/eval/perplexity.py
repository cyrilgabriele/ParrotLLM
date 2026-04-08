"""Evaluate perplexity on Wikitext-103 and OWT validation split.

Uses a sliding-window approach (stride = context_length // 2) following
Radford et al. (2019) and the HuggingFace evaluate convention so that
reported numbers are directly comparable to published results.
"""

import logging
import math
from typing import Dict

log = logging.getLogger("parrotllm.eval")

import numpy as np
import torch
import torch.nn.functional as F

from configs import EvalDatasetConfig, ProjectConfig
from src.model import ParrotLLM
from src.utils import build_tokenizer
from datasets import load_dataset


def compute_perplexity(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    context_length: int,
    device: torch.device,
    batch_size: int = 32,
    max_sequences: int = 512,
    stride: int | None = None,
) -> float:
    """Token-weighted perplexity with sliding-window evaluation.

    Args:
        stride: Step between consecutive windows.  Defaults to
            ``context_length // 2`` (sliding window, paper-comparable).
            Set to ``context_length`` for non-overlapping evaluation
            matching training-time ``estimate_loss``.
    """
    model.eval()
    if stride is None:
        stride = context_length // 2

    n_tokens = len(token_ids)
    total_nll = 0.0
    total_scored = 0

    # Build (begin, score_from) pairs.
    # score_from: index within the window from which tokens are scored.
    # First window scores all positions; later windows score only the
    # last `stride` positions (the new, non-overlapping tokens).
    windows: list[tuple[int, int]] = []
    for begin in range(0, n_tokens - context_length, stride):
        score_from = 0 if begin == 0 else context_length - stride
        windows.append((begin, score_from))
        if len(windows) >= max_sequences:
            break

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        xs, ys = [], []
        for begin, _ in batch_windows:
            chunk = token_ids[begin : begin + context_length + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        with torch.no_grad():
            logits, _ = model(x, targets=None, return_logits=True)

        # Score only the non-overlapping portion of each window.
        for i, (_, score_from) in enumerate(batch_windows):
            scored_logits = logits[i, score_from:]
            scored_targets = y[i, score_from:]
            nll = F.cross_entropy(scored_logits, scored_targets, reduction="sum")
            total_nll += nll.item()
            total_scored += scored_targets.numel()

    avg_loss = total_nll / total_scored
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
        load_kwargs["token"] = hf_token
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

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", full_config)

    model = ParrotLLM(ckpt_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"loaded checkpoint: {checkpoint}")
    log.info(f"parameters: {model.count_parameters():,}")

    WIKITEXT_NAMES = {"wikitext", "wikitext103-test"}

    for ds_cfg in eval_cfg.datasets:
        try:
            if ds_cfg.name in WIKITEXT_NAMES:
                ppl = eval_wikitext(
                    model, ckpt_config, device, ds_cfg,
                    eval_cfg.batch_size, eval_cfg.max_sequences, hf_token,
                )
            elif ds_cfg.path and ds_cfg.path.endswith(".bin"):
                ppl = eval_owt_val(
                    model, ckpt_config, device, ds_cfg,
                    eval_cfg.batch_size, eval_cfg.max_sequences,
                )
            else:
                log.warning(f"Dataset '{ds_cfg.name}' skipped: unsupported type")
                continue
            log.info(f"{ds_cfg.name} perplexity: {ppl:.2f}")
        except Exception as e:
            log.warning(f"{ds_cfg.name} skipped: {e}")
