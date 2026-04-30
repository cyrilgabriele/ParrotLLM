"""One full optimizer step on the active device must produce no NaN/Inf
in any logged metric.
"""
import copy
import json
from pathlib import Path

import pytest
import torch

from src.posttraining.dpo.trainer import (
    PreferencePackedDataset,
    build_dpo_collator,
    dpo_train_step,
)
from src.utils import build_tokenizer, get_device


def _write_tiny_dataset(tmp_path: Path) -> Path:
    """Two tiny pairs sharing the prompt prefix in tokens."""
    tok = build_tokenizer()
    prompt = tok.encode("### Instruction:\nHi\n\n### Response:\n")
    chosen = prompt + tok.encode("Hello!")
    rejected = prompt + tok.encode("...")
    path = tmp_path / "pairs.jsonl"
    with path.open("w") as f:
        for _ in range(4):
            f.write(json.dumps({
                "prompt_tokens": prompt,
                "chosen_tokens": chosen,
                "rejected_tokens": rejected,
                "prompt_len": len(prompt),
            }) + "\n")
    return path


def test_one_train_step_produces_finite_metrics(tmp_path: Path) -> None:
    from src.model import ParrotLLM
    cfg = {"model": {
        "vocab_size": 50258, "d_model": 32, "n_layers": 2, "n_heads": 4,
        "d_ff": 64, "context_length": 64, "bias": False,
        "dropout": 0.0, "rope_theta": 10000.0,
        "gradient_checkpointing": False,
        "pad_token_id": 50257, "bos_token_id": 50256, "eos_token_id": 50256,
    }}
    device = get_device("cpu")  # CPU keeps test reproducible across machines
    policy = ParrotLLM(cfg).to(device)
    reference = copy.deepcopy(policy).to(device)
    for p in reference.parameters():
        p.requires_grad = False
    reference.eval()

    path = _write_tiny_dataset(tmp_path)
    dataset = PreferencePackedDataset(path)
    pad_id = cfg["model"]["pad_token_id"]
    collate = build_dpo_collator(pad_token_id=pad_id)
    batch = collate([dataset[i] for i in range(4)])
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-7)
    loss, metrics = dpo_train_step(
        policy=policy,
        reference=reference,
        batch=batch,
        beta=0.1,
        optimizer=optimizer,
        grad_clip=1.0,
    )
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    for k, v in metrics.items():
        assert v == v, f"metric {k} is NaN"  # NaN != NaN
        assert abs(v) < 1e6, f"metric {k} blew up: {v}"
