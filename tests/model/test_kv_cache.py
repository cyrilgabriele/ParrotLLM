"""Equivalence test: KV-cached forward must produce bit-identical logits to a full-sequence forward."""

from __future__ import annotations

import torch

from src.model.transformer import ParrotLLM


def _make_tiny_model(seed: int = 42) -> ParrotLLM:
    torch.manual_seed(seed)
    config = {
        "model": {
            "vocab_size": 256,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "d_ff": 64,
            "context_length": 32,
            "bias": False,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "gradient_checkpointing": False,
        }
    }
    model = ParrotLLM(config)
    model.eval()
    return model


@torch.no_grad()
def test_kv_cache_matches_full_forward_logits():
    """Cached prefill + per-token decode must produce the same logits as one full forward."""
    model = _make_tiny_model()
    torch.manual_seed(0)
    idx = torch.randint(low=3, high=256, size=(1, 12))

    # Reference: one full forward pass over the entire sequence.
    logits_full, _ = model(idx)  # (B, T, V)

    # Compare against: prefill 8 tokens with cache, then decode the remaining 4 one at a time.
    prefix, suffix = idx[:, :8], idx[:, 8:]
    out_prefix, _, past_kv = model(prefix, use_cache=True)
    # out_prefix logits for the last prefix token must match logits_full at position 7.
    torch.testing.assert_close(
        out_prefix[:, -1, :],
        logits_full[:, 7, :],
        atol=1e-5,
        rtol=1e-5,
    )

    # Decode each suffix token, one at a time, threading the cache through.
    for step in range(suffix.size(1)):
        next_tok = suffix[:, step:step + 1]
        out_step, _, past_kv = model(next_tok, past_kv=past_kv, use_cache=True)
        torch.testing.assert_close(
            out_step[:, -1, :],
            logits_full[:, 8 + step, :],
            atol=1e-5,
            rtol=1e-5,
            msg=f"mismatch at decode step {step}",
        )


@torch.no_grad()
def test_kv_cache_shapes():
    """Cache should grow by T per call."""
    model = _make_tiny_model()
    idx = torch.randint(low=3, high=256, size=(1, 5))

    _, _, past_kv = model(idx, use_cache=True)
    assert isinstance(past_kv, list)
    assert len(past_kv) == 2  # n_layers
    for k, v in past_kv:
        assert k.shape == (1, 2, 5, 16)  # B, n_heads, T, d_head
        assert v.shape == (1, 2, 5, 16)

    # Decode one more token; cache grows by 1.
    next_tok = torch.randint(low=3, high=256, size=(1, 1))
    _, _, past_kv2 = model(next_tok, past_kv=past_kv, use_cache=True)
    for k, v in past_kv2:
        assert k.shape == (1, 2, 6, 16)
        assert v.shape == (1, 2, 6, 16)


@torch.no_grad()
def test_no_cache_path_unchanged():
    """When use_cache is not set, return must remain a 2-tuple (backward compat)."""
    model = _make_tiny_model()
    idx = torch.randint(low=3, high=256, size=(1, 5))
    out = model(idx)
    assert isinstance(out, tuple)
    assert len(out) == 2  # (logits, loss=None)


@torch.no_grad()
def test_generate_uses_cache():
    """generate() should produce the same tokens as a non-cache greedy reference."""
    from src.eval.inference import generate
    model = _make_tiny_model()
    idx = torch.randint(low=3, high=256, size=(1, 4))

    # Greedy generate via the public generate() — should call into the cache path.
    out_cached = generate(
        model,
        idx,
        max_new_tokens=8,
        temperature=0.0,
    )
    # Reference: naive greedy (no cache) — call model fresh each step on the full prefix.
    cur = idx.clone()
    for _ in range(8):
        logits, _ = model(cur)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    out_naive = cur

    torch.testing.assert_close(out_cached, out_naive)
