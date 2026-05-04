"""KV-cache parity tests for ParrotLLM (VL09 slides 33–35).

The invariant we verify: feeding a sequence one token at a time through
``forward_with_cache`` (prefill + per-step decode) must yield the same
final-position logits as running the full sequence through the standard
``forward()`` path. Any divergence means the cache is silently producing
different distributions than training-time attention saw — the kind of
bug that only shows up as "the chat sounds slightly off after we shipped
the demo."
"""
from __future__ import annotations

import torch

# Import the class directly from the module to bypass src.model.__init__,
# which also imports the HF GPT-2 wrapper (transformers heavy dep) — not
# needed for these tests.
from src.model.transformer import ParrotLLM


def _tiny_config() -> dict:
    return {
        "model": {
            "vocab_size": 64,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 64,
            "context_length": 32,
            "bias": False,
            "dropout": 0.0,
            "rope_theta": 10000.0,
        }
    }


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = ParrotLLM(_tiny_config()).eval()

    prompt = torch.randint(0, 64, (1, 8))
    new_tokens = torch.randint(0, 64, (1, 5))

    # ── Reference: full forward on the entire concatenated sequence ─────
    full = torch.cat([prompt, new_tokens], dim=1)
    with torch.no_grad():
        ref_logits, _ = model(full)

    # ── Cached path: prefill the prompt, then feed each new token alone
    with torch.no_grad():
        prefill_logits, cache = model.forward_with_cache(prompt, cache=None)
        per_step_logits = [prefill_logits[:, -1, :]]
        for t in range(new_tokens.size(1)):
            step_logits, cache = model.forward_with_cache(
                new_tokens[:, t : t + 1], cache=cache,
            )
            per_step_logits.append(step_logits[:, -1, :])

    # Compare logits at positions [prompt_len-1, prompt_len, ..., full_len-1]
    # — i.e. every position where the cached path produced a "next token"
    # prediction. These must match the full forward at the same indices.
    expected_positions = list(range(prompt.size(1) - 1, full.size(1)))
    for i, pos in enumerate(expected_positions):
        ref = ref_logits[:, pos, :]
        got = per_step_logits[i]
        assert torch.allclose(ref, got, atol=1e-4), (
            f"KV-cache divergence at position {pos}: max diff "
            f"{(ref - got).abs().max().item():.2e}"
        )


def test_repetition_penalty_pushes_seen_tokens_down() -> None:
    """VL09 slide 25: ``z'_i = z_i / θ`` for tokens already seen."""
    from src.eval.inference import _apply_repetition_penalty

    logits = torch.tensor([[2.0, -1.0, 3.0, 0.5, -2.0]])
    seq = [0, 2]  # tokens 0 and 2 already in the sequence
    out = _apply_repetition_penalty(logits.clone(), seq, penalty=1.5)

    # Token 0 had +2.0 → divide by 1.5 → ~1.333
    assert torch.isclose(out[0, 0], torch.tensor(2.0 / 1.5))
    # Token 2 had +3.0 → divide by 1.5 → 2.0
    assert torch.isclose(out[0, 2], torch.tensor(3.0 / 1.5))
    # Untouched tokens unchanged
    assert torch.isclose(out[0, 1], torch.tensor(-1.0))
    assert torch.isclose(out[0, 3], torch.tensor(0.5))
    assert torch.isclose(out[0, 4], torch.tensor(-2.0))


def test_repetition_penalty_off_is_noop() -> None:
    from src.eval.inference import _apply_repetition_penalty

    logits = torch.randn(1, 10)
    out = _apply_repetition_penalty(logits.clone(), [0, 1, 2], penalty=1.0)
    assert torch.equal(out, logits)
