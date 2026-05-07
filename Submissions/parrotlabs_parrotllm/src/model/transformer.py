"""ParrotLLM transformer — a decoder-only language model."""
from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, arXiv:1910.07467).

    Removes the mean-centering step from LayerNorm, keeping only RMS re-scaling.
    Equivalent quality, 11-34% faster for transformers. Used by LLaMA, Mistral,
    MobileLLM, Gemma.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex RoPE frequencies (Su et al., arXiv:2104.09864).

    Returns a (max_seq_len, dim // 2) complex64 tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x.

    x: (B, n_heads, T, d_head)
    freqs_cis: (T, d_head // 2) complex
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head//2)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).reshape(x.shape).type_as(x)


# ── Multi-Head Attention ─────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # QK-Norm: bound attention logit magnitude for training stability at depth
        # (Dehghani et al., arXiv:2302.05442). Applied before RoPE, after projection.
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply QK-Norm then RoPE (norm-before-rotation, as in Gemma 3/4 and OLMo 2).
        # When a KV cache is supplied, freqs_cis already corresponds to the
        # absolute positions [cached_len .. cached_len + T_new) of the new tokens.
        q = apply_rope(self.q_norm(q), freqs_cis)
        k = apply_rope(self.k_norm(k), freqs_cis)

        # If a KV cache is present, concat prior K/V (computed at their own
        # absolute positions, already RoPE-rotated) with the new K/V along time.
        if past_kv is not None:
            k_cached, v_cached = past_kv
            full_k = torch.cat([k_cached, k], dim=2)
            full_v = torch.cat([v_cached, v], dim=2)
        else:
            full_k = k
            full_v = v

        # Causal self-attention; Flash Attention if available.
        # PyTorch's is_causal aligns the causal mask to the TOP-LEFT, which is
        # only correct when q_len == k_len. For cached decode (q_len < k_len),
        # we build an explicit mask: q at position (cached_len + i) may attend
        # to all k positions in [0, cached_len + i + 1).
        if past_kv is not None:
            k_len = full_k.size(2)
            cached_len = k_len - T
            # Row r corresponds to query position (cached_len + r); allowed
            # columns are [0, cached_len + r + 1).
            row_pos = torch.arange(T, device=q.device) + cached_len
            col_pos = torch.arange(k_len, device=q.device)
            attn_mask = col_pos.unsqueeze(0) <= row_pos.unsqueeze(1)  # (T, k_len) bool
            out = F.scaled_dot_product_attention(
                q, full_k, full_v, attn_mask=attn_mask, is_causal=False,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, full_k, full_v, is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(out))

        if past_kv is not None or return_kv:
            return out, (full_k, full_v)
        return out


# ── SwiGLU MLP ───────────────────────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Shazeer, arXiv:2002.05202).

    Uses SiLU-gated mechanism with 3 projections. d_ff should be 8/3 * d_model
    (rounded) to match the parameter count of a standard 4x GELU FFN.
    Used by LLaMA, Mistral, PaLM, MobileLLM, Gemma.
    """
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        x = gate * self.up_proj(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


# ── Transformer Block ───────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Transformer block using Peri-LN normalization (arXiv:2502.02732).

    Peri-LN applies RMSNorm both before (pre-norm) and after (post-sublayer norm)
    each sub-layer: x = x + Norm(Module(Norm(x))). This is the strategy used by
    OLMo 2 (arXiv:2501.00656) and shown to outperform plain Pre-LN at all tested
    scales (400M–3.2B): more stable gradient norms, fewer loss spikes, and higher
    downstream accuracy (+1.9 avg zero-shot at 400M scale).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, bias, dropout)
        self.ln_1_out = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff, bias, dropout)
        self.ln_2_out = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if past_kv is not None or return_kv:
            attn_out, new_kv = self.attn(
                self.ln_1(x), freqs_cis, past_kv=past_kv, return_kv=True
            )
            x = x + self.ln_1_out(attn_out)
        else:
            x = x + self.ln_1_out(self.attn(self.ln_1(x), freqs_cis))
            new_kv = None
        mlp_in = self.ln_2(x) if hasattr(self, "ln_2") else x
        mlp_out = self.mlp(mlp_in)
        if hasattr(self, "ln_2_out"):
            mlp_out = self.ln_2_out(mlp_out)
        x = x + mlp_out
        if new_kv is not None:
            return x, new_kv
        return x


# ── ParrotLLM ────────────────────────────────────────────────────────────────

class ParrotLLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        self.config = mc
        self.gradient_checkpointing = bool(mc.get("gradient_checkpointing", False))

        self.tok_emb = nn.Embedding(mc["vocab_size"], mc["d_model"])
        self.dropout = nn.Dropout(mc.get("dropout", 0.0))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                mc["d_model"], mc["n_heads"], mc["d_ff"],
                mc.get("bias", False), mc.get("dropout", 0.0),
            )
            for _ in range(mc["n_layers"])
        ])
        self.ln_f = RMSNorm(mc["d_model"])
        self.lm_head = nn.Linear(mc["d_model"], mc["vocab_size"], bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies — not a learned parameter, just a buffer
        d_head = mc["d_model"] // mc["n_heads"]
        freqs_cis = precompute_rope_freqs(
            d_head, mc["context_length"], theta=mc.get("rope_theta", 10000.0)
        )
        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        n_layers = self.config["n_layers"]
        # Truncated normal initialization (OLMo 2 style): same std as GPT-2 but
        # truncated at ±3σ to prevent rare large initial weights that cause early
        # instability. trunc_normal_ clips values outside [a, b].
        for name, p in self.named_parameters():
            if name.endswith("weight") and p.dim() >= 2:
                # Scaled init for residual projections (GPT-2 style depth scaling)
                if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                    std = 0.02 / math.sqrt(2 * n_layers)
                else:
                    std = 0.02
                nn.init.trunc_normal_(p, mean=0.0, std=std, a=-3 * std, b=3 * std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def _compute_loss_in_chunks(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
        z_loss_coeff: float = 0.0,
        loss_chunk_rows: int = 2048,
    ) -> torch.Tensor:
        """Compute CE (+ optional z-loss) without materializing full-sequence logits."""
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_targets = targets.reshape(-1)
        flat_mask = loss_mask.reshape(-1).to(dtype=torch.float32) if loss_mask is not None else None

        total_ce = torch.zeros((), device=hidden.device, dtype=torch.float32)
        total_z = torch.zeros((), device=hidden.device, dtype=torch.float32)
        denom = flat_mask.sum().clamp_min(1.0) if flat_mask is not None else torch.tensor(
            float(flat_targets.numel()),
            device=hidden.device,
            dtype=torch.float32,
        )

        def accumulate_chunk(
            hidden_chunk: torch.Tensor,
            target_chunk: torch.Tensor,
            mask_chunk: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            try:
                logits_chunk = F.linear(hidden_chunk, self.lm_head.weight, self.lm_head.bias)
                ce_chunk = F.cross_entropy(
                    logits_chunk,
                    target_chunk,
                    reduction="none",
                )
                ce_total = (ce_chunk * mask_chunk).sum() if mask_chunk is not None else ce_chunk.sum()

                z_total = torch.zeros((), device=hidden_chunk.device, dtype=torch.float32)
                if z_loss_coeff > 0.0:
                    z_chunk = torch.logsumexp(logits_chunk.float(), dim=-1).pow(2)
                    z_total = (z_chunk * mask_chunk).sum() if mask_chunk is not None else z_chunk.sum()
                return ce_total, z_total
            except RuntimeError as exc:
                if "MPS backend out of memory" not in str(exc) or hidden_chunk.size(0) <= 1:
                    raise
                if hidden_chunk.device.type == "mps":
                    gc.collect()
                    torch.mps.empty_cache()
                midpoint = hidden_chunk.size(0) // 2
                left_ce, left_z = accumulate_chunk(
                    hidden_chunk[:midpoint],
                    target_chunk[:midpoint],
                    mask_chunk[:midpoint] if mask_chunk is not None else None,
                )
                right_ce, right_z = accumulate_chunk(
                    hidden_chunk[midpoint:],
                    target_chunk[midpoint:],
                    mask_chunk[midpoint:] if mask_chunk is not None else None,
                )
                return left_ce + right_ce, left_z + right_z

        for start in range(0, flat_hidden.size(0), loss_chunk_rows):
            stop = start + loss_chunk_rows
            hidden_chunk = flat_hidden[start:stop]
            target_chunk = flat_targets[start:stop]
            mask_chunk = flat_mask[start:stop] if flat_mask is not None else None
            ce_total, z_total = accumulate_chunk(hidden_chunk, target_chunk, mask_chunk)
            total_ce = total_ce + ce_total
            total_z = total_z + z_total

        loss = total_ce / denom
        if z_loss_coeff > 0.0:
            loss = loss + z_loss_coeff * (total_z / denom)
        return loss

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        loss_mask: torch.Tensor | None = None,
        return_logits: bool = True,
        z_loss_coeff: float = 0.0,
        loss_chunk_rows: int = 2048,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]
    ]:
        _, T = idx.shape
        cache_active = use_cache or past_kv is not None
        cached_len = past_kv[0][0].size(2) if past_kv is not None else 0

        x = self.dropout(self.tok_emb(idx))

        freqs_cis = self.freqs_cis[cached_len : cached_len + T]
        new_past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if cache_active else None
        )
        for layer_idx, block in enumerate(self.blocks):
            layer_past = past_kv[layer_idx] if past_kv is not None else None
            if self.gradient_checkpointing and self.training:
                def custom_forward(tensor):
                    return block(tensor, freqs_cis)

                x = checkpoint(custom_forward, x, use_reentrant=False)
            elif cache_active:
                x, layer_new_kv = block(
                    x, freqs_cis, past_kv=layer_past, return_kv=True
                )
                new_past_kv.append(layer_new_kv)
            else:
                x = block(x, freqs_cis)

        if hasattr(self, "ln_f"):
            x = self.ln_f(x)

        logits = self.lm_head(x) if (targets is None or return_logits) else None
        loss = None
        if targets is not None:
            if logits is None:
                loss = self._compute_loss_in_chunks(
                    x,
                    targets,
                    loss_mask=loss_mask,
                    z_loss_coeff=z_loss_coeff,
                    loss_chunk_rows=loss_chunk_rows,
                )
            else:
                losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                ).view_as(targets)
                if loss_mask is not None:
                    denom = loss_mask.sum().clamp_min(1.0)
                    loss = (losses * loss_mask).sum() / denom
                else:
                    loss = losses.mean()
                if z_loss_coeff > 0.0:
                    z_term = torch.logsumexp(logits.float(), dim=-1).pow(2)
                    if loss_mask is not None:
                        denom = loss_mask.sum().clamp_min(1.0)
                        loss = loss + z_loss_coeff * ((z_term * loss_mask).sum() / denom)
                    else:
                        loss = loss + z_loss_coeff * z_term.mean()
        if cache_active:
            return logits, loss, new_past_kv
        return logits, loss

    def count_parameters(self) -> int:
        """Count trainable parameters (excluding weight-tied lm_head)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total
