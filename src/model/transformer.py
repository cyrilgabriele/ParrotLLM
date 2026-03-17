"""ParrotLLM transformer — a decoder-only language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # (Dehghani et al., arXiv:2302.05442). Applied after RoPE, before attention.
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE then QK-Norm to Q and K
        q = self.q_norm(apply_rope(q, freqs_cis))
        k = self.k_norm(apply_rope(k, freqs_cis))

        # Causal self-attention; Flash Attention if available
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(out))
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, bias, dropout)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff, bias, dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# ── ParrotLLM ────────────────────────────────────────────────────────────────

class ParrotLLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        self.config = mc

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
        # GPT-2 style initialization
        for name, p in self.named_parameters():
            if name.endswith("weight") and p.dim() >= 2:
                # Scaled init for residual projections
                if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape

        x = self.dropout(self.tok_emb(idx))

        freqs_cis = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, freqs_cis)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
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
