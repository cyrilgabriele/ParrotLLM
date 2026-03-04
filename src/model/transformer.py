"""ParrotLLM transformer — a decoder-only language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Causal self-attention; Flash Attention if available
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        
        # Re-assemble all head outputs side by side
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and residual dropout
        out = self.resid_dropout(self.o_proj(out))
        return out


# ── GELU MLP ─────────────────────────────────────────────────────────────────

class GELUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff, bias=bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ── Transformer Block ───────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, bias, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GELUMLP(d_model, d_ff, bias, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── ParrotLLM ────────────────────────────────────────────────────────────────

class ParrotLLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        mc = config["model"]
        self.config = mc

        self.tok_emb = nn.Embedding(mc["vocab_size"], mc["d_model"])
        self.pos_emb = nn.Embedding(mc["context_length"], mc["d_model"])
        self.dropout = nn.Dropout(mc.get("dropout", 0.0))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                mc["d_model"], mc["n_heads"], mc["d_ff"],
                mc.get("bias", False), mc.get("dropout", 0.0),
            )
            for _ in range(mc["n_layers"])
        ])
        self.ln_f = nn.LayerNorm(mc["d_model"])
        self.lm_head = nn.Linear(mc["d_model"], mc["vocab_size"], bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        n_layers = self.config["n_layers"]
        # GPT-2 style initialization
        for name, p in self.named_parameters():
            if name.endswith("weight") and p.dim() >= 2:
                # Scaled init for residual projections
                if name.endswith("o_proj.weight") or name.endswith("c_proj.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        tok_emb = self.tok_emb(idx) # (B, T, d_model)
        pos_emb = self.pos_emb(pos) # (T, d_model)
        
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

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
