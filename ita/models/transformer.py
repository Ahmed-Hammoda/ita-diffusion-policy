"""Transformer building blocks written from scratch.

Rules for this file (part of the grading):
  * Do NOT use nn.Transformer, nn.TransformerEncoderLayer, nn.MultiheadAttention or
    F.scaled_dot_product_attention. Only nn.Linear, nn.LayerNorm, nn.Dropout, activations, matmul, softmax.
  * Shapes are always batch-first: (B, T, D).
  * Mask convention: boolean tensor, True = "this query MAY attend to this key".
    Shape (T_q, T_k) or (B, T_q, T_k); masked scores are set to -inf before the softmax.

The specification is in tests/test_transformer.py; notebooks/step4_transformer.ipynb explains and
checks every class with small experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_head)) V with n_heads heads, then an output projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None  # weights of the last call, kept for inspection

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, d_head)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """q: (B, T_q, D), k/v: (B, T_k, D) -> (B, T_q, D)"""
        B, T_q, D = q.shape

        # Project, then split into heads.
        Q = self._split_heads(self.W_q(q))   # (B, H, T_q, d_head)
        K = self._split_heads(self.W_k(k))   # (B, H, T_k, d_head)
        V = self._split_heads(self.W_v(v))   # (B, H, T_k, d_head)

        # Similarity of every query with every key, scaled so the softmax stays smooth.
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)   # (B, H, T_q, T_k)

        # Mask: True = may attend. Add the missing dims so it broadcasts over batch and heads.
        if mask is not None:
            if mask.dim() == 2:          # (T_q, T_k)
                mask = mask[None, None]
            elif mask.dim() == 3:        # (B, T_q, T_k)
                mask = mask[:, None]
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)   # over the keys
        self.last_attn = weights.detach()
        weights = self.dropout(weights)

        out = weights @ V                                  # (B, H, T_q, d_head)
        out = out.transpose(1, 2).reshape(B, T_q, D)       # merge heads -> (B, T_q, D)
        return self.W_o(out)


def _sinusoid(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sin/cos features of 'Attention Is All You Need'. positions: (N,) -> (N, dim).

    Feature pair i uses the frequency 1 / 10000^(2i / dim): sin on even columns, cos on odd ones.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=positions.device) / half)
    args = positions.float()[:, None] * freqs[None, :]          # (N, half)
    out = torch.zeros(positions.shape[0], dim, device=positions.device)
    out[:, 0:2 * half:2] = torch.sin(args)
    out[:, 1:2 * half:2] = torch.cos(args)
    return out


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos encoding from 'Attention Is All You Need'. Adds PE to x: (B, T, D) -> (B, T, D)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = _sinusoid(torch.arange(max_len), d_model)               # (max_len, D)
        self.register_buffer("pe", pe[None], persistent=False)       # (1, max_len, D), not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        assert T <= self.pe.shape[1], f"sequence length {T} > max_len {self.pe.shape[1]}"
        return x + self.pe[:, :T]


class SinusoidalTimestepEmbedding(nn.Module):
    """Embeds the diffusion step k (B,) integer -> (B, D). Same sin/cos formula, then an MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoid(t, self.d_model))


class FeedForward(nn.Module):
    """Linear(D, d_ff) -> GELU -> Dropout -> Linear(d_ff, D), applied to every token independently."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN: x = x + SelfAttn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN: x = x + SelfAttn(LN(x)); x = x + CrossAttn(LN(x), memory); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                self_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) action tokens, memory: (B, S, D) encoder output -> (B, T, D)"""
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, self_mask))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, cross_mask))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


class Encoder(nn.Module):
    """Stack of n_layers EncoderLayer + final LayerNorm."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of n_layers DecoderLayer + final LayerNorm."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                self_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)
