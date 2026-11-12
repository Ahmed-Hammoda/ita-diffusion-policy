"""Transformer building blocks written FROM SCRATCH.   [Owner: Model — week 1]

Rules for this file (part of the grading):
  * Do NOT use nn.Transformer, nn.TransformerEncoderLayer, nn.MultiheadAttention or
    F.scaled_dot_product_attention. Only nn.Linear, nn.LayerNorm, nn.Dropout, activations, matmul, softmax.
  * Shapes are always batch-first: (B, T, D).
  * Mask convention: boolean tensor, True = "this query MAY attend to this key".
    Shape (T_q, T_k) or (B, T_q, T_k); masked scores are set to -inf before the softmax.

The specification is in tests/test_transformer.py — run `pytest tests/test_transformer.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_head)) V with n_heads heads, then an output projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        # TODO: W_q, W_k, W_v, W_o as nn.Linear(d_model, d_model)
        raise NotImplementedError("TODO (Model)")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """q: (B, T_q, D), k/v: (B, T_k, D) -> (B, T_q, D)"""
        raise NotImplementedError


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos encoding from 'Attention Is All You Need'. Adds PE to x: (B, T, D) -> (B, T, D)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        raise NotImplementedError("TODO (Model): build the (max_len, d_model) table and register_buffer it")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SinusoidalTimestepEmbedding(nn.Module):
    """Embeds the diffusion step k (B,) integer -> (B, D). Same sin/cos formula, then an MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForward(nn.Module):
    """Linear(D, d_ff) -> GELU -> Dropout -> Linear(d_ff, D)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EncoderLayer(nn.Module):
    """Pre-LN: x = x + SelfAttn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class DecoderLayer(nn.Module):
    """Pre-LN: x = x + SelfAttn(LN(x)); x = x + CrossAttn(LN(x), memory); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                self_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) action tokens, memory: (B, S, D) encoder output -> (B, T, D)"""
        raise NotImplementedError


class Encoder(nn.Module):
    """Stack of n_layers EncoderLayer + final LayerNorm."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class Decoder(nn.Module):
    """Stack of n_layers DecoderLayer + final LayerNorm."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("TODO (Model)")

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                self_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
