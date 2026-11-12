"""Specification for ita/models/transformer.py — these pass once the modules are implemented."""

import torch

from ita.models.transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    SinusoidalTimestepEmbedding,
)

B, T, S, D, H, FF = 2, 16, 7, 64, 4, 128


def test_attention_shape():
    attn = MultiHeadAttention(D, H)
    q, kv = torch.randn(B, T, D), torch.randn(B, S, D)
    assert attn(q, kv, kv).shape == (B, T, D)


def test_attention_mask_hides_key():
    """If key j is masked out, changing token j must not change the other outputs."""
    torch.manual_seed(0)
    attn = MultiHeadAttention(D, H).eval()
    x = torch.randn(B, S, D)
    mask = torch.ones(S, S, dtype=torch.bool)
    mask[:, -1] = False  # nobody may attend to the last token
    y1 = attn(x, x, x, mask=mask)
    x2 = x.clone()
    x2[:, -1] += 10.0
    y2 = attn(x2, x2, x2, mask=mask)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


def test_attention_is_permutation_equivariant_without_pe():
    torch.manual_seed(0)
    attn = MultiHeadAttention(D, H).eval()
    x = torch.randn(B, S, D)
    perm = torch.randperm(S)
    assert torch.allclose(attn(x, x, x)[:, perm], attn(x[:, perm], x[:, perm], x[:, perm]), atol=1e-5)


def test_positional_encoding():
    pe = SinusoidalPositionalEncoding(D, max_len=64)
    x = torch.zeros(B, T, D)
    out = pe(x)
    assert out.shape == (B, T, D)
    assert not torch.allclose(out[:, 0], out[:, 1])  # different positions get different codes
    assert torch.allclose(out[0], out[1])            # same code for every batch element


def test_timestep_embedding_shape():
    emb = SinusoidalTimestepEmbedding(D)
    assert emb(torch.tensor([0, 50])).shape == (2, D)


def test_layers_and_stacks_shapes():
    x, mem = torch.randn(B, T, D), torch.randn(B, S, D)
    assert EncoderLayer(D, H, FF)(mem).shape == (B, S, D)
    assert DecoderLayer(D, H, FF)(x, mem).shape == (B, T, D)
    assert Encoder(2, D, H, FF)(mem).shape == (B, S, D)
    assert Decoder(2, D, H, FF)(x, mem).shape == (B, T, D)


def test_gradients_flow():
    enc, dec = Encoder(2, D, H, FF), Decoder(2, D, H, FF)
    x, obs = torch.randn(B, T, D), torch.randn(B, S, D)
    dec(x, enc(obs)).pow(2).mean().backward()
    for p in list(enc.parameters()) + list(dec.parameters()):
        assert p.grad is not None
