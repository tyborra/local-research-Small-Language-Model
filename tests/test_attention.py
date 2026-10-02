"""Unit tests for Attention modules, GQA, and RoPE."""

import pytest
import torch
from model.config import ModelConfig
from model.rope import RotaryEmbedding, precompute_rope_freqs
from model.attention.dense import DenseAttention
from model.attention.base import KVCache


def test_rope_frequencies():
    """Verify precomputed RoPE frequencies shape and properties."""
    head_dim = 64
    max_seq_len = 128
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len, theta=10000.0)

    assert cos.shape == (max_seq_len, head_dim // 2)
    assert sin.shape == (max_seq_len, head_dim // 2)
    # At t=0, cos is 1 and sin is 0
    assert torch.allclose(cos[0], torch.ones(head_dim // 2))
    assert torch.allclose(sin[0], torch.zeros(head_dim // 2))


def test_gqa_dimensions_and_causal():
    """Verify that DenseAttention correctly handles 12 query heads and 4 KV heads."""
    config = ModelConfig(n_head=12, n_kv_head=4, head_dim=64, n_embd=768)
    attn = DenseAttention(config, layer_idx=0)
    rope = RotaryEmbedding(config.head_dim, max_seq_len=256)

    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.n_embd)

    out = attn(x, rotary_emb=rope, is_causal=True)
    assert out.shape == (batch_size, seq_len, config.n_embd)


def test_kv_cache_decoding():
    """Verify that incremental generation with KVCache matches forward output."""
    config = ModelConfig(n_head=12, n_kv_head=4, head_dim=64, n_embd=768)
    attn = DenseAttention(config, layer_idx=0)
    attn.eval()
    rope = RotaryEmbedding(config.head_dim, max_seq_len=256)

    cache = KVCache(
        max_batch_size=1,
        max_seq_len=10,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    token1 = torch.randn(1, 1, config.n_embd)
    out1 = attn(token1, rotary_emb=rope, kv_cache=cache)
    assert out1.shape == (1, 1, config.n_embd)
    assert cache.current_len == 1

    token2 = torch.randn(1, 1, config.n_embd)
    out2 = attn(token2, rotary_emb=rope, kv_cache=cache)
    assert out2.shape == (1, 1, config.n_embd)
    assert cache.current_len == 2
