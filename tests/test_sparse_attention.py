"""Tests for Sparse Attention backend, block gathering, and full-model integration."""

import pytest
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.attention.base import KVCache
from model.attention.sparse import SparseAttention, gather_kv_blocks
from model.model import SparDAResearchModel


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=1000,
        n_layer=4,
        n_embd=256,
        n_head=8,
        n_kv_head=2,
        head_dim=32,
        ffn_dim=512,
        forecast_heads=2,
        forecast_dim=32,
        context_length=512,
        block_size=16,
        top_k_blocks=2,
        attention_backend="sparse",
    )


def test_gather_kv_blocks():
    batch = 2
    n_kv_head = 2
    seq_len = 48  # exactly 3 blocks of size 16
    head_dim = 32
    block_size = 16

    k = torch.randn(batch, n_kv_head, seq_len, head_dim, requires_grad=True)
    v = torch.randn(batch, n_kv_head, seq_len, head_dim, requires_grad=True)

    # Select blocks [0, 2] -> 2 blocks per head
    block_indices = torch.tensor([[[0, 2], [1, 2]], [[2, 0], [0, 1]]])  # [2, 2, 2]

    k_gat, v_gat = gather_kv_blocks(k, v, block_indices, block_size)
    assert k_gat.shape == (batch, n_kv_head, 2 * block_size, head_dim)
    assert v_gat.shape == (batch, n_kv_head, 2 * block_size, head_dim)

    # Check that block 0 content matches k[:, :, 0:16]
    assert torch.allclose(k_gat[0, 0, :16], k[0, 0, :16])
    # Check that block 2 content matches k[:, :, 32:48]
    assert torch.allclose(k_gat[0, 0, 16:], k[0, 0, 32:48])

    # Check gradient backpropagation
    loss = k_gat.sum() + v_gat.sum()
    loss.backward()
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(k.grad).any()


def test_sparse_attention_output_shape_and_grad(config):
    batch = 2
    seq_len = 64
    x = torch.randn(batch, seq_len, config.n_embd, requires_grad=True)
    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=seq_len)
    forecast_vec = torch.randn(batch, seq_len, config.forecast_heads, config.forecast_dim)

    attn = SparseAttention(config, layer_idx=1)
    out = attn(x, rotary_emb=rotary, forecast_vector=forecast_vec)

    assert out.shape == (batch, seq_len, config.n_embd)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_sparse_attention_causality(config):
    batch = 1
    seq_len = 64
    x = torch.randn(batch, seq_len, config.n_embd)
    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=seq_len)
    forecast_vec = torch.randn(batch, seq_len, config.forecast_heads, config.forecast_dim)

    attn = SparseAttention(config, layer_idx=1)
    _, weights = attn(x, rotary_emb=rotary, forecast_vector=forecast_vec, return_attn_weights=True)

    # Upper triangular beyond diagonal must have zero attention weight
    upper_tri = torch.triu(weights[0, 0], diagonal=1)
    assert torch.all(upper_tri == 0.0)


def test_sparse_attention_kv_cache_decoding(config):
    batch = 1
    prompt_len = 48
    attn = SparseAttention(config, layer_idx=1)
    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=128)
    cache = KVCache(batch, 128, config.n_kv_head, config.head_dim, torch.device("cpu"), torch.float32)

    prompt = torch.randn(batch, prompt_len, config.n_embd)
    fc_prompt = torch.randn(batch, prompt_len, config.forecast_heads, config.forecast_dim)

    # 1. Prefill
    _ = attn(prompt, rotary_emb=rotary, kv_cache=cache, forecast_vector=fc_prompt)
    assert cache.current_len == prompt_len

    # 2. Step-by-step autoregressive decoding
    for step in range(5):
        tok = torch.randn(batch, 1, config.n_embd)
        fc_tok = torch.randn(batch, 1, config.forecast_heads, config.forecast_dim)
        out = attn(tok, rotary_emb=rotary, kv_cache=cache, forecast_vector=fc_tok)
        assert out.shape == (batch, 1, config.n_embd)

    assert cache.current_len == prompt_len + 5


def test_full_model_sparse_forward_and_generation(config):
    model = SparDAResearchModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    targets = torch.randint(0, config.vocab_size, (2, 64))

    # Forward pass with loss
    out = model(input_ids, targets=targets, return_forecasts=True)
    assert "logits" in out
    assert "loss" in out
    assert "forecasts" in out
    assert len(out["forecasts"]) == config.n_layer
    assert out["loss"].item() > 0

    # Backward pass with combined LM and auxiliary forecast loss
    total_loss = out["loss"] + 0.01 * sum(f.sum() for f in out["forecasts"])
    total_loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"

    # Generation
    prompt = torch.tensor([[10, 20, 30]])
    gen = model.generate(prompt, max_new_tokens=10, temperature=0.7)
    assert gen.shape == (1, 13)
