"""Unit tests for Forecast-Guided KV Cache CPU Offloading."""

import pytest
import torch

from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.attention.offload import OffloadedKVCache, ForecastOffloadAttention
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
        context_length=256,
        block_size=16,
        top_k_blocks=2,
        attention_backend="offload",
    )


def test_offloaded_kv_cache_pinned_and_transfer():
    batch = 1
    max_len = 64
    n_kv_head = 2
    head_dim = 32
    block_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = OffloadedKVCache(
        max_batch_size=batch,
        max_seq_len=max_len,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        gpu_device=device,
        dtype=torch.float32,
    )

    # Initial state
    assert cache.current_len == 0
    assert cache.cpu_k.device.type == "cpu"
    assert cache.cpu_v.device.type == "cpu"
    if device.type == "cuda":
        assert cache.cpu_k.is_pinned()

    # Append 32 tokens
    k_chunk = torch.randn(batch, 32, n_kv_head, head_dim, device=device)
    v_chunk = torch.randn(batch, 32, n_kv_head, head_dim, device=device)
    curr_len = cache.update(k_chunk, v_chunk)
    assert curr_len == 32
    assert cache.current_len == 32

    # Verify CPU slice matches input
    assert torch.allclose(cache.cpu_k[:, :32], k_chunk.cpu(), atol=1e-5)

    # Transfer block 0 and block 1
    block_indices = torch.tensor([[[0], [1]]], device=device)  # [1, 2, 1]
    gpu_k, gpu_v, bytes_trans = cache.transfer_selected_blocks(
        block_indices=block_indices,
        block_size=block_size,
    )

    assert gpu_k.device.type == device.type
    assert gpu_v.device.type == device.type
    assert gpu_k.shape == (batch, n_kv_head, block_size, head_dim)
    assert bytes_trans == (gpu_k.numel() + gpu_v.numel()) * gpu_k.element_size()
    assert cache.total_bytes_transferred == bytes_trans


def test_forecast_offload_attention_decode_step(config):
    batch = 1
    prompt_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = ForecastOffloadAttention(config, layer_idx=1).to(device)
    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=128)
    cache = OffloadedKVCache(
        max_batch_size=batch,
        max_seq_len=128,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim,
        gpu_device=device,
        dtype=torch.float32,
    )

    # 1. Prefill
    prompt = torch.randn(batch, prompt_len, config.n_embd, device=device)
    _ = attn(prompt, rotary_emb=rotary, kv_cache=cache)
    assert cache.current_len == prompt_len

    # 2. Decode step
    tok = torch.randn(batch, 1, config.n_embd, device=device)
    fc_tok = torch.randn(batch, 1, config.forecast_heads, config.forecast_dim, device=device)
    out = attn(tok, rotary_emb=rotary, kv_cache=cache, forecast_vector=fc_tok)

    assert out.shape == (batch, 1, config.n_embd)
    assert cache.current_len == prompt_len + 1
    assert "selection_ms" in attn.last_profile
    assert "transfer_ms" in attn.last_profile
    assert "attention_ms" in attn.last_profile
    assert "total_ms" in attn.last_profile
    assert attn.last_profile["bytes_transferred"] > 0


def test_full_model_offload_forward_and_generate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparDAResearchModel(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    targets = torch.randint(0, config.vocab_size, (2, 32), device=device)

    # Forward pass
    out = model(input_ids, targets=targets, return_forecasts=True)
    assert "logits" in out
    assert "loss" in out
    assert out["loss"].item() > 0

    # Generation test
    prompt = torch.tensor([[10, 20, 30]], device=device)
    gen = model.generate(prompt, max_new_tokens=5, temperature=0.7)
    assert gen.shape == (1, 8)
