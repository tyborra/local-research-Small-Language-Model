"""Unit tests for Asynchronous Forecast-Guided KV Cache Prefetching (Phase 6)."""

import pytest
import torch

from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.attention.offload import OffloadedKVCache, ForecastOffloadAttention
from model.attention.async_offload import AsyncOffloadedKVCache, AsyncForecastOffloadAttention
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
        attention_backend="async_offload",
    )


def test_async_offloaded_kv_cache_pinned_and_stream():
    batch = 1
    max_len = 64
    n_kv_head = 2
    head_dim = 32
    block_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = AsyncOffloadedKVCache(
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
        assert cache.transfer_stream is not None

    # Append 32 tokens
    k_chunk = torch.randn(batch, 32, n_kv_head, head_dim, device=device)
    v_chunk = torch.randn(batch, 32, n_kv_head, head_dim, device=device)
    curr_len = cache.update(k_chunk, v_chunk)
    assert curr_len == 32
    assert cache.current_len == 32

    # Verify CPU slice matches input
    assert torch.allclose(cache.cpu_k[:, :32], k_chunk.cpu(), atol=1e-5)

    # Async transfer block 0 and block 1
    block_indices = torch.tensor([[[0], [1]]], device=device)
    gpu_k, gpu_v, event, bytes_trans = cache.async_transfer_selected_blocks(
        block_indices=block_indices,
        block_size=block_size,
    )

    if device.type == "cuda" and event is not None:
        torch.cuda.current_stream().wait_event(event)

    assert gpu_k.device.type == device.type
    assert gpu_v.device.type == device.type
    assert gpu_k.shape == (batch, n_kv_head, block_size, head_dim)
    assert bytes_trans == (gpu_k.numel() + gpu_v.numel()) * gpu_k.element_size()
    assert cache.total_bytes_transferred == bytes_trans


def test_async_forecast_offload_attention_prefetch_and_decode(config):
    batch = 1
    prompt_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn = AsyncForecastOffloadAttention(config, layer_idx=1).to(device)
    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=128)
    cache = AsyncOffloadedKVCache(
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

    # 2. Prefetch call (simulating Layer 0 triggering Layer 1 prefetch)
    forecast_vector = torch.randn(batch, 1, config.forecast_heads, config.forecast_dim, device=device)
    prefetched = attn.prefetch(forecast_vector, offload_cache=cache)
    assert prefetched is True
    assert attn.prefetched_gpu_k is not None
    assert attn.prefetched_gpu_v is not None
    if device.type == "cuda":
        assert attn.prefetch_event is not None

    # 3. Decode step
    new_token = torch.randn(batch, 1, config.n_embd, device=device)
    out = attn(new_token, rotary_emb=rotary, kv_cache=cache, forecast_vector=forecast_vector)

    assert out.shape == (batch, 1, config.n_embd)
    assert not torch.isnan(out).any()
    assert cache.current_len == prompt_len + 1
    # Verify prefetch state was consumed and reset
    assert attn.prefetch_event is None
    assert attn.prefetched_gpu_k is None


def test_async_offload_numerical_parity_with_sync_offload(config):
    """Verifies that AsyncForecastOffloadAttention outputs match ForecastOffloadAttention."""
    batch = 1
    prompt_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sync offload module
    sync_config = ModelConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        head_dim=config.head_dim,
        ffn_dim=config.ffn_dim,
        forecast_heads=config.forecast_heads,
        forecast_dim=config.forecast_dim,
        context_length=config.context_length,
        block_size=config.block_size,
        top_k_blocks=config.top_k_blocks,
        attention_backend="offload",
    )
    sync_attn = ForecastOffloadAttention(sync_config, layer_idx=1).to(device)
    async_attn = AsyncForecastOffloadAttention(config, layer_idx=1).to(device)

    # Share weights
    async_attn.load_state_dict(sync_attn.state_dict())

    rotary = RotaryEmbedding(head_dim=config.head_dim, max_seq_len=128)
    sync_cache = OffloadedKVCache(batch, 128, config.n_kv_head, config.head_dim, device, torch.float32)
    async_cache = AsyncOffloadedKVCache(batch, 128, config.n_kv_head, config.head_dim, device, torch.float32)

    # Prefill identical prompts
    torch.manual_seed(42)
    prompt = torch.randn(batch, prompt_len, config.n_embd, device=device)
    _ = sync_attn(prompt, rotary_emb=rotary, kv_cache=sync_cache)
    _ = async_attn(prompt, rotary_emb=rotary, kv_cache=async_cache)

    assert sync_cache.current_len == async_cache.current_len

    # Decode step with prefetch
    new_token = torch.randn(batch, 1, config.n_embd, device=device)
    forecast_vec = torch.randn(batch, 1, config.forecast_heads, config.forecast_dim, device=device)

    # Queue prefetch on async attention
    async_attn.prefetch(forecast_vec, offload_cache=async_cache)

    out_sync = sync_attn(new_token, rotary_emb=rotary, kv_cache=sync_cache, forecast_vector=forecast_vec)
    out_async = async_attn(new_token, rotary_emb=rotary, kv_cache=async_cache, forecast_vector=forecast_vec)

    assert torch.allclose(out_sync, out_async, atol=1e-4)


def test_full_model_async_forward_and_generate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparDAResearchModel(config).to(device)
    model.eval()

    # 1. Standard forward pass (prefill)
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    out = model(input_ids)
    assert out["logits"].shape == (1, 32, config.vocab_size)

    # 2. Step-by-step decoding with AsyncOffloadedKVCache per layer
    kv_caches = [
        AsyncOffloadedKVCache(
            max_batch_size=1,
            max_seq_len=64,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            gpu_device=device,
            dtype=torch.float32,
        )
        for _ in range(config.n_layer)
    ]

    # Prefill cache with prompt
    _ = model(input_ids, kv_caches=kv_caches)
    for c in kv_caches:
        assert c.current_len == 32

    # Step 1 token with async prefetch pipeline active across layers
    next_token_id = torch.randint(0, config.vocab_size, (1, 1), device=device)
    step_out = model(next_token_id, kv_caches=kv_caches)
    assert step_out["logits"].shape == (1, 1, config.vocab_size)
    for c in kv_caches:
        assert c.current_len == 33

    # 3. Model generation smoke test
    gen_tokens = model.generate(input_ids, max_new_tokens=4)
    assert gen_tokens.shape == (1, 36)
