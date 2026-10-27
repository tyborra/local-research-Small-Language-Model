"""Tests for Phase 8 Unified Scaling Matrix Benchmark Suite."""

import pytest
import torch
from model.config import ModelConfig
from benchmarks.scaling_matrix import (
    BACKEND_CONFIGS,
    build_model_for_backend,
    create_kv_caches,
    profile_single_backend_context,
    identify_crossover_points,
)


@pytest.fixture
def base_config():
    return ModelConfig(
        vocab_size=1000,
        n_layer=2,
        n_embd=128,
        n_head=4,
        n_kv_head=2,
        head_dim=32,
        ffn_dim=256,
        forecast_heads=2,
        forecast_dim=32,
        context_length=128,
        block_size=16,
        top_k_blocks=2,
    )


def test_build_model_for_all_backends(base_config):
    device = torch.device("cpu")
    for backend_key in BACKEND_CONFIGS.keys():
        model = build_model_for_backend(base_config, backend_key, device)
        assert model is not None
        assert model.config.attention_backend == BACKEND_CONFIGS[backend_key]["backend"]
        assert model.config.sparse_mode == BACKEND_CONFIGS[backend_key]["sparse_mode"]


def test_create_kv_caches_types(base_config):
    device = torch.device("cpu")
    model = build_model_for_backend(base_config, "dense", device)
    caches = create_kv_caches("dense", model, batch_size=1, max_seq_len=64, device=device, dtype=torch.float32)
    assert len(caches) == base_config.n_layer

    offload_caches = create_kv_caches("offload", model, batch_size=1, max_seq_len=64, device=device, dtype=torch.float32)
    assert len(offload_caches) == base_config.n_layer

    async_caches = create_kv_caches("async_offload", model, batch_size=1, max_seq_len=64, device=device, dtype=torch.float32)
    assert len(async_caches) == base_config.n_layer


def test_identify_crossover_points():
    comparative_matrix = [
        {
            "backend": "forecast",
            "context_length": 1024,
            "prefill_speedup_vs_dense": 0.8,
            "decode_speedup_vs_dense": 0.9,
            "kv_reduction_vs_dense": 1.0,
            "prefill_tok_per_sec": 10000.0,
            "decode_tok_per_sec": 20.0,
        },
        {
            "backend": "forecast",
            "context_length": 8192,
            "prefill_speedup_vs_dense": 1.2,
            "decode_speedup_vs_dense": 1.4,
            "kv_reduction_vs_dense": 4.0,
            "prefill_tok_per_sec": 15000.0,
            "decode_tok_per_sec": 25.0,
        },
        {
            "backend": "offload",
            "context_length": 1024,
            "prefill_speedup_vs_dense": 0.7,
            "decode_speedup_vs_dense": 0.4,
            "kv_reduction_vs_dense": 5.4,
            "prefill_tok_per_sec": 8000.0,
            "decode_tok_per_sec": 10.0,
        },
        {
            "backend": "offload",
            "context_length": 16384,
            "prefill_speedup_vs_dense": 0.6,
            "decode_speedup_vs_dense": 0.5,
            "kv_reduction_vs_dense": 18.2,
            "prefill_tok_per_sec": 7000.0,
            "decode_tok_per_sec": 9.0,
        },
    ]

    crossovers = identify_crossover_points(
        comparative_matrix,
        backends=["dense", "forecast", "offload"],
        context_lengths=[1024, 8192, 16384],
    )

    assert "forecast" in crossovers
    assert crossovers["forecast"]["prefill_speedup_crossover_tokens"] == 8192
    assert crossovers["forecast"]["decode_speedup_crossover_tokens"] == 8192
    assert crossovers["forecast"]["kv_50pct_reduction_crossover_tokens"] == 8192

    assert "offload" in crossovers
    assert crossovers["offload"]["kv_50pct_reduction_crossover_tokens"] == 1024
    assert crossovers["offload"]["kv_90pct_reduction_crossover_tokens"] == 16384
