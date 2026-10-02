"""Unit tests for evaluation harness (perplexity, memory, latency)."""

import math
import pytest
import torch
import numpy as np

from model.config import ModelConfig
from model.model import SparDAResearchModel
from evaluation.perplexity import compute_perplexity
from evaluation.memory import MemoryProfiler, calculate_kv_cache_bytes, profile_memory
from evaluation.latency import LatencyBenchmarker, benchmark_latency
from data.dataset import TokenizedDataset, create_dataloader


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=256,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        head_dim=16,
        ffn_dim=128,
        forecast_heads=2,
        forecast_dim=16,
        context_length=64,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return SparDAResearchModel(tiny_config)


def test_calculate_kv_cache_bytes():
    # batch_size=2, seq_len=1024, n_layer=12, n_kv_head=4, head_dim=64, bf16 (2 bytes)
    # formula: 2 * 2 * 1024 * 4 * 64 * 2 * 12 = 25,165,824 bytes = 24.0 MB
    bytes_count = calculate_kv_cache_bytes(
        batch_size=2,
        seq_len=1024,
        n_layer=12,
        n_kv_head=4,
        head_dim=64,
        dtype=torch.bfloat16,
    )
    assert bytes_count == 24 * 1024 * 1024


def test_perplexity_evaluation(tiny_model, tiny_config):
    device = torch.device("cpu")
    tiny_model.to(device)

    # Create synthetic token data (1,000 tokens)
    tokens = np.random.randint(0, tiny_config.vocab_size, size=500, dtype=np.uint16)
    dataset = TokenizedDataset(tokens, seq_len=tiny_config.context_length)
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

    results = compute_perplexity(
        model=tiny_model,
        val_data=dataloader,
        device=device,
        dtype=torch.float32,
    )

    assert "eval_loss" in results
    assert "perplexity" in results
    assert "bits_per_token" in results
    assert results["total_tokens"] > 0
    assert results["eval_loss"] > 0
    assert math.isclose(results["perplexity"], math.exp(results["eval_loss"]), rel_tol=1e-4)
    assert math.isclose(results["bits_per_token"], results["eval_loss"] / math.log(2.0), rel_tol=1e-4)


def test_memory_profiler_static(tiny_model):
    device = torch.device("cpu")
    profiler = MemoryProfiler(device)
    static_mem = profiler.profile_static_weights(tiny_model)

    assert "theoretical_weights_mb" in static_mem
    assert static_mem["theoretical_weights_mb"] > 0.0


def test_latency_benchmarker_cpu(tiny_model, tiny_config):
    device = torch.device("cpu")
    tiny_model.to(device)
    benchmarker = LatencyBenchmarker(device)

    prefill = benchmarker.benchmark_prefill(
        model=tiny_model,
        batch_size=1,
        prompt_len=32,
        dtype=torch.float32,
        warmup_runs=1,
        num_runs=2,
    )
    assert prefill["prefill_mean_ms"] > 0.0
    assert prefill["prefill_tok_per_sec"] > 0.0

    decode = benchmarker.benchmark_decode(
        model=tiny_model,
        batch_size=1,
        prompt_len=32,
        num_decode_tokens=4,
        dtype=torch.float32,
        warmup_runs=1,
        num_runs=2,
    )
    assert decode["decode_mean_ms_per_token"] > 0.0
    assert decode["decode_tok_per_sec"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_benchmarks_and_memory(tiny_model, tiny_config):
    device = torch.device("cuda")
    tiny_model.to(device)

    # Memory profiling
    mem = profile_memory(tiny_model, batch_size=1, seq_len=64, device=device, dtype=torch.bfloat16)
    assert mem["weights_mb"] > 0.0
    assert mem["peak_vram_mb"] > 0.0

    # Latency benchmarking
    lat = benchmark_latency(
        tiny_model,
        batch_size=1,
        context_length=64,
        num_decode_tokens=4,
        device=device,
        dtype=torch.bfloat16,
        warmup_runs=1,
        num_runs=2,
    )
    assert lat["prefill_mean_ms"] > 0.0
    assert lat["decode_mean_ms_per_token"] > 0.0
