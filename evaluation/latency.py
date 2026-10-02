"""High-precision latency benchmarking using synchronized CUDA events."""

import time
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn

from model.attention.base import KVCache
from model.config import ModelConfig


class LatencyBenchmarker:
    """Measures prefill and autoregressive decode latencies with sub-millisecond precision."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"

    def _sync(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def benchmark_prefill(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_len: int,
        dtype: torch.dtype = torch.bfloat16,
        warmup_runs: int = 3,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Measures prefill latency (processing prompt sequence of length prompt_len)."""
        cfg = getattr(model, "config", None)
        assert cfg is not None, "Model must have a config attribute"

        model.eval()
        dummy_input = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=(batch_size, prompt_len),
            device=self.device,
            dtype=torch.long,
        )

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype, enabled=self.is_cuda):
                    _ = model(dummy_input)
            self._sync()

        latencies_ms: List[float] = []

        if self.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(num_runs):
                self._sync()
                start_event.record()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype):
                        _ = model(dummy_input)
                end_event.record()
                self._sync()
                latencies_ms.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies_ms))
        median_ms = float(np.median(latencies_ms))
        std_ms = float(np.std(latencies_ms))
        min_ms = float(np.min(latencies_ms))
        max_ms = float(np.max(latencies_ms))

        # Tokens per second during prefill = (batch_size * prompt_len) / (mean_ms / 1000)
        tok_per_sec = (batch_size * prompt_len) / max(mean_ms / 1000.0, 1e-6)

        return {
            "prefill_mean_ms": mean_ms,
            "prefill_median_ms": median_ms,
            "prefill_std_ms": std_ms,
            "prefill_min_ms": min_ms,
            "prefill_max_ms": max_ms,
            "prefill_tok_per_sec": tok_per_sec,
            "time_to_first_token_ms": mean_ms,
        }

    def benchmark_decode(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_len: int,
        num_decode_tokens: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        warmup_runs: int = 2,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """Measures incremental token decoding latency with KVCache at a given prompt context length."""
        cfg = getattr(model, "config", None)
        assert cfg is not None, "Model must have a config attribute"

        max_seq_len = prompt_len + num_decode_tokens + 16
        model.eval()

        def run_decode_sequence() -> float:
            # Create fresh KV caches for all layers
            kv_caches = [
                KVCache(
                    max_batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    n_kv_head=cfg.n_kv_head,
                    head_dim=cfg.head_dim,
                    device=self.device,
                    dtype=dtype,
                )
                for _ in range(cfg.n_layer)
            ]

            # 1. Prefill step into KV cache
            prompt = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(batch_size, prompt_len),
                device=self.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype, enabled=self.is_cuda):
                    _ = model(prompt, kv_caches=kv_caches)
            self._sync()

            # 2. Time incremental decoding of num_decode_tokens
            next_token = torch.randint(
                low=0,
                high=cfg.vocab_size,
                size=(batch_size, 1),
                device=self.device,
                dtype=torch.long,
            )

            if self.is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype):
                        for _ in range(num_decode_tokens):
                            out = model(next_token, kv_caches=kv_caches)
                            next_token = torch.argmax(out["logits"][:, -1:, :], dim=-1)
                end_event.record()
                self._sync()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(num_decode_tokens):
                        out = model(next_token, kv_caches=kv_caches)
                        next_token = torch.argmax(out["logits"][:, -1:, :], dim=-1)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

            del kv_caches, prompt, next_token
            return elapsed_ms

        # Warmup
        for _ in range(warmup_runs):
            run_decode_sequence()

        # Measurement runs
        total_decode_latencies_ms: List[float] = []
        for _ in range(num_runs):
            total_decode_latencies_ms.append(run_decode_sequence())

        # Compute per-token statistics
        per_token_ms = [t / num_decode_tokens for t in total_decode_latencies_ms]

        mean_token_ms = float(np.mean(per_token_ms))
        median_token_ms = float(np.median(per_token_ms))
        std_token_ms = float(np.std(per_token_ms))
        min_token_ms = float(np.min(per_token_ms))
        max_token_ms = float(np.max(per_token_ms))

        # Decode tokens per second = (batch_size * num_decode_tokens) / (total_time_ms / 1000)
        mean_total_ms = float(np.mean(total_decode_latencies_ms))
        tok_per_sec = (batch_size * num_decode_tokens) / max(mean_total_ms / 1000.0, 1e-6)

        return {
            "decode_mean_ms_per_token": mean_token_ms,
            "decode_median_ms_per_token": median_token_ms,
            "decode_std_ms_per_token": std_token_ms,
            "decode_min_ms_per_token": min_token_ms,
            "decode_max_ms_per_token": max_token_ms,
            "decode_tok_per_sec": tok_per_sec,
            "num_decode_tokens": num_decode_tokens,
        }

    def benchmark_context(
        self,
        model: nn.Module,
        batch_size: int,
        context_length: int,
        num_decode_tokens: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        warmup_runs: int = 3,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """Runs unified prefill and decode benchmark for a given context length."""
        prefill_stats = self.benchmark_prefill(
            model=model,
            batch_size=batch_size,
            prompt_len=context_length,
            dtype=dtype,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
        )
        decode_stats = self.benchmark_decode(
            model=model,
            batch_size=batch_size,
            prompt_len=context_length,
            num_decode_tokens=num_decode_tokens,
            dtype=dtype,
            warmup_runs=max(1, warmup_runs // 2),
            num_runs=max(3, num_runs // 2),
        )

        return {
            "context_length": context_length,
            "batch_size": batch_size,
            **prefill_stats,
            **decode_stats,
        }


def benchmark_latency(
    model: nn.Module,
    batch_size: int,
    context_length: int,
    num_decode_tokens: int = 16,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    warmup_runs: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Helper function to run unified prefill & decode latency benchmark."""
    if device is None:
        device = next(model.parameters()).device
    benchmarker = LatencyBenchmarker(device)
    return benchmarker.benchmark_context(
        model=model,
        batch_size=batch_size,
        context_length=context_length,
        num_decode_tokens=num_decode_tokens,
        dtype=dtype,
        warmup_runs=warmup_runs,
        num_runs=num_runs,
    )
