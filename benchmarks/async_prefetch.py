"""Asynchronous Forecast-Guided KV Cache Prefetching Benchmark Suite.

Implements Section 12 of the SparDA specification:
Measures:
1. Overlap percentage (%) and transfer latency hidden (ms) behind Layer L compute.
2. Stalled wait latency (ms): Time compute stream waits for transfer stream.
3. PCIe transfer volume across PCIe (KB per decode step and cumulative MB).
4. Full decode step latency (ms/step) and decode throughput (tokens/sec).
5. Side-by-side comparison with synchronous KV offloading (Phase 5).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import SparDAResearchModel
from model.attention.async_offload import AsyncOffloadedKVCache
from evaluation.memory import MemoryProfiler, calculate_kv_cache_bytes
from training.checkpoint import load_checkpoint


def run_async_prefetch_benchmark(
    model: SparDAResearchModel,
    context_lengths: List[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    num_decode_tokens: int = 16,
    warmup_steps: int = 3,
) -> Dict[str, Any]:
    """Profiles asynchronous KV cache prefetching across context lengths."""
    if device is None:
        device = next(model.parameters()).device

    is_cuda = device.type == "cuda"
    cfg = model.config

    results = []

    print("=" * 110)
    print(f"SPARDA ASYNCHRONOUS KV PREFETCH BENCHMARK (Device: {device}, Dtype: {dtype})")
    print(f"Overlapping CPU -> GPU PCIe DMA Transfers with GPU Attention & FFN Computation")
    print("=" * 110)

    for ctx in context_lengths:
        print(f"\n>>> Profiling Context: {ctx:,} tokens (Decode Steps: {num_decode_tokens}) ...")
        torch.cuda.empty_cache()

        # 1. Theoretical & Real VRAM Calculations
        analytical_bytes = calculate_kv_cache_bytes(
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            batch_size=batch_size,
            seq_len=ctx,
            dtype=dtype,
        )
        dense_kv_mb = analytical_bytes / (1024 ** 2)

        max_sel_blocks = min((ctx + cfg.block_size - 1) // cfg.block_size, cfg.top_k_blocks + 2)
        offloaded_tokens_gpu = max_sel_blocks * cfg.block_size
        offload_gpu_kv_bytes = calculate_kv_cache_bytes(
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            batch_size=batch_size,
            seq_len=offloaded_tokens_gpu,
            dtype=dtype,
        )
        offload_gpu_kv_mb = offload_gpu_kv_bytes / (1024 ** 2)
        vram_reduction_ratio = dense_kv_mb / max(offload_gpu_kv_mb, 1e-6)

        # 2. Setup AsyncOffloadedKVCache per layer
        kv_caches = [
            AsyncOffloadedKVCache(
                max_batch_size=batch_size,
                max_seq_len=ctx + num_decode_tokens + 32,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                gpu_device=device,
                dtype=dtype,
                pin_memory=True,
            )
            for _ in range(cfg.n_layer)
        ]

        # 3. Prompt Prefill into CPU Cache
        prompt = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=(batch_size, ctx),
            device=device,
            dtype=torch.long,
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
                _ = model(prompt, kv_caches=kv_caches)
        if is_cuda:
            torch.cuda.synchronize(device)

        # Reset counters for decode profiling
        for c in kv_caches:
            c.total_bytes_transferred = 0

        # 4. Warmup decode
        next_token = torch.randint(0, cfg.vocab_size, (batch_size, 1), device=device, dtype=torch.long)
        for _ in range(warmup_steps):
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
                    out = model(next_token, kv_caches=kv_caches)
                    next_token = torch.argmax(out["logits"][:, -1:, :], dim=-1)
        if is_cuda:
            torch.cuda.synchronize(device)

        # Reset counters after warmup
        for c in kv_caches:
            c.total_bytes_transferred = 0

        # 5. Timed Decode with Overlap & Latency Breakdown
        step_latencies_ms = []
        layer_wait_ms = []
        layer_trans_ms = []
        layer_attn_ms = []
        layer_hidden_ms = []
        layer_launch_ms = []

        for step in range(num_decode_tokens):
            if is_cuda:
                torch.cuda.synchronize(device)
            t_step_start = time.perf_counter()

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
                    out = model(next_token, kv_caches=kv_caches)
                    next_token = torch.argmax(out["logits"][:, -1:, :], dim=-1)

            if is_cuda:
                torch.cuda.synchronize(device)
            t_step_end = time.perf_counter()

            step_latencies_ms.append((t_step_end - t_step_start) * 1000.0)

            # Collect metrics across layers
            for layer in model.layers:
                attn = layer.attention
                prof = getattr(attn, "last_profile", None)
                if prof:
                    layer_wait_ms.append(prof["wait_ms"])
                    layer_trans_ms.append(prof["transfer_ms"])
                    layer_attn_ms.append(prof["attention_ms"])
                    layer_hidden_ms.append(prof["latency_hidden_ms"])
                    layer_launch_ms.append(prof["transfer_launch_ms"])

        total_bytes = sum(c.total_bytes_transferred for c in kv_caches)
        bytes_per_step = total_bytes / max(num_decode_tokens, 1)
        kb_per_step = bytes_per_step / 1024.0

        mean_step_ms = sum(step_latencies_ms) / len(step_latencies_ms)
        mean_wait_ms = (sum(layer_wait_ms) / len(layer_wait_ms)) * cfg.n_layer if layer_wait_ms else 0.0
        mean_trans_ms = (sum(layer_trans_ms) / len(layer_trans_ms)) * cfg.n_layer if layer_trans_ms else 0.0
        mean_attn_ms = (sum(layer_attn_ms) / len(layer_attn_ms)) * cfg.n_layer if layer_attn_ms else 0.0
        mean_hidden_ms = (sum(layer_hidden_ms) / len(layer_hidden_ms)) * cfg.n_layer if layer_hidden_ms else 0.0
        mean_launch_ms = (sum(layer_launch_ms) / len(layer_launch_ms)) * cfg.n_layer if layer_launch_ms else 0.0

        tok_per_sec = 1000.0 / max(mean_step_ms, 1e-6)
        overlap_pct = min(100.0, max(0.0, (1.0 - (mean_wait_ms / max(mean_trans_ms, 1e-6))) * 100.0))

        entry = {
            "context_length": ctx,
            "dense_gpu_kv_mb": dense_kv_mb,
            "offload_gpu_kv_mb": offload_gpu_kv_mb,
            "vram_reduction_ratio": vram_reduction_ratio,
            "transfer_kb_per_step": kb_per_step,
            "launch_latency_ms": mean_launch_ms,
            "transfer_latency_ms": mean_trans_ms,
            "wait_latency_ms": mean_wait_ms,
            "hidden_latency_ms": mean_hidden_ms,
            "overlap_percentage": overlap_pct,
            "attention_latency_ms": mean_attn_ms,
            "total_decode_step_ms": mean_step_ms,
            "decode_tok_per_sec": tok_per_sec,
        }
        results.append(entry)

        print(f"  Context: {ctx:,} | Dense KV: {dense_kv_mb:.2f} MB -> Async Offload: {offload_gpu_kv_mb:.2f} MB ({vram_reduction_ratio:.1f}x reduction)")
        print(f"  Transfer Volume: {kb_per_step:,.1f} KB/step")
        print(f"  DMA Launch Latency:  {mean_launch_ms:.2f} ms")
        print(f"  Transfer Latency:    {mean_trans_ms:.2f} ms")
        print(f"  Stalled Wait Time:   {mean_wait_ms:.2f} ms")
        print(f"  Hidden Latency:      {mean_hidden_ms:.2f} ms")
        print(f"  Stream Overlap:      {overlap_pct:.1f}%")
        print(f"  Attention Latency:   {mean_attn_ms:.2f} ms")
        print(f"  Total Decode Latency:{mean_step_ms:.2f} ms/tok -> Throughput: {tok_per_sec:.2f} tok/s")

    print("\n" + "=" * 110)
    print("SUMMARY OF ASYNCHRONOUS PREFETCH RESULTS")
    print("=" * 110)
    header = f"{'Context':>8} | {'Dense KV':>10} | {'Async KV':>10} | {'VRAM Red.':>10} | {'Transfer/Step':>14} | {'Transfer':>10} | {'Wait':>8} | {'Overlap':>9} | {'Total Decode':>14} | {'Tok/s':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['context_length']:>8,d} | "
            f"{r['dense_gpu_kv_mb']:>9.2f}M | "
            f"{r['offload_gpu_kv_mb']:>9.2f}M | "
            f"{r['vram_reduction_ratio']:>9.1f}x | "
            f"{r['transfer_kb_per_step']:>11.1f} KB | "
            f"{r['transfer_latency_ms']:>8.2f}ms | "
            f"{r['wait_latency_ms']:>6.2f}ms | "
            f"{r['overlap_percentage']:>8.1f}% | "
            f"{r['total_decode_step_ms']:>12.2f}ms | "
            f"{r['decode_tok_per_sec']:>7.2f}"
        )
    print("=" * 110)

    return {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "n_layer": cfg.n_layer,
        "block_size": cfg.block_size,
        "top_k_blocks": cfg.top_k_blocks,
        "benchmarks": results,
    }


def main():
    parser = argparse.ArgumentParser(description="SparDA Asynchronous KV Prefetching Benchmark")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base/model_step_100.pt", help="Checkpoint path")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096], help="Context lengths to benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--decode-tokens", type=int, default=16, help="Number of decode tokens to profile")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup decode steps")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for KV blocks")
    parser.add_argument("--top-k-blocks", type=int, default=4, help="Top-K blocks to prefetch per head")
    parser.add_argument("--output-json", type=str, default="docs/async_benchmark.json", help="Path to save benchmark JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # Load base config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Configure for async offload
    model_dict = cfg_dict.get("model", cfg_dict)
    model_dict["attention_backend"] = "async_offload"
    model_dict["block_size"] = args.block_size
    model_dict["top_k_blocks"] = args.top_k_blocks
    model_dict["context_length"] = max(args.contexts) + args.decode_tokens + 64

    config = ModelConfig(**model_dict)
    model = SparDAResearchModel(config).to(device)

    # Load weights
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model=model, device=device)
    else:
        print(f"Checkpoint not found at {args.checkpoint}, using initialized weights.")

    model.eval()

    results = run_async_prefetch_benchmark(
        model=model,
        context_lengths=args.contexts,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        num_decode_tokens=args.decode_tokens,
        warmup_steps=args.warmup_steps,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Benchmark results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
