"""Unified 5-Way Scaling Matrix Benchmark Suite (1K to 32K Tokens).

Implements Phase 8 of the SparDA specification (Section 15, 18, 19):
Compares 5 foundational and experimental architectures across context scaling:
1. Dense GQA Baseline (dense)
2. Static Local + Sink Sparse (sparse / static_sparse)
3. Forecast-Guided GPU Sparse (forecast)
4. Synchronous Forecast KV Offload (offload)
5. Asynchronous Prefetched KV Offload (async_offload)

Across context lengths:
1,024 (1K), 2,048 (2K), 4,096 (4K), 8,192 (8K), 16,384 (16K), 32,768 (32K).

Measures:
- Prefill Latency (ms) & Throughput (tokens/sec)
- Autoregressive Decode Latency (ms/token) & Throughput (tokens/sec)
- Effective GPU KV-Cache Memory (MB) & Peak Allocated VRAM (GB)
- PCIe Transfer Overhead & Overlap Percentage (%)
- Crossover Points where sparse/offload/async surpass dense execution.
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
from model.attention.base import KVCache
from model.attention.offload import OffloadedKVCache
from model.attention.async_offload import AsyncOffloadedKVCache
from evaluation.memory import MemoryProfiler, calculate_kv_cache_bytes
from evaluation.latency import LatencyBenchmarker
from training.checkpoint import load_checkpoint


BACKEND_CONFIGS = {
    "dense": {
        "name": "Dense GQA Baseline",
        "backend": "dense",
        "sparse_mode": "static",
        "cache_type": "standard",
    },
    "static_sparse": {
        "name": "Static Local + Sink Sparse",
        "backend": "static_sparse",
        "sparse_mode": "static",
        "cache_type": "standard",
    },
    "forecast": {
        "name": "Forecast-Guided GPU Sparse",
        "backend": "forecast",
        "sparse_mode": "forecast",
        "cache_type": "standard",
    },
    "offload": {
        "name": "Synchronous Forecast Offload",
        "backend": "offload",
        "sparse_mode": "forecast",
        "cache_type": "offload",
    },
    "async_offload": {
        "name": "Asynchronous Prefetched Offload",
        "backend": "async_offload",
        "sparse_mode": "forecast",
        "cache_type": "async_offload",
    },
}


def build_model_for_backend(
    base_config: ModelConfig,
    backend_key: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> SparDAResearchModel:
    """Builds a model instance configured for a specific attention backend."""
    info = BACKEND_CONFIGS[backend_key]
    cfg_dict = vars(base_config).copy()
    cfg_dict["attention_backend"] = info["backend"]
    cfg_dict["sparse_mode"] = info["sparse_mode"]
    
    cfg = ModelConfig(**cfg_dict)
    model = SparDAResearchModel(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            load_checkpoint(checkpoint_path, model)
        except Exception as e:
            print(f"  [Warning] Could not load checkpoint into {backend_key}: {e}")

    model.eval()
    return model


def create_kv_caches(
    backend_key: str,
    model: SparDAResearchModel,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Any]:
    """Instantiates the appropriate KV cache hierarchy for the given backend."""
    cfg = model.config
    info = BACKEND_CONFIGS[backend_key]
    cache_type = info["cache_type"]

    if cache_type == "standard":
        return [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(cfg.n_layer)
        ]
    elif cache_type == "offload":
        return [
            OffloadedKVCache(
                max_batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                gpu_device=device,
                dtype=dtype,
                pin_memory=True,
            )
            for _ in range(cfg.n_layer)
        ]
    elif cache_type == "async_offload":
        return [
            AsyncOffloadedKVCache(
                max_batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                gpu_device=device,
                dtype=dtype,
                pin_memory=True,
            )
            for _ in range(cfg.n_layer)
        ]
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


def profile_single_backend_context(
    model: SparDAResearchModel,
    backend_key: str,
    ctx_len: int,
    batch_size: int = 1,
    num_decode_tokens: int = 16,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    warmup_steps: int = 2,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Profiles prefill and autoregressive decode performance for a single backend and context."""
    if device is None:
        device = next(model.parameters()).device

    is_cuda = device.type == "cuda"
    cfg = model.config
    max_seq_len = ctx_len + num_decode_tokens + 32
    info = BACKEND_CONFIGS[backend_key]

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Prefill Profiling
    prompt_ids = torch.randint(
        low=0, high=cfg.vocab_size, size=(batch_size, ctx_len), device=device, dtype=torch.long
    )

    # Prefill Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
                _ = model(prompt_ids)
        if is_cuda:
            torch.cuda.synchronize(device)

    # Timed Prefill
    prefill_times_ms: List[float] = []
    for _ in range(num_runs):
        if is_cuda:
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    _ = model(prompt_ids)
            end_event.record()
            torch.cuda.synchronize(device)
            prefill_times_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(prompt_ids)
            t1 = time.perf_counter()
            prefill_times_ms.append((t1 - t0) * 1000.0)

    prefill_mean_ms = float(torch.tensor(prefill_times_ms).mean().item())
    prefill_tok_per_sec = (batch_size * ctx_len) / max(prefill_mean_ms / 1000.0, 1e-6)

    # 2. Decode Profiling (Autoregressive Token Generation with Cache)
    kv_caches = create_kv_caches(backend_key, model, batch_size, max_seq_len, device, dtype)

    # Prefill prompt into cache
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
            _ = model(prompt_ids, kv_caches=kv_caches)
    if is_cuda:
        torch.cuda.synchronize(device)

    # Warmup decode
    next_token = torch.randint(0, cfg.vocab_size, (batch_size, 1), device=device, dtype=torch.long)
    for _ in range(warmup_steps):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=is_cuda):
                out = model(next_token, kv_caches=kv_caches)
                next_token = torch.argmax(out["logits"][:, -1:, :], dim=-1)
    if is_cuda:
        torch.cuda.synchronize(device)

    # Reset metrics / counters if applicable
    if info["cache_type"] in ("offload", "async_offload"):
        for c in kv_caches:
            c.total_bytes_transferred = 0

    # Timed Decode steps
    decode_step_times_ms: List[float] = []
    layer_overlap_pcts: List[float] = []
    layer_hidden_latencies: List[float] = []

    for _ in range(num_decode_tokens):
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

        decode_step_times_ms.append((t_step_end - t_step_start) * 1000.0)

        # Collect async overlap stats if present
        if backend_key == "async_offload":
            for layer in model.layers:
                prof = getattr(layer.attention, "last_profile", None)
                if prof and "overlap_pct" in prof:
                    layer_overlap_pcts.append(prof["overlap_pct"])
                    layer_hidden_latencies.append(prof.get("latency_hidden_ms", prof.get("transfer_hidden_ms", 0.0)))

    decode_mean_ms = float(torch.tensor(decode_step_times_ms).mean().item())
    decode_tok_per_sec = 1000.0 / max(decode_mean_ms, 1e-6)

    # 3. Memory Profiling
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if is_cuda else 0.0
    total_gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) if is_cuda else 16.0
    headroom_vram_gb = max(0.0, total_gpu_mem_gb - peak_vram_gb)

    # Calculate actual GPU KV cache bytes
    if info["cache_type"] == "standard":
        # For standard GPU KV cache
        gpu_kv_tokens = ctx_len + num_decode_tokens
        gpu_kv_bytes = calculate_kv_cache_bytes(
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            batch_size=batch_size,
            seq_len=gpu_kv_tokens,
            dtype=dtype,
        )
    else:
        # For offloaded KV cache: only max_sel_blocks reside in GPU VRAM
        max_sel_blocks = min(
            (ctx_len + cfg.block_size - 1) // cfg.block_size,
            cfg.top_k_blocks + 2,
        )
        gpu_kv_tokens = max_sel_blocks * cfg.block_size
        gpu_kv_bytes = calculate_kv_cache_bytes(
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            batch_size=batch_size,
            seq_len=gpu_kv_tokens,
            dtype=dtype,
        )

    gpu_kv_mb = gpu_kv_bytes / (1024 ** 2)

    # Transfer statistics for offload backends
    total_transferred_mb = 0.0
    transfer_kb_per_step = 0.0
    if info["cache_type"] in ("offload", "async_offload"):
        total_bytes = sum(c.total_bytes_transferred for c in kv_caches)
        total_transferred_mb = total_bytes / (1024 ** 2)
        transfer_kb_per_step = (total_bytes / (1024 * max(num_decode_tokens, 1)))

    overlap_mean_pct = float(torch.tensor(layer_overlap_pcts).mean().item()) if layer_overlap_pcts else 0.0
    hidden_mean_ms = float(torch.tensor(layer_hidden_latencies).mean().item()) if layer_hidden_latencies else 0.0

    # Cleanup
    del kv_caches, prompt_ids, next_token
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "backend": backend_key,
        "backend_name": info["name"],
        "context_length": ctx_len,
        "batch_size": batch_size,
        "prefill_mean_ms": prefill_mean_ms,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_mean_ms_per_token": decode_mean_ms,
        "decode_tok_per_sec": decode_tok_per_sec,
        "gpu_kv_mb": gpu_kv_mb,
        "peak_vram_gb": peak_vram_gb,
        "headroom_vram_gb": headroom_vram_gb,
        "transfer_kb_per_step": transfer_kb_per_step,
        "total_transferred_mb": total_transferred_mb,
        "overlap_mean_pct": overlap_mean_pct,
        "hidden_mean_ms": hidden_mean_ms,
    }


def run_full_scaling_matrix(
    base_config: ModelConfig,
    context_lengths: List[int],
    backends: Optional[List[str]] = None,
    batch_size: int = 1,
    num_decode_tokens: int = 16,
    device_str: str = "cuda",
    dtype_str: str = "bf16",
    checkpoint_path: Optional[str] = None,
    warmup_steps: int = 2,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Executes the full 5-way scaling sweep across all specified context lengths."""
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    dtype = torch.bfloat16 if dtype_str == "bf16" and torch.cuda.is_bf16_supported() else torch.float16

    if backends is None:
        backends = ["dense", "static_sparse", "forecast", "offload", "async_offload"]

    print("=" * 116)
    print("SPARDA FULL SCALING MATRIX BENCHMARK SUITE (PHASE 8)")
    print(f"Device: {device} | Dtype: {dtype} | Batch Size: {batch_size}")
    print(f"Context Scaling Matrix: {context_lengths} tokens")
    print(f"Backends ({len(backends)}): {', '.join(backends)}")
    print("=" * 116)

    matrix_results: Dict[str, List[Dict[str, Any]]] = {b: [] for b in backends}
    dense_baseline: Dict[int, Dict[str, Any]] = {}

    for b_idx, backend_key in enumerate(backends, 1):
        info = BACKEND_CONFIGS[backend_key]
        print(f"\n[{b_idx}/{len(backends)}] Initializing Model for Backend: {info['name']} ({backend_key}) ...")
        
        model = build_model_for_backend(base_config, backend_key, device, checkpoint_path)
        
        for ctx in context_lengths:
            print(f"  --> Context {ctx:,} tokens ...", end="", flush=True)
            res = profile_single_backend_context(
                model=model,
                backend_key=backend_key,
                ctx_len=ctx,
                batch_size=batch_size,
                num_decode_tokens=num_decode_tokens,
                device=device,
                dtype=dtype,
                warmup_steps=warmup_steps,
                num_runs=num_runs,
            )
            matrix_results[backend_key].append(res)
            if backend_key == "dense":
                dense_baseline[ctx] = res

            print(
                f" Prefill: {res['prefill_mean_ms']:.1f}ms ({res['prefill_tok_per_sec']:,.0f} tok/s) | "
                f"Decode: {res['decode_mean_ms_per_token']:.2f}ms/tok ({res['decode_tok_per_sec']:.1f} tok/s) | "
                f"KV: {res['gpu_kv_mb']:.1f}MB | Peak VRAM: {res['peak_vram_gb']:.2f}GB"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Calculate comparative ratios vs Dense
    comparative_matrix = []
    for ctx in context_lengths:
        d_res = dense_baseline.get(ctx, {})
        d_prefill_ms = d_res.get("prefill_mean_ms", 1.0)
        d_decode_ms = d_res.get("decode_mean_ms_per_token", 1.0)
        d_kv_mb = d_res.get("gpu_kv_mb", 1.0)

        for b_key in backends:
            res = next((r for r in matrix_results[b_key] if r["context_length"] == ctx), None)
            if res:
                entry = res.copy()
                entry["prefill_speedup_vs_dense"] = d_prefill_ms / max(entry["prefill_mean_ms"], 1e-6)
                entry["decode_speedup_vs_dense"] = d_decode_ms / max(entry["decode_mean_ms_per_token"], 1e-6)
                entry["kv_reduction_vs_dense"] = d_kv_mb / max(entry["gpu_kv_mb"], 1e-6)
                comparative_matrix.append(entry)

    # Pinpoint crossover points
    crossover_summary = identify_crossover_points(comparative_matrix, backends, context_lengths)

    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "total_vram_gb": torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) if device.type == "cuda" else 0.0,
            "dtype": str(dtype),
        },
        "parameters": {
            "batch_size": batch_size,
            "num_decode_tokens": num_decode_tokens,
            "context_lengths": context_lengths,
            "backends": backends,
        },
        "matrix_results": matrix_results,
        "comparative_matrix": comparative_matrix,
        "crossover_summary": crossover_summary,
    }

    return full_output


def identify_crossover_points(
    comparative_matrix: List[Dict[str, Any]],
    backends: List[str],
    context_lengths: List[int],
) -> Dict[str, Any]:
    """Analyzes comparative metrics to pinpoint crossover points where backends outperform dense."""
    crossovers = {}

    for b in backends:
        if b == "dense":
            continue

        b_entries = sorted([e for e in comparative_matrix if e["backend"] == b], key=lambda x: x["context_length"])
        
        prefill_crossover = None
        decode_crossover = None
        kv_saving_50pct = None
        kv_saving_90pct = None

        for e in b_entries:
            ctx = e["context_length"]
            # Prefill speedup
            if e["prefill_speedup_vs_dense"] > 1.0 and prefill_crossover is None:
                prefill_crossover = ctx
            # Decode speedup
            if e["decode_speedup_vs_dense"] > 1.0 and decode_crossover is None:
                decode_crossover = ctx
            # KV reduction
            if e["kv_reduction_vs_dense"] >= 2.0 and kv_saving_50pct is None:
                kv_saving_50pct = ctx
            if e["kv_reduction_vs_dense"] >= 10.0 and kv_saving_90pct is None:
                kv_saving_90pct = ctx

        crossovers[b] = {
            "prefill_speedup_crossover_tokens": prefill_crossover,
            "decode_speedup_crossover_tokens": decode_crossover,
            "kv_50pct_reduction_crossover_tokens": kv_saving_50pct,
            "kv_90pct_reduction_crossover_tokens": kv_saving_90pct,
            "max_kv_reduction_ratio": max(e["kv_reduction_vs_dense"] for e in b_entries) if b_entries else 1.0,
            "peak_prefill_tok_per_sec": max(e["prefill_tok_per_sec"] for e in b_entries) if b_entries else 0.0,
            "peak_decode_tok_per_sec": max(e["decode_tok_per_sec"] for e in b_entries) if b_entries else 0.0,
        }

    return crossovers


def generate_matrix_markdown_tables(results: Dict[str, Any]) -> str:
    """Generates comprehensive markdown tables for documentation and reports."""
    hw = results["hardware"]
    contexts = results["parameters"]["context_lengths"]
    backends = results["parameters"]["backends"]
    comp = results["comparative_matrix"]

    md = []
    md.append(f"# SparDA Full Scaling Performance Matrix (1K to 32K)\n")
    md.append(f"- **Evaluated Device**: {hw['gpu_name']} ({hw['total_vram_gb']:.2f} GB VRAM)")
    md.append(f"- **Precision**: `{hw['dtype']}` | **Batch Size**: {results['parameters']['batch_size']}\n")

    # Table 1: Prefill Throughput & Latency Matrix
    md.append("### 1. Prefill Latency (ms) & Throughput (tokens/sec)\n")
    header = "| Context | " + " | ".join(BACKEND_CONFIGS[b]["name"] for b in backends) + " |"
    sep = "|:---:| " + " | ".join("---:" for _ in backends) + " |"
    md.append(header)
    md.append(sep)

    for ctx in contexts:
        row = [f"**{ctx:,}**"]
        for b in backends:
            e = next((x for x in comp if x["context_length"] == ctx and x["backend"] == b), None)
            if e:
                speedup_str = f" ({e['prefill_speedup_vs_dense']:.2f}x)" if b != "dense" else ""
                row.append(f"{e['prefill_mean_ms']:.1f} ms<br>*{e['prefill_tok_per_sec']:,.0f} tok/s*{speedup_str}")
            else:
                row.append("N/A")
        md.append("| " + " | ".join(row) + " |")

    # Table 2: Decode Latency & Throughput Matrix
    md.append("\n### 2. Autoregressive Decode Latency (ms/token) & Throughput (tokens/sec)\n")
    md.append(header)
    md.append(sep)

    for ctx in contexts:
        row = [f"**{ctx:,}**"]
        for b in backends:
            e = next((x for x in comp if x["context_length"] == ctx and x["backend"] == b), None)
            if e:
                speedup_str = f" ({e['decode_speedup_vs_dense']:.2f}x)" if b != "dense" else ""
                row.append(f"{e['decode_mean_ms_per_token']:.2f} ms<br>*{e['decode_tok_per_sec']:.1f} tok/s*{speedup_str}")
            else:
                row.append("N/A")
        md.append("| " + " | ".join(row) + " |")

    # Table 3: GPU KV-Cache Memory (MB) & Peak VRAM (GB)
    md.append("\n### 3. Effective GPU KV-Cache (MB) & Peak Allocated VRAM (GB)\n")
    md.append(header)
    md.append(sep)

    for ctx in contexts:
        row = [f"**{ctx:,}**"]
        for b in backends:
            e = next((x for x in comp if x["context_length"] == ctx and x["backend"] == b), None)
            if e:
                reduc_str = f" (-{(1 - 1/max(e['kv_reduction_vs_dense'], 1e-6))*100:.0f}%)" if b != "dense" else ""
                row.append(f"{e['gpu_kv_mb']:.1f} MB{reduc_str}<br>*{e['peak_vram_gb']:.2f} GB peak*")
            else:
                row.append("N/A")
        md.append("| " + " | ".join(row) + " |")

    # Table 4: Crossover Points Summary
    md.append("\n### 4. Empirical Architectural Crossover Points\n")
    md.append("| Attention Backend | Prefill Crossover | Decode Crossover | 50% KV Savings | 90% KV Savings | Max KV Reduction |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    crossovers = results["crossover_summary"]
    for b, stats in crossovers.items():
        name = BACKEND_CONFIGS[b]["name"]
        p_cross = f"{stats['prefill_speedup_crossover_tokens']:,} tokens" if stats['prefill_speedup_crossover_tokens'] else "None (Bound by gather)"
        d_cross = f"{stats['decode_speedup_crossover_tokens']:,} tokens" if stats['decode_speedup_crossover_tokens'] else "None"
        kv_50 = f"{stats['kv_50pct_reduction_crossover_tokens']:,} tokens" if stats['kv_50pct_reduction_crossover_tokens'] else "N/A"
        kv_90 = f"{stats['kv_90pct_reduction_crossover_tokens']:,} tokens" if stats['kv_90pct_reduction_crossover_tokens'] else "N/A"
        md.append(f"| **{name}** | {p_cross} | {d_cross} | {kv_50} | {kv_90} | **{stats['max_kv_reduction_ratio']:.1f}x** |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Unified 5-Way Scaling Matrix Benchmark Suite (1K to 32K)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Base model configuration YAML")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768], help="Context lengths to sweep")
    parser.add_argument("--backends", type=str, nargs="+", default=["dense", "static_sparse", "forecast", "offload", "async_offload"], help="Backends to profile")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--decode-tokens", type=int, default=16, help="Number of decode steps to evaluate")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=5, help="Measurement iterations")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/base/model_step_100.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--dtype", type=str, default="bf16", help="Precision: bf16 or fp16")
    parser.add_argument("--output-json", type=str, default="benchmark_results/scaling_matrix_results.json", help="Output JSON path")
    parser.add_argument("--output-markdown", type=str, default="benchmark_results/scaling_matrix_summary.md", help="Output Markdown path")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg_dict["model"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    results = run_full_scaling_matrix(
        base_config=model_cfg,
        context_lengths=args.contexts,
        backends=args.backends,
        batch_size=args.batch_size,
        num_decode_tokens=args.decode_tokens,
        device_str=args.device,
        dtype_str=args.dtype,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        warmup_steps=args.warmup_steps,
        num_runs=args.num_runs,
    )

    # Save JSON results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifact Saved] JSON Results: {args.output_json}")

    # Generate and save Markdown tables
    md_content = generate_matrix_markdown_tables(results)
    with open(args.output_markdown, "w") as f:
        f.write(md_content)
    print(f"[Artifact Saved] Markdown Tables: {args.output_markdown}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
