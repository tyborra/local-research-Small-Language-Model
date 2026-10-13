"""Head-to-head Wall-clock Benchmark: Dense Attention vs. Sparse Attention.

Rigorously measures Section 13 performance principles:
- Compares real wall-clock prefill and autoregressive decode latencies.
- Profiles throughput (tokens/sec) and peak/KV VRAM consumption across context lengths (1K to 8K+).
- Evaluates where physical block gathering and top-K selection achieve real acceleration
  vs. where kernel overhead/gather cost dominates at smaller contexts.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import SparDAResearchModel
from evaluation.memory import MemoryProfiler
from evaluation.latency import LatencyBenchmarker
from training.checkpoint import load_checkpoint


def run_comparative_benchmark(
    dense_model: SparDAResearchModel,
    sparse_model: SparDAResearchModel,
    context_lengths: List[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    num_decode_tokens: int = 16,
    warmup_runs: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Profiles both dense and sparse models across context lengths."""
    if device is None:
        device = next(dense_model.parameters()).device

    profiler = MemoryProfiler(device)
    benchmarker = LatencyBenchmarker(device)

    comparisons = []

    print("=" * 96)
    print(f"SPARDA HEAD-TO-HEAD BENCHMARK: DENSE vs. SPARSE (Device: {device}, Dtype: {dtype})")
    print(f"Contexts to evaluate: {context_lengths} | Batch Size: {batch_size}")
    print("=" * 96)

    for ctx in context_lengths:
        print(f"\n>>> Evaluating Context Length: {ctx:,} tokens ...")

        # 1. Dense Profile
        print("    [1/2] Profiling Dense GQA Baseline...")
        torch.cuda.empty_cache()
        dense_mem = profiler.profile_context_vram(dense_model, batch_size=batch_size, seq_len=ctx, dtype=dtype)
        dense_lat = benchmarker.benchmark_context(
            dense_model, batch_size=batch_size, context_length=ctx,
            num_decode_tokens=num_decode_tokens, dtype=dtype,
            warmup_runs=warmup_runs, num_runs=num_runs,
        )

        # 2. Sparse Profile
        print("    [2/2] Profiling Sparse GQA Backend...")
        torch.cuda.empty_cache()
        sparse_mem = profiler.profile_context_vram(sparse_model, batch_size=batch_size, seq_len=ctx, dtype=dtype)
        sparse_lat = benchmarker.benchmark_context(
            sparse_model, batch_size=batch_size, context_length=ctx,
            num_decode_tokens=num_decode_tokens, dtype=dtype,
            warmup_runs=warmup_runs, num_runs=num_runs,
        )

        # Speedups (> 1.0 means sparse is faster, < 1.0 means dense is faster)
        prefill_speedup = dense_lat["prefill_mean_ms"] / max(sparse_lat["prefill_mean_ms"], 1e-6)
        decode_speedup = dense_lat["decode_mean_ms_per_token"] / max(sparse_lat["decode_mean_ms_per_token"], 1e-6)

        entry = {
            "context_length": ctx,
            "batch_size": batch_size,
            "dense_prefill_ms": dense_lat["prefill_mean_ms"],
            "dense_prefill_tok_per_sec": dense_lat["prefill_tok_per_sec"],
            "dense_decode_ms_per_token": dense_lat["decode_mean_ms_per_token"],
            "dense_decode_tok_per_sec": dense_lat["decode_tok_per_sec"],
            "dense_peak_vram_gb": dense_mem["peak_vram_gb"],
            "dense_kv_cache_mb": dense_mem["kv_cache_actual_mb"],
            "sparse_prefill_ms": sparse_lat["prefill_mean_ms"],
            "sparse_prefill_tok_per_sec": sparse_lat["prefill_tok_per_sec"],
            "sparse_decode_ms_per_token": sparse_lat["decode_mean_ms_per_token"],
            "sparse_decode_tok_per_sec": sparse_lat["decode_tok_per_sec"],
            "sparse_peak_vram_gb": sparse_mem["peak_vram_gb"],
            "sparse_kv_cache_mb": sparse_mem["kv_cache_actual_mb"],
            "prefill_speedup": prefill_speedup,
            "decode_speedup": decode_speedup,
        }
        comparisons.append(entry)

        print(f"    Dense:  Prefill {dense_lat['prefill_mean_ms']:.2f} ms | Decode {dense_lat['decode_mean_ms_per_token']:.2f} ms/tok | Peak VRAM {dense_mem['peak_vram_gb']:.2f} GB")
        print(f"    Sparse: Prefill {sparse_lat['prefill_mean_ms']:.2f} ms | Decode {sparse_lat['decode_mean_ms_per_token']:.2f} ms/tok | Peak VRAM {sparse_mem['peak_vram_gb']:.2f} GB")
        print(f"    Speedup: Prefill {prefill_speedup:.2f}x | Decode {decode_speedup:.2f}x")

    summary = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "comparisons": comparisons,
    }
    return summary


def format_comparison_markdown(results: Dict[str, Any]) -> str:
    """Renders comparative benchmark into markdown table."""
    rows = results["comparisons"]
    md = []
    md.append("### Sparse vs. Dense Attention Wall-Clock Benchmark (RTX-Series 16GB GPU)\n")
    md.append("| Context | Dense Prefill | Sparse Prefill | Prefill Speedup | Dense Decode | Sparse Decode | Decode Speedup | Dense VRAM | Sparse VRAM |")
    md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rows:
        ctx_str = f"{r['context_length'] // 1024}K ({r['context_length']})" if r['context_length'] >= 1024 else str(r['context_length'])
        d_pref = f"{r['dense_prefill_ms']:.2f} ms ({r['dense_prefill_tok_per_sec']:,.0f} tok/s)"
        s_pref = f"{r['sparse_prefill_ms']:.2f} ms ({r['sparse_prefill_tok_per_sec']:,.0f} tok/s)"
        pref_speed = f"**{r['prefill_speedup']:.2f}x**"

        d_dec = f"{r['dense_decode_ms_per_token']:.2f} ms ({r['dense_decode_tok_per_sec']:.1f} tok/s)"
        s_dec = f"{r['sparse_decode_ms_per_token']:.2f} ms ({r['sparse_decode_tok_per_sec']:.1f} tok/s)"
        dec_speed = f"**{r['decode_speedup']:.2f}x**"

        d_vram = f"{r['dense_peak_vram_gb']:.2f} GB"
        s_vram = f"{r['sparse_peak_vram_gb']:.2f} GB"

        md.append(f"| **{ctx_str}** | {d_pref} | {s_pref} | {pref_speed} | {d_dec} | {s_dec} | {dec_speed} | {d_vram} | {s_vram} |")

    md.append("\n> [!NOTE]")
    md.append("> **Section 13 Wall-Clock Analysis**: In compliance with the Critical Performance Principle, theoretical FLOP reductions only convert into wall-clock speedup when memory access patterns and gather overhead do not exceed the cost of dense GEMM operations. Autoregressive decoding with physical block gathering bounds the attended KV cache to Top-K + Sink + Local blocks, providing substantial scaling headroom and enabling CPU offloading (Phase 5).")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Dense vs. Sparse Comparative Benchmark")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192], help="Context lengths")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_decode_tokens", type=int, default=16, help="Tokens to decode per run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_md", type=str, default="docs/results.md", help="Path to write markdown results")
    parser.add_argument("--output_json", type=str, default="docs/sparse_benchmark.json", help="Path to write JSON results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Initialize Dense Model
    dense_cfg = ModelConfig(**cfg["model"])
    dense_cfg.attention_backend = "dense"
    dense_model = SparDAResearchModel(dense_cfg).to(device)

    # Initialize Sparse Model
    sparse_cfg = ModelConfig(**cfg["model"])
    sparse_cfg.attention_backend = "sparse"
    sparse_model = SparDAResearchModel(sparse_cfg).to(device)

    # Load weights from checkpoint if available to share identical weights
    ckpt_path = args.checkpoint or os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/base"), "model_step_100.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights from {ckpt_path}...")
        load_checkpoint(ckpt_path, model=dense_model, device=device)
        # Copy matching state dict to sparse model
        sparse_model.load_state_dict(dense_model.state_dict(), strict=False)
        print("Loaded checkpoint weights into both models.")
    else:
        print(f"No checkpoint at {ckpt_path}; using initialized weights.")
        sparse_model.load_state_dict(dense_model.state_dict(), strict=False)

    results = run_comparative_benchmark(
        dense_model=dense_model,
        sparse_model=sparse_model,
        context_lengths=args.contexts,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        num_decode_tokens=args.num_decode_tokens,
    )

    md_content = format_comparison_markdown(results)
    print("\n" + "=" * 96)
    print("COMPARATIVE BENCHMARK OUTPUT:")
    print("=" * 96)
    print(md_content)
    print("=" * 96)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to {args.output_json}")

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
        # Append to docs/results.md
        existing = ""
        if os.path.exists(args.output_md):
            with open(args.output_md, "r") as f:
                existing = f.read()
        
        full_md = existing + "\n\n" + md_content + "\n"
        with open(args.output_md, "w") as f:
            f.write(full_md)
        print(f"Updated Markdown results in {args.output_md}")


if __name__ == "__main__":
    main()
