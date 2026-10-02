"""Dense Attention Baseline Benchmark Suite across Context Lengths (1K to 8K+)."""

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
from evaluation.memory import MemoryProfiler, calculate_kv_cache_bytes
from evaluation.latency import LatencyBenchmarker
from training.checkpoint import load_checkpoint


def run_dense_benchmarks(
    model: SparDAResearchModel,
    context_lengths: List[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    num_decode_tokens: int = 16,
    warmup_runs: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Runs comprehensive dense baseline benchmarks across specified context lengths.
    
    Measures:
        - Prefill latency (ms), TTFT (ms), and prefill throughput (tokens/sec)
        - Decode latency per token (ms) and decode throughput (tokens/sec)
        - Weights VRAM, Analytical KV Cache VRAM, Actual KV Cache VRAM, Activation VRAM, Peak VRAM, Headroom
    """
    if device is None:
        device = next(model.parameters()).device

    profiler = MemoryProfiler(device)
    benchmarker = LatencyBenchmarker(device)

    static_weights = profiler.profile_static_weights(model)
    results_by_context = []

    print("=" * 88)
    print(f"SPARDA DENSE ATTENTION BASELINE PROFILER (Device: {device}, Dtype: {dtype})")
    print(f"Parameters: {model.count_parameters()['total']:,} | Model Weights VRAM: {static_weights['allocated_weights_mb']:.2f} MB")
    print("=" * 88)

    for ctx in context_lengths:
        print(f"\n>>> Benchmarking Context Length: {ctx:,} tokens (Batch Size: {batch_size}) ...")

        # 1. Profile Memory
        mem_stats = profiler.profile_context_vram(
            model=model,
            batch_size=batch_size,
            seq_len=ctx,
            dtype=dtype,
        )

        # 2. Profile Latency
        lat_stats = benchmarker.benchmark_context(
            model=model,
            batch_size=batch_size,
            context_length=ctx,
            num_decode_tokens=num_decode_tokens,
            dtype=dtype,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
        )

        combined = {
            "context_length": ctx,
            "batch_size": batch_size,
            **mem_stats,
            **lat_stats,
        }
        results_by_context.append(combined)

        print(f"  Prefill: {lat_stats['prefill_mean_ms']:.2f} ms ({lat_stats['prefill_tok_per_sec']:,.0f} tok/s)")
        print(f"  Decode:  {lat_stats['decode_mean_ms_per_token']:.2f} ms/tok ({lat_stats['decode_tok_per_sec']:.1f} tok/s)")
        print(f"  KV Cache: {mem_stats['kv_cache_actual_mb']:.2f} MB | Peak VRAM: {mem_stats['peak_vram_gb']:.2f} GB (Headroom: {mem_stats['headroom_vram_gb']:.2f} GB)")

    summary = {
        "model_parameters": model.count_parameters(),
        "static_weights": static_weights,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "benchmarks": results_by_context,
    }
    return summary


def format_markdown_table(benchmark_results: Dict[str, Any]) -> str:
    """Formats benchmark results into a clean markdown table for docs/results.md."""
    rows = benchmark_results["benchmarks"]
    md = []
    md.append("### Dense Attention Baseline Benchmark (RTX-Series 16GB GPU)\n")
    md.append("| Context | Batch | Prefill Latency | Prefill Throughput | Decode Latency | Decode Throughput | KV Cache | Peak VRAM | Headroom |")
    md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rows:
        ctx_str = f"{r['context_length'] // 1024}K ({r['context_length']})" if r['context_length'] >= 1024 else str(r['context_length'])
        prefill_str = f"{r['prefill_mean_ms']:.2f} ms"
        prefill_tok_str = f"{r['prefill_tok_per_sec']:,.0f} tok/s"
        decode_str = f"{r['decode_mean_ms_per_token']:.2f} ms/tok"
        decode_tok_str = f"{r['decode_tok_per_sec']:.1f} tok/s"
        kv_str = f"{r['kv_cache_actual_mb']:.2f} MB"
        peak_str = f"{r['peak_vram_gb']:.2f} GB"
        headroom_str = f"{r['headroom_vram_gb']:.2f} GB"

        md.append(f"| **{ctx_str}** | {r['batch_size']} | {prefill_str} | {prefill_tok_str} | {decode_str} | {decode_tok_str} | {kv_str} | {peak_str} | {headroom_str} |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Dense Baseline Attention Benchmark")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192], help="Context lengths")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_decode_tokens", type=int, default=16, help="Tokens to decode per run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_md", type=str, default="docs/results.md", help="Path to write markdown results")
    parser.add_argument("--output_json", type=str, default="docs/dense_benchmark.json", help="Path to write JSON results")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Instantiate model
    model_cfg = ModelConfig(**cfg["model"])
    model = SparDAResearchModel(model_cfg).to(device)

    # Load checkpoint if specified or exists
    ckpt_path = args.checkpoint or os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/base"), "model_step_100.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        load_checkpoint(ckpt_path, model=model, device=device)
        print("Checkpoint successfully loaded.")
    else:
        print(f"No checkpoint found at {ckpt_path}. Using freshly initialized model.")

    results = run_dense_benchmarks(
        model=model,
        context_lengths=args.contexts,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        num_decode_tokens=args.num_decode_tokens,
    )

    # Format Markdown
    md_content = format_markdown_table(results)
    print("\n" + "=" * 88)
    print("MARKDOWN TABLE OUTPUT:")
    print("=" * 88)
    print(md_content)
    print("=" * 88)

    # Save to files
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to {args.output_json}")

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
        # Check if file exists, if so append or write
        header = "# SparDA Empirical Experimental Results & Profiling Data\n\n"
        full_md = header + md_content + "\n"
        with open(args.output_md, "w") as f:
            f.write(full_md)
        print(f"Saved Markdown results to {args.output_md}")


if __name__ == "__main__":
    main()
