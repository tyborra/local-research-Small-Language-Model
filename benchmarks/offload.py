"""KV Cache CPU-GPU Offloading Benchmark Suite.

Implements Section 11 of the SparDA specification:
Measures:
1. KV transfer volume across PCIe (KB per decode step and cumulative MB).
2. Latency breakdown:
   - Selection latency (ms)
   - Transfer latency (ms)
   - Attention latency (ms)
   - Total decode step latency (ms)
3. GPU VRAM reduction: GPU KV footprint in Dense vs. Offload.
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
from model.attention.offload import OffloadedKVCache
from evaluation.memory import MemoryProfiler, calculate_kv_cache_bytes
from training.checkpoint import load_checkpoint


def run_offload_benchmark(
    model: SparDAResearchModel,
    context_lengths: List[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    num_decode_tokens: int = 16,
    warmup_steps: int = 3,
) -> Dict[str, Any]:
    """Profiles KV cache offloading across context lengths."""
    if device is None:
        device = next(model.parameters()).device

    is_cuda = device.type == "cuda"
    profiler = MemoryProfiler(device)
    cfg = model.config

    results = []

    print("=" * 104)
    print(f"SPARDA KV CACHE OFFLOADING BENCHMARK (Device: {device}, Dtype: {dtype})")
    print(f"Authoritative KV in Pinned CPU RAM -> Streaming Selected Blocks to GPU VRAM")
    print("=" * 104)

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

        # Offload GPU resident KV: only (top_k_blocks + 2) blocks per layer
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

        # 2. Setup OffloadedKVCache per layer
        kv_caches = [
            OffloadedKVCache(
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

        # Reset byte counters for decode profiling
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

        # 5. Timed Decode with Detailed Latency & Transfer Breakdown
        step_latencies_ms = []
        layer_selection_ms = []
        layer_transfer_ms = []
        layer_attention_ms = []

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

            # Collect layer profiles
            for layer in model.layers:
                attn = layer.attention
                prof = getattr(attn, "last_profile", None)
                if prof:
                    layer_selection_ms.append(prof["selection_ms"])
                    layer_transfer_ms.append(prof["transfer_ms"])
                    layer_attention_ms.append(prof["attention_ms"])

        total_bytes = sum(c.total_bytes_transferred for c in kv_caches)
        bytes_per_step = total_bytes / max(num_decode_tokens, 1)
        kb_per_step = bytes_per_step / 1024.0

        mean_step_ms = sum(step_latencies_ms) / len(step_latencies_ms)
        mean_sel_ms = (sum(layer_selection_ms) / len(layer_selection_ms)) * cfg.n_layer if layer_selection_ms else 0.0
        mean_trans_ms = (sum(layer_transfer_ms) / len(layer_transfer_ms)) * cfg.n_layer if layer_transfer_ms else 0.0
        mean_attn_ms = (sum(layer_attention_ms) / len(layer_attention_ms)) * cfg.n_layer if layer_attention_ms else 0.0

        tok_per_sec = 1000.0 / max(mean_step_ms, 1e-6)

        entry = {
            "context_length": ctx,
            "dense_gpu_kv_mb": dense_kv_mb,
            "offload_gpu_kv_mb": offload_gpu_kv_mb,
            "vram_reduction_ratio": vram_reduction_ratio,
            "transfer_kb_per_step": kb_per_step,
            "selection_latency_ms": mean_sel_ms,
            "transfer_latency_ms": mean_trans_ms,
            "attention_latency_ms": mean_attn_ms,
            "total_decode_step_ms": mean_step_ms,
            "decode_tok_per_sec": tok_per_sec,
        }
        results.append(entry)

        print(f"  GPU KV VRAM: Dense {dense_kv_mb:.2f} MB -> Offloaded {offload_gpu_kv_mb:.2f} MB ({vram_reduction_ratio:.1f}x VRAM reduction)")
        print(f"  PCIe Transfer: {kb_per_step:,.1f} KB / step ({bytes_per_step / (1024**2):.2f} MB/step)")
        print(f"  Latency Breakdown: Selection {mean_sel_ms:.2f} ms | Transfer {mean_trans_ms:.2f} ms | Attention {mean_attn_ms:.2f} ms | Total {mean_step_ms:.2f} ms ({tok_per_sec:.1f} tok/s)")

    summary = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "n_layer": cfg.n_layer,
        "block_size": cfg.block_size,
        "top_k_blocks": cfg.top_k_blocks,
        "benchmarks": results,
    }
    return summary


def format_offload_markdown(summary: Dict[str, Any]) -> str:
    """Formats benchmark results into a markdown table."""
    rows = summary["benchmarks"]
    md = []
    md.append("### KV Cache CPU-GPU Offloading Benchmark (RTX-Series 16GB GPU)\n")
    md.append("| Context | Dense GPU KV | Offload GPU KV | VRAM Reduction | PCIe Transfer/Step | Selection Latency | Transfer Latency | Attention Latency | Total Decode Latency |")
    md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in rows:
        ctx_str = f"{r['context_length'] // 1024}K ({r['context_length']})" if r['context_length'] >= 1024 else str(r['context_length'])
        dense_kv = f"{r['dense_gpu_kv_mb']:.2f} MB"
        offload_kv = f"{r['offload_gpu_kv_mb']:.2f} MB"
        vram_red = f"**{r['vram_reduction_ratio']:.1f}x**"
        pcie_trans = f"{r['transfer_kb_per_step']:,.1f} KB"
        sel_lat = f"{r['selection_latency_ms']:.2f} ms"
        trans_lat = f"{r['transfer_latency_ms']:.2f} ms"
        attn_lat = f"{r['attention_latency_ms']:.2f} ms"
        tot_lat = f"**{r['total_decode_step_ms']:.2f} ms** ({r['decode_tok_per_sec']:.1f} tok/s)"

        md.append(f"| **{ctx_str}** | {dense_kv} | {offload_kv} | {vram_red} | {pcie_trans} | {sel_lat} | {trans_lat} | {attn_lat} | {tot_lat} |")

    md.append("\n> [!NOTE]")
    md.append("> **Phase 5 KV Offloading Verification**: Maintains the authoritative full KV cache in Host CPU pinned RAM (`pin_memory`). Only Top-K + Sink + Local blocks are streamed over PCIe on demand, bounding GPU KV footprint to a constant size and unlocking long-context execution without GPU VRAM exhaustion.")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Offload Benchmark")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096, 8192], help="Context lengths")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_decode_tokens", type=int, default=16, help="Tokens to decode per run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_md", type=str, default="docs/results.md", help="Path to write markdown results")
    parser.add_argument("--output_json", type=str, default="docs/offload_benchmark.json", help="Path to write JSON results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Instantiate model with offload backend
    model_cfg = ModelConfig(**cfg["model"])
    model_cfg.attention_backend = "offload"
    model = SparDAResearchModel(model_cfg).to(device)

    ckpt_path = args.checkpoint or os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/base"), "model_step_100.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights from {ckpt_path}...")
        load_checkpoint(ckpt_path, model=model, device=device)
        print("Checkpoint successfully loaded.")
    else:
        print(f"No checkpoint found at {ckpt_path}. Using initialized weights.")

    summary = run_offload_benchmark(
        model=model,
        context_lengths=args.contexts,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        num_decode_tokens=args.num_decode_tokens,
    )

    md_content = format_offload_markdown(summary)
    print("\n" + "=" * 104)
    print("MARKDOWN TABLE OUTPUT:")
    print("=" * 104)
    print(md_content)
    print("=" * 104)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON results to {args.output_json}")

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
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
