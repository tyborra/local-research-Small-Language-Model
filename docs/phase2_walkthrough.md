# Phase 2 Walkthrough: Production Baseline, Profiling & Context Scaling

## Objective Completed
Milestone 2 from [SparDA-Style Local Research Model.md](../SparDA-Style%20Local%20Research%20Model.md) and [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) is complete:
- Built validation perplexity and loss evaluation harness (`evaluation/perplexity.py` and `scripts/evaluate.py`).
- Built memory profiling engine (`evaluation/memory.py`) measuring theoretical vs actual KV cache, weights, and peak VRAM headroom.
- Built latency benchmarking suite (`evaluation/latency.py`) with CUDA event synchronization measuring TTFT, prefill throughput, and decode latency/throughput.
- Profiled full context scaling matrix (**1K, 2K, 4K, 8K, 16K**) on the 16 GB RTX-series GPU with BF16 precision.
- Logged empirical metrics to `docs/results.md` and `docs/dense_benchmark.json`.
- All 17 unit tests passing.

---

## 1. Validation Perplexity Results
Evaluated step 100 model checkpoint (`checkpoints/base/model_step_100.pt`) on held-out validation tokens (`data/val.bin`):
- **Cross-Entropy Loss**: `5.8189 nats`
- **Perplexity (PPL)**: `336.59`
- **Bits per Token**: `8.3949 bits/tok`
- **Total Evaluated Tokens**: `237,568`
- **Eval Throughput**: `39,066 tokens/sec`

---

## 2. Empirical Context Scaling Matrix (1K to 16K)
All measurements conducted in BF16 with PyTorch SDPA on the 16 GB NVIDIA RTX-Series GPU:

| Context Length | Batch | Prefill Latency | Prefill Throughput | Decode Latency | Decode Throughput | KV Cache VRAM | Peak VRAM | Headroom |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1K (1,024)** | 1 | 73.37 ms | 13,956 tok/s | 61.34 ms/tok | 16.3 tok/s | 12.00 MB | 0.67 GB | 15.32 GB |
| **2K (2,048)** | 1 | 75.66 ms | 27,070 tok/s | 67.90 ms/tok | 14.7 tok/s | 24.00 MB | 0.74 GB | 15.25 GB |
| **4K (4,096)** | 1 | 100.35 ms | 40,818 tok/s | 64.81 ms/tok | 15.4 tok/s | 49.88 MB | 0.88 GB | 15.11 GB |
| **8K (8,192)** | 1 | 229.52 ms | 35,692 tok/s | 56.76 ms/tok | 17.6 tok/s | 96.88 MB | 1.18 GB | 14.82 GB |
| **16K (16,384)** | 1 | 729.05 ms | 22,473 tok/s | 59.24 ms/tok | 16.9 tok/s | 192.88 MB | 1.77 GB | 14.23 GB |

### Key Observations:
1. **Memory Scalability**: At 16K context length, the KV cache footprint is only **192.88 MB** (due to 4 KV heads and GQA 3:1), with total peak VRAM under **1.8 GB**, leaving over **14.2 GB of VRAM headroom** on the 16 GB GPU.
2. **Prefill Throughput**: Peaks around 4K context at over **40,000 tok/s** as GPU compute saturation balances against memory bandwidth.
3. **Decode Latency**: Remains exceptionally steady across all context lengths (~56–67 ms per token).

---

## 3. Next Milestone: Phase 3 (Forecast Module)
With the dense baseline established and benchmarked, we advance to **Phase 3**:
- Implementing KV Block Compression and Prototype Generation.
- Building the Dense Teacher Selector to extract ground-truth block importance.
- Implementing the Forecast Block Selector and auxiliary Forecast Loss ($L_{\text{forecast}}$).
- Evaluating Forecast accuracy using Recall@K and Precision@K metrics.
