# SparDA Empirical Experimental Results & Profiling Data

### Dense Attention Baseline Benchmark (RTX-Series 16GB GPU)

| Context | Batch | Prefill Latency | Prefill Throughput | Decode Latency | Decode Throughput | KV Cache | Peak VRAM | Headroom |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1K (1024)** | 1 | 73.37 ms | 13,956 tok/s | 61.34 ms/tok | 16.3 tok/s | 12.00 MB | 0.67 GB | 15.32 GB |
| **2K (2048)** | 1 | 75.66 ms | 27,070 tok/s | 67.90 ms/tok | 14.7 tok/s | 24.00 MB | 0.74 GB | 15.25 GB |
| **4K (4096)** | 1 | 100.35 ms | 40,818 tok/s | 64.81 ms/tok | 15.4 tok/s | 49.88 MB | 0.88 GB | 15.11 GB |
| **8K (8192)** | 1 | 229.52 ms | 35,692 tok/s | 56.76 ms/tok | 17.6 tok/s | 96.88 MB | 1.18 GB | 14.82 GB |
| **16K (16384)** | 1 | 729.05 ms | 22,473 tok/s | 59.24 ms/tok | 16.9 tok/s | 192.88 MB | 1.77 GB | 14.23 GB |


### Sparse vs. Dense Attention Wall-Clock Benchmark (RTX-Series 16GB GPU)

| Context | Dense Prefill | Sparse Prefill | Prefill Speedup | Dense Decode | Sparse Decode | Decode Speedup | Dense VRAM | Sparse VRAM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1K (1024)** | 70.48 ms (14,529 tok/s) | 82.08 ms (12,476 tok/s) | **0.86x** | 50.86 ms (19.7 tok/s) | 88.05 ms (11.4 tok/s) | **0.58x** | 1.05 GB | 1.06 GB |
| **2K (2048)** | 51.33 ms (39,901 tok/s) | 79.39 ms (25,795 tok/s) | **0.65x** | 53.69 ms (18.6 tok/s) | 92.18 ms (10.8 tok/s) | **0.58x** | 1.12 GB | 1.12 GB |
| **4K (4096)** | 73.64 ms (55,625 tok/s) | 169.43 ms (24,175 tok/s) | **0.43x** | 51.89 ms (19.3 tok/s) | 82.08 ms (12.2 tok/s) | **0.63x** | 1.26 GB | 1.62 GB |

> [!NOTE]
> **Section 13 Wall-Clock Analysis**: In compliance with the Critical Performance Principle, theoretical FLOP reductions only convert into wall-clock speedup when memory access patterns and gather overhead do not exceed the cost of dense GEMM operations. Autoregressive decoding with physical block gathering bounds the attended KV cache to Top-K + Sink + Local blocks, providing substantial scaling headroom and enabling CPU offloading (Phase 5).
