# SparDA Empirical Experimental Results & Profiling Data

### Dense Attention Baseline Benchmark (RTX-Series 16GB GPU)

| Context | Batch | Prefill Latency | Prefill Throughput | Decode Latency | Decode Throughput | KV Cache | Peak VRAM | Headroom |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1K (1024)** | 1 | 73.37 ms | 13,956 tok/s | 61.34 ms/tok | 16.3 tok/s | 12.00 MB | 0.67 GB | 15.32 GB |
| **2K (2048)** | 1 | 75.66 ms | 27,070 tok/s | 67.90 ms/tok | 14.7 tok/s | 24.00 MB | 0.74 GB | 15.25 GB |
| **4K (4096)** | 1 | 100.35 ms | 40,818 tok/s | 64.81 ms/tok | 15.4 tok/s | 49.88 MB | 0.88 GB | 15.11 GB |
| **8K (8192)** | 1 | 229.52 ms | 35,692 tok/s | 56.76 ms/tok | 17.6 tok/s | 96.88 MB | 1.18 GB | 14.82 GB |
| **16K (16384)** | 1 | 729.05 ms | 22,473 tok/s | 59.24 ms/tok | 16.9 tok/s | 192.88 MB | 1.77 GB | 14.23 GB |
