# SparDA Full Scaling Performance Matrix (1K to 32K)

- **Evaluated Device**: NVIDIA GeForce RTX 4090 Laptop GPU (15.99 GB VRAM)
- **Precision**: `torch.bfloat16` | **Batch Size**: 1

### 1. Prefill Latency (ms) & Throughput (tokens/sec)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 78.2 ms<br>*13,090 tok/s* | 95.4 ms<br>*10,729 tok/s* (0.82x) | 106.4 ms<br>*9,623 tok/s* (0.74x) | 71.4 ms<br>*14,348 tok/s* (1.10x) | 88.0 ms<br>*11,631 tok/s* (0.89x) |

### 2. Autoregressive Decode Latency (ms/token) & Throughput (tokens/sec)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 85.76 ms<br>*11.7 tok/s* | 124.50 ms<br>*8.0 tok/s* (0.69x) | 133.05 ms<br>*7.5 tok/s* (0.64x) | 185.77 ms<br>*5.4 tok/s* (0.46x) | 217.08 ms<br>*4.6 tok/s* (0.40x) |

### 3. Effective GPU KV-Cache (MB) & Peak Allocated VRAM (GB)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 12.0 MB<br>*0.74 GB peak* | 12.0 MB (-0%)<br>*0.74 GB peak* | 12.0 MB (-0%)<br>*0.74 GB peak* | 2.2 MB (-81%)<br>*0.73 GB peak* | 2.2 MB (-81%)<br>*0.73 GB peak* |

### 4. Empirical Architectural Crossover Points

| Attention Backend | Prefill Crossover | Decode Crossover | 50% KV Savings | 90% KV Savings | Max KV Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Static Local + Sink Sparse** | None (Bound by gather) | None | N/A | N/A | **1.0x** |
| **Forecast-Guided GPU Sparse** | None (Bound by gather) | None | N/A | N/A | **1.0x** |
| **Synchronous Forecast Offload** | 1,024 tokens | None | 1,024 tokens | N/A | **5.4x** |
| **Asynchronous Prefetched Offload** | None (Bound by gather) | None | 1,024 tokens | N/A | **5.4x** |