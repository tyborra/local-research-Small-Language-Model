# SparDA Full Scaling Performance Matrix (1K to 32K)

- **Evaluated Device**: NVIDIA GeForce RTX 4090 Laptop GPU (15.99 GB VRAM)
- **Precision**: `torch.bfloat16` | **Batch Size**: 1

### 1. Prefill Latency (ms) & Throughput (tokens/sec)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 56.4 ms<br>*18,164 tok/s* | 99.7 ms<br>*10,266 tok/s* (0.57x) | 84.4 ms<br>*12,136 tok/s* (0.67x) | 64.8 ms<br>*15,796 tok/s* (0.87x) | 66.1 ms<br>*15,480 tok/s* (0.85x) |
| **2,048** | 61.3 ms<br>*33,386 tok/s* | 72.6 ms<br>*28,222 tok/s* (0.85x) | 75.4 ms<br>*27,162 tok/s* (0.81x) | 68.3 ms<br>*30,001 tok/s* (0.90x) | 70.0 ms<br>*29,258 tok/s* (0.88x) |
| **4,096** | 71.5 ms<br>*57,278 tok/s* | 149.9 ms<br>*27,326 tok/s* (0.48x) | 167.2 ms<br>*24,502 tok/s* (0.43x) | 75.5 ms<br>*54,253 tok/s* (0.95x) | 95.0 ms<br>*43,119 tok/s* (0.75x) |
| **8,192** | 166.0 ms<br>*49,355 tok/s* | 346.4 ms<br>*23,648 tok/s* (0.48x) | 564.4 ms<br>*14,516 tok/s* (0.29x) | 193.8 ms<br>*42,274 tok/s* (0.86x) | 198.9 ms<br>*41,189 tok/s* (0.83x) |
| **16,384** | 468.2 ms<br>*34,992 tok/s* | 19087.5 ms<br>*858 tok/s* (0.02x) | 22152.0 ms<br>*740 tok/s* (0.02x) | 489.0 ms<br>*33,508 tok/s* (0.96x) | 539.5 ms<br>*30,370 tok/s* (0.87x) |
| **32,768** | 1374.6 ms<br>*23,839 tok/s* | 38830.1 ms<br>*844 tok/s* (0.04x) | 39917.5 ms<br>*821 tok/s* (0.03x) | 1364.1 ms<br>*24,022 tok/s* (1.01x) | 1632.0 ms<br>*20,078 tok/s* (0.84x) |

### 2. Autoregressive Decode Latency (ms/token) & Throughput (tokens/sec)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 61.00 ms<br>*16.4 tok/s* | 95.20 ms<br>*10.5 tok/s* (0.64x) | 95.88 ms<br>*10.4 tok/s* (0.64x) | 172.84 ms<br>*5.8 tok/s* (0.35x) | 185.68 ms<br>*5.4 tok/s* (0.33x) |
| **2,048** | 61.02 ms<br>*16.4 tok/s* | 105.61 ms<br>*9.5 tok/s* (0.58x) | 98.80 ms<br>*10.1 tok/s* (0.62x) | 169.27 ms<br>*5.9 tok/s* (0.36x) | 193.49 ms<br>*5.2 tok/s* (0.32x) |
| **4,096** | 61.24 ms<br>*16.3 tok/s* | 111.88 ms<br>*8.9 tok/s* (0.55x) | 100.71 ms<br>*9.9 tok/s* (0.61x) | 174.57 ms<br>*5.7 tok/s* (0.35x) | 188.07 ms<br>*5.3 tok/s* (0.33x) |
| **8,192** | 48.84 ms<br>*20.5 tok/s* | 79.88 ms<br>*12.5 tok/s* (0.61x) | 114.08 ms<br>*8.8 tok/s* (0.43x) | 177.85 ms<br>*5.6 tok/s* (0.27x) | 194.14 ms<br>*5.2 tok/s* (0.25x) |
| **16,384** | 56.75 ms<br>*17.6 tok/s* | 68.80 ms<br>*14.5 tok/s* (0.82x) | 120.45 ms<br>*8.3 tok/s* (0.47x) | 196.87 ms<br>*5.1 tok/s* (0.29x) | 196.01 ms<br>*5.1 tok/s* (0.29x) |
| **32,768** | 75.06 ms<br>*13.3 tok/s* | 89.75 ms<br>*11.1 tok/s* (0.84x) | 105.90 ms<br>*9.4 tok/s* (0.71x) | 213.14 ms<br>*4.7 tok/s* (0.35x) | 189.11 ms<br>*5.3 tok/s* (0.40x) |

### 3. Effective GPU KV-Cache (MB) & Peak Allocated VRAM (GB)

| Context | Dense GQA Baseline | Static Local + Sink Sparse | Forecast-Guided GPU Sparse | Synchronous Forecast Offload | Asynchronous Prefetched Offload |
|:---:| ---: | ---: | ---: | ---: | ---: |
| **1,024** | 12.2 MB<br>*0.74 GB peak* | 12.2 MB (-0%)<br>*0.74 GB peak* | 12.2 MB (-0%)<br>*0.74 GB peak* | 2.2 MB (-82%)<br>*0.73 GB peak* | 2.2 MB (-82%)<br>*0.73 GB peak* |
| **2,048** | 24.2 MB<br>*0.89 GB peak* | 24.2 MB (-0%)<br>*0.89 GB peak* | 24.2 MB (-0%)<br>*0.89 GB peak* | 2.2 MB (-91%)<br>*0.85 GB peak* | 2.2 MB (-91%)<br>*0.85 GB peak* |
| **4,096** | 48.2 MB<br>*1.16 GB peak* | 48.2 MB (-0%)<br>*1.16 GB peak* | 48.2 MB (-0%)<br>*1.53 GB peak* | 2.2 MB (-95%)<br>*1.11 GB peak* | 2.2 MB (-95%)<br>*1.11 GB peak* |
| **8,192** | 96.2 MB<br>*1.71 GB peak* | 96.2 MB (-0%)<br>*1.82 GB peak* | 96.2 MB (-0%)<br>*3.81 GB peak* | 2.2 MB (-98%)<br>*1.61 GB peak* | 2.2 MB (-98%)<br>*1.61 GB peak* |
| **16,384** | 192.2 MB<br>*2.82 GB peak* | 192.2 MB (-0%)<br>*2.82 GB peak* | 192.2 MB (-0%)<br>*2.82 GB peak* | 2.2 MB (-99%)<br>*2.63 GB peak* | 2.2 MB (-99%)<br>*2.63 GB peak* |
| **32,768** | 384.2 MB<br>*5.04 GB peak* | 384.2 MB (-0%)<br>*5.04 GB peak* | 384.2 MB (-0%)<br>*5.04 GB peak* | 2.2 MB (-99%)<br>*4.66 GB peak* | 2.2 MB (-99%)<br>*4.66 GB peak* |

### 4. Empirical Architectural Crossover Points

| Attention Backend | Prefill Crossover | Decode Crossover | 50% KV Savings | 90% KV Savings | Max KV Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Static Local + Sink Sparse** | None (Bound by gather) | None | N/A | N/A | **1.0x** |
| **Forecast-Guided GPU Sparse** | None (Bound by gather) | None | N/A | N/A | **1.0x** |
| **Synchronous Forecast Offload** | 32,768 tokens | None | 1,024 tokens | 2,048 tokens | **170.8x** |
| **Asynchronous Prefetched Offload** | None (Bound by gather) | None | 1,024 tokens | 2,048 tokens | **170.8x** |