# Phase 4 Walkthrough: Sparse Attention Backend & Comparative Benchmarking

## Objective Completed
Milestone 4 from [SparDA-Style Local Research Model.md](../SparDA-Style%20Local%20Research%20Model.md) and [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) is complete:
- Built `gather_kv_blocks` in [`model/attention/sparse.py`](../model/attention/sparse.py) implementing vectorized physical block gathering along the sequence dimension: $[B, n_{\text{kv\_head}}, K \cdot \text{block\_size}, \text{head\_dim}]$.
- Built `SparseAttention` subclassing [`BaseAttention`](../model/attention/base.py) with:
  1. Autoregressive decode path (`_forward_decode_gather`): gathers only Sink (Block 0) + Local (Current Block) + Top-K selected past blocks, reducing memory traffic from $O(T)$ to $O(K \cdot \text{block\_size})$.
  2. Full-sequence prefill path (`_forward_masked_sdpa`): vectorized block-sparse SDPA with causal masking.
  3. Chunked prefill path (`_forward_chunked_gather`): block-by-block gathering for physical gather profiling.
- Integrated `SparseAttention` into the architecture factory [`build_attention_backend`](../model/transformer.py) under `attention_backend="sparse"`.
- Unit tests written in [`tests/test_sparse_attention.py`](../tests/test_sparse_attention.py) covering block gather, causality preservation, gradient backpropagation, KV cache decode, and full model generation (5/5 passing; full test suite passes 28/28 tests).
- Built comparative benchmark suite [`benchmarks/sparse.py`](../benchmarks/sparse.py) profiling head-to-head wall-clock performance between Dense and Sparse backends across context lengths on an RTX-series 16GB GPU.
- Verified and empirically documented the Critical Performance Principle (Section 13).

---

## 1. Technical Architecture & Block Gather Mechanics

```
Full KV Cache (CPU / GPU)
[Block 0] [Block 1] [Block 2] ... [Block N-1]
    │                                   ▲
    │                                   │ Gather Selected Blocks
    ▼                                   │
Forecast F_{L-1} ──► Top-K Selector ────┤
                        + Sink (Block 0) ──┘
                        + Local (Block N-1)
                                │
                                ▼
                   Gathered Compact KV Tensor:
                   [B, n_kv_head, (Top_K + 2) * block_size, head_dim]
                                │
                                ▼
                       SDPA Attention Engine
```

### Key Properties:
- **Bounded Decoding Context**: During step-by-step autoregressive generation, regardless of whether the context is 1K, 8K, or 32K tokens, the attended KV footprint per step is bounded to:
  $$\text{Tokens Attended} = (K_{\text{top}} + 2) \times \text{block\_size}$$
  For $K=4, \text{block\_size}=32$, this is strictly **192 tokens** (~85x fewer tokens attended at 16K context).
- **PCIe Readiness**: By selecting explicit block indices, this module provides the exact interface required for Phase 5 (KV Cache CPU $\leftrightarrow$ GPU Offloading) and Phase 6 (Asynchronous Prefetch).

---

## 2. Head-to-Head Benchmark Results (`benchmarks/sparse.py`)

Conducted on a 16 GB RTX-series GPU with BF16 precision, loading `checkpoints/base/model_step_100.pt`:

| Context Length | Dense Prefill | Sparse Prefill | Prefill Speedup | Dense Decode | Sparse Decode | Decode Speedup | Dense Peak VRAM | Sparse Peak VRAM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1K (1,024)** | 70.48 ms (14,529 tok/s) | 82.08 ms (12,476 tok/s) | **0.86x** | 50.86 ms (19.7 tok/s) | 88.05 ms (11.4 tok/s) | **0.58x** | 1.05 GB | 1.06 GB |
| **2K (2,048)** | 51.33 ms (39,901 tok/s) | 79.39 ms (25,795 tok/s) | **0.65x** | 53.69 ms (18.6 tok/s) | 92.18 ms (10.8 tok/s) | **0.58x** | 1.12 GB | 1.12 GB |
| **4K (4,096)** | 73.64 ms (55,625 tok/s) | 169.43 ms (24,175 tok/s) | **0.43x** | 51.89 ms (19.3 tok/s) | 82.08 ms (12.2 tok/s) | **0.63x** | 1.26 GB | 1.62 GB |

### Empirical Insights on Section 13 (Critical Performance Principle):
1. **Wall-clock vs. FLOP Divergence**:
   At sequence lengths $\le$ 4K, dense FlashAttention/SDPA leverages hardware-level tensor core pipelining that outpaces Python-level tensor indexing, prototype compression, and gather kernel overhead.
2. **Where the SparDA Payoff Occurs**:
   The primary objective of Forecast Sparse Attention in local small models is not to out-accelerate tensor cores on small in-cache matrices, but rather to:
   - Cap active KV memory footprint during long-context decoding.
   - Restrict the data volume that must be transferred across the PCIe bus from host CPU RAM (Phase 5 & 6).
   - Maintain constant-time decoding memory access when context scales into tens of thousands of tokens.

---

## 3. Next Milestone: Phase 5 (KV Cache CPU Offloading)
Next milestone implements:
- Pinned CPU host memory KV cache storage (`model/attention/offload.py`).
- Selective CPU $\to$ GPU block streaming using the block indices identified by the Forecast selector.
- PCIe transfer volume and latency profiling (`benchmarks/offload.py`).
