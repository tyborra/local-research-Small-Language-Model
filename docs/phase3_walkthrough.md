# Phase 3 Walkthrough: Forecast Module & Block Selection

## Objective Completed
Milestone 3 from [SparDA-Style Local Research Model.md](../SparDA-Style%20Local%20Research%20Model.md) and [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) is complete:
- Built `ForecastProjection` ($W_f \in \mathbb{R}^{d_{\text{embd}} \times (n_{\text{kv\_head}} \cdot d_{\text{forecast}})}$) projecting hidden state $x_L$ to forecast representations $[B, T, 4, 64]$.
- Built `KVBlockCompressor` pooling layer $L+1$ keys into block prototype vectors $[B, N_{\text{blocks}}, 4, 64]$ with arbitrary sequence padding support.
- Built `ForecastBlockSelector` executing scaled dot-product scoring against block prototypes with strict causal masking and Top-K selection.
- Built `TeacherBlockSelector` aggregating dense multi-head attention weights across GQA head groups ($12 \to 4$) into ground-truth block importance targets and probability distributions.
- Built `compute_forecast_loss` supporting both KL-divergence and Binary Cross-Entropy with logits.
- Built `evaluation/forecast_metrics.py` computing Recall@K, Precision@K, Top-1 Block Selection Accuracy, and empirical KL-divergence.
- Verified end-to-end against model checkpoint across all 11 layer transitions via `scripts/benchmark_forecast.py`.
- Unit tests written in `tests/test_forecast.py` (6/6 passing); full test suite passes 23/23 tests.

---

## 1. Mathematical Architecture & Interfaces

### Forecast Data Flow:
```
Layer L Hidden State (x_L)
          │
          ▼
   RMSNorm(x_L)
     ├──► GQA Attention (Layer L)
     └──► ForecastProjection (W_f)
                │
                ▼
        F_L [B, T_q, 4, 64]
                │
                ▼
      ForecastBlockSelector ◄─── KVBlockCompressor(K_{L+1})
                │
                ▼
    Top-K Block Indices [B, 4, T_q, K]
                │
                ▼
      Layer L+1 Sparse Attention
```

---

## 2. Empirical Validation Results (`scripts/benchmark_forecast.py`)
Evaluated across all 11 layer transitions on held-out tokens (`data/val.bin`, block size 32 tokens, 16 GB RTX-series GPU):

| Layer Transition | Top-1 Accuracy | Recall@1 / Prec@1 | Recall@4 / Prec@4 | Recall@8 / Prec@8 | KL Divergence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 0 $\to$ 1** | 14.43% | 14.43% | 38.47% | 59.17% | 0.4894 |
| **Layer 1 $\to$ 2** | 12.55% | 12.55% | 37.16% | 58.38% | 0.1607 |
| **Layer 2 $\to$ 3** | 12.59% | 12.59% | 36.89% | 57.92% | 0.5181 |
| **Layer 3 $\to$ 4** | 12.76% | 12.76% | 37.74% | 58.24% | 0.6397 |
| **Layer 4 $\to$ 5** | 13.97% | 13.97% | 37.80% | 58.31% | 0.4481 |
| **Layer 5 $\to$ 6** | 13.20% | 13.20% | 38.30% | 59.23% | 0.6881 |
| **Layer 6 $\to$ 7** | 13.07% | 13.07% | 37.15% | 58.33% | 1.1758 |
| **Layer 7 $\to$ 8** | 12.04% | 12.04% | 36.57% | 58.17% | 1.2527 |
| **Layer 8 $\to$ 9** | 11.74% | 11.74% | 37.39% | 58.71% | 0.7558 |
| **Layer 9 $\to$ 10** | 11.50% | 11.50% | 37.10% | 58.52% | 1.1254 |
| **Layer 10 $\to$ 11** | 11.84% | 11.84% | 37.57% | 58.71% | 1.3316 |

### Analysis:
- Even prior to auxiliary forecast joint training, the causal block structure and prototype pooling achieves **~58.5% Recall@8** over 32 blocks (where uniform random selection would yield $8 / 32 = 25\%$).
- The pipeline provides the required teacher guidance signals for training and evaluation.

---

## 3. Next Milestone: Phase 4 (Sparse Attention Backend)
Next milestone implements:
- Block-gather kernel and `SparseAttention` backend ([model/attention/sparse.py](../model/attention/sparse.py)).
- Direct comparison of wall-clock latency, throughput, and VRAM between Dense and Sparse backends.
