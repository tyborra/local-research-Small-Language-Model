# SparDA-Style Local Research Model

## Technical Build Specification

### 1. Objective

Build a small language model and experimental attention system **from scratch**, designed to investigate:

1. Forecast-based sparse attention
2. Long-context inference
3. KV-cache reduction/offloading
4. Asynchronous CPU↔GPU KV prefetch
5. Continual learning and replay
6. The interaction between Forecast-based attention and continual learning

The project must run locally on a **16 GB NVIDIA RTX 4090-class GPU**.

### Important definition of "from scratch"

"From scratch" means **the language model must be pretrained from randomly initialized weights**.

It does **not** mean manually implementing every software component.

Allowed:

* PyTorch
* CUDA
* Triton
* FlashAttention / PyTorch SDPA
* Hugging Face tokenizers or equivalent tokenizer libraries
* NumPy
* standard Python libraries
* standard dataset tooling
* existing open-source kernels/utilities

Not allowed as a starting point:

* Qwen
* Llama
* Mistral
* Gemma
* Phi
* GPT-family pretrained checkpoints
* LoRA/adapters applied to an existing pretrained model
* Any other pretrained language-model weights

The model architecture, weights, tokenizer vocabulary, and training process should be created for this project.

---

# 2. Target Model

Initial architecture:

```text
vocab_size       = 32,000
n_layer          = 12
n_embd           = 768
n_head           = 12
n_kv_head        = 4
head_dim         = 64
ffn_dim          = 2,048
forecast_heads   = 4
forecast_dim     = 64
context_length   = 1,024 initially
```

Attention:

```text
12 query heads
4 KV heads
3 query heads per KV head
GQA ratio = 3:1
```

Normalization:

```text
RMSNorm
```

Position encoding:

```text
RoPE
```

Feed-forward:

```text
SwiGLU
```

Embeddings:

```text
32K vocabulary
tied input/output embeddings
```

Expected model size:

```text
~84M parameters
```

A separate untied LM head would increase the model to approximately 108M parameters, but tied embeddings are preferred for the initial implementation.

---

# 3. Hardware Target

Primary development target:

```text
GPU: RTX 4090 Laptop / 16 GB VRAM
Precision: BF16
Framework: PyTorch
```

The model should be designed to leave substantial VRAM headroom for:

* activations
* optimizer state
* gradient checkpointing
* KV cache
* Forecast projections
* experimental CPU/GPU KV transfers
* profiling

Initial training configuration should prioritize fast iteration rather than maximum batch size.

Starting point:

```text
context:       1024
microbatch:    4–8
gradient accumulation: as required
precision:     BF16
gradient checkpointing: enabled
```

Exact batch size must be determined empirically.

---

# 4. Context Scaling

Do not optimize exclusively for 1K–4K context.

The primary research value comes from increasing context length.

Benchmark progressively:

```text
1K
2K
4K
8K
16K
32K
```

Potential later experiments:

```text
64K
128K
```

The small model is intentionally useful here because the goal is to study the **attention/KV behavior**, not to reproduce the quality of a large production LLM.

---

# 5. Architecture

The attention system should be modular.

Each Transformer layer contains:

```text
RMSNorm
   ↓
GQA Attention
   ├── Q projection
   ├── K projection
   ├── V projection
   └── Forecast projection
   ↓
Residual
   ↓
RMSNorm
   ↓
SwiGLU
   ↓
Residual
```

Forecast:

```text
Forecast_L
     │
     ▼
predict KV blocks required by Layer L+1
```

The important relationship is:

```text
Layer L
   │
   ├── normal attention
   │
   └── Forecast_L
           │
           ▼
      select KV blocks
           │
           ▼
Layer L+1 sparse attention
```

Forecast is therefore **one layer ahead**, rather than simply being another input to the same layer's attention.

---

# 6. Forecast Projection

Each layer should contain:

```python
wf = nn.Linear(
    n_embd,
    n_kv_head * forecast_dim,
    bias=False
)
```

With the initial configuration:

```text
4 forecast groups
64 dimensions/group
```

Forecast output:

```text
[B, T, 4, 64]
```

The Forecast mechanism should remain modular enough to experiment with:

* 32-dimensional Forecast
* 64-dimensional Forecast
* different scoring mechanisms
* different block sizes
* different top-k ratios

---

# 7. Sparse Attention Architecture

Do NOT hard-code sparse attention directly into the Transformer.

Create interchangeable attention backends:

```text
DenseAttention
SparseAttention
ForecastAttention
ForecastOffloadAttention
ForecastAsyncAttention
```

The same model should be capable of switching between them.

Conceptually:

```text
Model
  │
  └── AttentionBackend
          ├── Dense
          ├── Sparse
          ├── Forecast
          ├── Forecast + Offload
          └── Forecast + Async Prefetch
```

This is essential for controlled experiments.

---

# 8. Dense Baseline

The first working model must use ordinary dense GQA attention.

Requirements:

* model trains
* loss decreases
* checkpoint save/load works
* generation works
* validation perplexity works
* VRAM is measured
* tokens/sec is measured

This becomes the baseline against which every later experiment is compared.

Do not implement sparse attention first.

---

# 9. Sparse Selection

KV should be divided into blocks.

Example:

```text
context
   ↓
KV blocks

[0] [1] [2] [3] [4] [5] ... [N]
```

Forecast predicts which blocks the next layer will require.

Conceptual flow:

```text
Forecast
    ×
compressed KV block representation
    ↓
block scores
    ↓
Top-K
    ↓
selected KV blocks
```

The exact selector should be implemented as a separate module.

Possible interface:

```python
selected_blocks = selector(
    forecast,
    kv_block_representations,
    top_k
)
```

---

# 10. Forecast Training

Initial implementation should support two approaches.

### Approach A — Joint training

Train the Forecast projection together with the model.

### Approach B — Teacher-based Forecast training

Use a dense/full selector as a teacher.

Train Forecast to predict the blocks that the teacher would select.

Metrics:

```text
Recall@K
Precision@K
KL divergence
block-selection accuracy
```

Start with the teacher-based approach if it makes debugging easier.

---

# 11. KV Offloading

After Forecast selection works, introduce CPU/GPU KV movement.

Concept:

```text
CPU KV cache
      │
      │ selected blocks
      ▼
GPU KV cache
      │
      ▼
Sparse attention
```

Measure:

```text
KV transfer volume
transfer latency
selection latency
attention latency
total latency
```

The purpose is to determine whether Forecast can reduce the amount of KV that must reside on the GPU.

---

# 12. Asynchronous Prefetch

Final experimental architecture:

```text
Layer L computation
        │
        ├──────────────► GPU compute
        │
        └── Forecast
              │
              ▼
        predict Layer L+1
              │
              ▼
        select KV blocks
              │
              ▼
       asynchronous prefetch
              │
              ▼
        Layer L+1 attention
```

The goal is to overlap:

```text
CPU → GPU KV transfer
```

with:

```text
GPU computation
```

This is more important than merely reducing theoretical attention FLOPs.

A naïve gather implementation is insufficient to demonstrate a real performance advantage.

---

# 13. Critical Performance Principle

Do not assume:

```text
fewer attention FLOPs = faster inference
```

Irregular sparse GPU operations can be slower than dense matrix operations because of:

* memory access patterns
* gather operations
* kernel launch overhead
* poor GPU utilization
* synchronization

Therefore the project must measure **wall-clock performance**, not just theoretical FLOPs.

Primary metrics:

```text
tokens/sec
latency/token
prefill latency
decode latency
peak VRAM
KV-cache VRAM
CPU→GPU transfer volume
GPU utilization
```

---

# 14. Continual Learning Experiment

Continual learning is a **separate experimental question** from SparDA itself.

Initial staged dataset structure:

```text
Stage 1
General Python + technical documentation

Stage 2
Systems programming / CUDA / low-level material

Stage 3
Specialized domain

Optional Stage 4
Held-out domain
```

Training:

```text
Stage 1 → train to convergence

Stage 2 → new data + replay

Stage 3 → new data + replay
```

Initial replay range:

```text
5–20% previous data
```

Optional regularization:

```text
EWC
```

Forecast heads may optionally receive separate protection/regularization.

---

# 15. Experiment Matrix

Minimum experiments:

| Experiment | Dense | Sparse | Forecast | Replay |
| ---------- | ----: | -----: | -------: | -----: |
| A          |     ✓ |        |          |        |
| B          |       |      ✓ |          |        |
| C          |       |      ✓ |        ✓ |        |
| D          |       |      ✓ |        ✓ |      ✓ |
| E          |     ✓ |        |          |      ✓ |

Do not combine every feature immediately.

Each experiment should isolate a specific effect.

---

# 16. Evaluation

### Language modeling

```text
training loss
validation loss
perplexity
```

### Forecast quality

```text
Recall@K
Precision@K
KL divergence
block-selection accuracy
```

### Inference

```text
tokens/sec
prefill latency
decode latency
latency/token
```

### Memory

```text
peak VRAM
model VRAM
activation VRAM
KV-cache VRAM
CPU memory
```

### Continual learning

For each stage:

```text
performance before new stage
performance after new stage
```

Calculate forgetting:

```text
Forgetting =
performance_before - performance_after
```

Also measure forward transfer:

```text
tokens/steps required to reach target validation loss
```

---

# 17. Required Repository Structure

Suggested structure:

```text
project/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── 1k.yaml
│   ├── 4k.yaml
│   ├── 8k.yaml
│   └── 16k.yaml
│
├── model/
│   ├── config.py
│   ├── embeddings.py
│   ├── norm.py
│   ├── rope.py
│   ├── attention/
│   │   ├── base.py
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── forecast.py
│   │   ├── offload.py
│   │   └── async_offload.py
│   ├── forecast.py
│   ├── transformer.py
│   └── model.py
│
├── data/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── packing.py
│   └── preprocessing.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── replay.py
│   ├── continual.py
│   └── checkpoint.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── forecast_metrics.py
│   ├── latency.py
│   ├── memory.py
│   └── forgetting.py
│
├── benchmarks/
│   ├── dense.py
│   ├── sparse.py
│   ├── forecast.py
│   └── long_context.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── benchmark.py
│
├── tests/
│   ├── test_attention.py
│   ├── test_forecast.py
│   ├── test_model.py
│   └── test_training.py
│
└── docs/
    ├── architecture.md
    ├── experiments.md
    └── results.md
```

---

# 18. Development Order

Implement in this order.

### Phase 1 — Base model

```text
Tokenizer
↓
Dataset pipeline
↓
Embeddings
↓
RoPE
↓
GQA
↓
RMSNorm
↓
SwiGLU
↓
Transformer
↓
LM head
```

Train a small run and verify learning.

### Phase 2 — Production-quality baseline

Add:

```text
BF16
Flash/SDPA attention
gradient checkpointing
checkpoint save/load
logging
validation
generation
VRAM profiling
tokens/sec profiling
```

### Phase 3 — Forecast

Add:

```text
Forecast projection
KV block representation
teacher selector
Forecast loss
Recall@K metrics
```

### Phase 4 — Sparse attention

Add:

```text
block selection
KV gather
sparse attention
```

Compare directly against dense attention.

### Phase 5 — KV offload

Add:

```text
CPU KV storage
selected-block transfer
VRAM measurements
transfer profiling
```

### Phase 6 — Async prefetch

Add:

```text
Forecast
↓
selection
↓
async KV transfer
↓
overlap with computation
```

### Phase 7 — Continual learning

Add:

```text
sequential datasets
replay buffer
forgetting metrics
optional EWC
```

### Phase 8 — Full experiment

Run:

```text
Dense
Sparse
Forecast
Forecast + Offload
Forecast + Async Offload
```

across:

```text
1K
2K
4K
8K
16K
32K
```

---

# 19. Success Criteria

The project is successful even if it does **not** beat dense attention everywhere.

The research questions are:

### Q1

Can a small model learn a useful one-layer-ahead Forecast?

### Q2

How accurately can Forecast predict the KV blocks required by the next layer?

### Q3

At what context length does sparse attention become beneficial?

### Q4

Does Forecast reduce KV transfer requirements?

### Q5

Can asynchronous prefetch hide enough transfer latency to produce a measurable wall-clock improvement?

### Q6

How does the mechanism behave during continual learning?

### Q7

Does replay interact differently with dense vs Forecast-based attention?

A negative result is still useful if the experiment identifies where the proposed architecture stops providing an advantage.

---

# 20. Core Design Principle

The project should **not** be built as "a tiny Qwen."

It should be built as:

> **A purpose-built research language model for experimentally studying Forecast-based sparse attention and long-context KV management on commodity 16 GB GPUs.**

The small model is intentional.

The objective is:

```text
fast iteration
+
controlled experiments
+
long-context scaling
+
measurable memory behavior
+
reproducible benchmarks
```

rather than maximizing language-model benchmark performance.

# 21. First Implementation Target

The first milestone should be extremely simple:

```text
Randomly initialized ~84M model
        ↓
Train on a small corpus
        ↓
Loss decreases
        ↓
Generate coherent-ish text
        ↓
Measure:
    VRAM
    tokens/sec
    validation loss
```

Only after this works should Forecast or sparse attention be introduced.

No pretrained model should be used at any stage as the model initialization.
