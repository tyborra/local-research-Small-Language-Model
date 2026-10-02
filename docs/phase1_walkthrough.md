# Phase 1 Walkthrough: Base Model Architecture & Tokenizer

## Objective Completed
Milestone 1 from [SparDA-Style Local Research Model.md](../SparDA-Style%20Local%20Research%20Model.md) is complete:
- Initialized model from scratch (randomly initialized weights, no pretrained weights).
- Custom 32,000 Byte-fallback BPE tokenizer trained from raw text.
- Dataset preparation pipeline with uint16 binary sequence caching.
- Full Transformer architecture built: Tied Embeddings, RMSNorm, RoPE, GQA (12 Q heads, 4 KV heads), SwiGLU FFN, Forecast projection hook.
- Production BF16 training loop verified on a 16 GB NVIDIA RTX-Series GPU (Testbed: RTX 4090 Laptop GPU, 16 GB).
- Model trained for 100 steps: loss decreased monotonically from 9.93 to 6.20.
- Checkpoints saved atomically.
- Autoregressive text generation verified from checkpoint.
- All 12 unit tests passing.

---

## 1. Environment & Hardware Verification
- **Conda Environment**: `local_slm` (`C:\Users\tybor\miniconda3\envs\local_slm`)
- **Python**: `3.11.17`
- **PyTorch**: `2.6.0+cu124`
- **GPU Target**: 16 GB NVIDIA RTX-Series GPUs *(Testbed: NVIDIA GeForce RTX 4090 Laptop GPU, 16,376 MiB VRAM)*
- **Compute Capability**: `(8, 9)` (Ada Lovelace, native BF16 supported)
- **Power Profile (Eco Mode note)**: Development and Phase 1 runs were executed with the laptop in **Eco Mode with dGPU** (~80W TGP). All benchmark evaluations will be re-run in full **Performance Mode** (150W–175W TGP) upon project completion.

---

## 2. Architecture & Parameter Specification

```
Model Configuration:
├── vocab_size:       32,000 (tied input/output embeddings)
├── n_layer:          12
├── n_embd:           768
├── n_head:           12 (Query heads, head_dim 64)
├── n_kv_head:        4  (KV heads, GQA ratio 3:1)
├── ffn_dim:          2,048 (SwiGLU: w_gate, w_up, w_down)
├── forecast_heads:   4
├── forecast_dim:     64
└── context_length:   1,024

Parameter Breakdown:
├── Tied Embeddings:   24,576,000
├── Per Layer:
│   ├── Attention (Q, K, V, Out): 1,572,864
│   ├── Forecast Projection (Wf):   196,608
│   ├── SwiGLU (Gate, Up, Down):  4,718,592
│   └── RMSNorms (x2):               1,536
│   └── Layer Total:               6,489,600
├── 12 Layers Total:   77,875,200
├── Final RMSNorm:            768
└── Total Model:      102,451,968 parameters
```

---

## 3. Files Implemented in Project Root

All files have been written directly to the project repository:

```
local-research-LLM/
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── base.yaml                 # Base hyperparameters & training config
│
├── model/
│   ├── __init__.py
│   ├── config.py                 # ModelConfig dataclass
│   ├── norm.py                   # RMSNorm with FP32 precision stability
│   ├── rope.py                   # Rotary Position Embeddings with cache expansion
│   ├── embeddings.py             # TiedEmbedding for shared vocab weights
│   ├── forecast.py               # ForecastProjection & KVBlockCompressor
│   ├── transformer.py            # SwiGLU MLP & TransformerBlock
│   ├── model.py                  # SparDAResearchModel & generate()
│   └── attention/
│       ├── __init__.py
│       ├── base.py               # BaseAttention & KVCache
│       └── dense.py              # Dense GQA with PyTorch SDPA
│
├── data/
│   ├── __init__.py
│   ├── tokenizer.py              # SparDATokenizer (Byte-fallback BPE 32K)
│   ├── dataset.py                # TokenizedDataset & DataLoader
│   └── packing.py                # Sequence packing utilities
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                # BF16 training loop & performance telemetry
│   └── checkpoint.py             # Atomic checkpoint save and load
│
├── scripts/
│   ├── prepare_data.py           # Corpus download & tokenizer training
│   ├── train.py                  # Training CLI runner
│   └── generate.py               # Autoregressive generation CLI
│
└── tests/
    ├── __init__.py
    ├── test_attention.py         # RoPE, GQA, and KV-cache decoding tests
    ├── test_model.py             # Parameter count, tying, forward, backward tests
    ├── test_data.py              # Tokenizer, Dataset, Packing tests
    └── test_checkpoint.py        # Checkpoint roundtrip tests
```

---

## 4. Test Suite Results

```text
============================= test session starts =============================
platform win32 -- Python 3.11.17, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\tybor\GitHub\local-research-LLM
configfile: pyproject.toml
testpaths: tests

tests/test_attention.py::test_rope_frequencies PASSED                    [  8%]
tests/test_attention.py::test_gqa_dimensions_and_causal PASSED           [ 16%]
tests/test_attention.py::test_kv_cache_decoding PASSED                   [ 25%]
tests/test_checkpoint.py::test_checkpoint_roundtrip PASSED               [ 33%]
tests/test_data.py::test_tokenizer_training_and_encoding PASSED          [ 41%]
tests/test_data.py::test_tokenized_dataset PASSED                        [ 50%]
tests/test_data.py::test_pack_sequences PASSED                           [ 58%]
tests/test_model.py::test_model_parameter_count PASSED                   [ 66%]
tests/test_model.py::test_tied_embeddings PASSED                         [ 75%]
tests/test_model.py::test_forward_pass_and_loss PASSED                   [ 83%]
tests/test_model.py::test_backward_pass PASSED                           [ 91%]
tests/test_model.py::test_generate PASSED                                [100%]

============================= 12 passed in 21.77s =============================
```

---

## 5. Phase 1 Training Run & Generation Output

### Training Log (100 Steps on RTX 4090, BF16)
```text
Loading training data from data/train.bin...
Loaded 2,587,227 training tokens.
Loaded 241,420 validation tokens.
Initialized SparDAResearchModel (102,451,968 parameters)
Starting training run on cuda (dtype: torch.bfloat16)
Total steps: 100, Grad Accum: 4, Context: 1024

Step    10/  100 | loss: 9.9352 | lr: 5.94e-05 | tok/s:   10350 | VRAM: 3.23 GB
Step    20/  100 | loss: 8.7223 | lr: 1.19e-04 | tok/s:   11514 | VRAM: 3.23 GB
Step    30/  100 | loss: 8.0001 | lr: 1.78e-04 | tok/s:   11752 | VRAM: 3.23 GB
Step    40/  100 | loss: 7.2073 | lr: 2.38e-04 | tok/s:   11577 | VRAM: 3.23 GB
Step    50/  100 | loss: 6.9000 | lr: 2.97e-04 | tok/s:   11862 | VRAM: 3.23 GB
Step    60/  100 | loss: 6.7396 | lr: 3.56e-04 | tok/s:   12254 | VRAM: 3.23 GB
Step    70/  100 | loss: 6.5821 | lr: 4.16e-04 | tok/s:   12182 | VRAM: 3.23 GB
Step    80/  100 | loss: 6.4832 | lr: 4.75e-04 | tok/s:   11319 | VRAM: 3.23 GB
Step    90/  100 | loss: 6.3603 | lr: 5.35e-04 | tok/s:   11423 | VRAM: 3.23 GB
Step   100/  100 | loss: 6.2056 | val_loss: 5.7643 | lr: 5.94e-04 | tok/s:   12023 | VRAM: 3.23 GB

Saved checkpoint to checkpoints/base/model_step_100.pt
Training completed in 145.9s (1,638,400 total tokens processed)
```

### Text Generation from Checkpoint
```text
Command: python scripts/generate.py --checkpoint checkpoints/base/model_step_100.pt --prompt "The model is trained from scratch" --max_new_tokens 40

Prompt: The model is trained from scratch
Generated Output:
The model is trained from scratch and four . The " As the  "   that " ) of several have be of  , the first time of the year at  . 
 The next ,
```
