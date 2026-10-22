"""Multi-stage dataset pipeline for Continual Learning experiments.

Implements Section 14 of the SparDA specification:
- Stage 1: General Python & Technical Documentation.
- Stage 2: Low-Level Systems Programming & CUDA C++ Kernels.
- Stage 3: Specialized Domain (Mathematical Reasoning / Algorithmic Code).
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import TokenizedDataset, create_dataloader
from data.tokenizer import SparDATokenizer

# Stage 1: General Python & Technical Documentation Corpus
STAGE1_PYTHON_DOCS = """
def quicksort(arr: list[int]) -> list[int]:
    \"\"\"Sorts a list of integers using the divide-and-conquer quicksort algorithm.
    
    Time Complexity:
        Average: O(n log n)
        Worst: O(n^2) when array is already sorted and pivot is poorly chosen.
    Space Complexity:
        O(log n) auxiliary stack space for recursion.
    \"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class LRUCache:
    \"\"\"Least Recently Used (LRU) Cache implementation with O(1) operations.
    
    Utilizes a doubly linked list coupled with a hash map to achieve constant time
    lookup, insertion, and eviction.
    \"\"\"
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: str) -> Optional[int]:
        if key not in self.cache:
            return None
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: str, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

import asyncio
from typing import AsyncGenerator

async def fetch_telemetry_stream(endpoint: str, timeout: float = 5.0) -> AsyncGenerator[dict, None]:
    \"\"\"Asynchronously polls telemetry events over HTTP/2 streaming socket.\"\"\"
    reader, writer = await asyncio.open_connection('127.0.0.1', 8080)
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            yield {"raw": line.decode().strip(), "status": 200}
    finally:
        writer.close()
        await writer.wait_closed()
""" * 20

# Stage 2: Systems Programming, CUDA C++, & Low-level Memory
STAGE2_SYSTEMS_CUDA = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;

    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (row < N && m * BLOCK_SIZE + tx < N)
            s_a[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        else
            s_a[ty][tx] = 0.0f;

        if (col < N && m * BLOCK_SIZE + ty < N)
            s_b[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            s_b[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_a[ty][k] * s_b[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_cuda_dma_prefetch(float* d_dst, const float* h_src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA DMA Transfer Error: %s\\n", cudaGetErrorString(err));
    }
}
""" * 20

# Stage 3: Mathematical Reasoning & Algorithmic Code
STAGE3_MATH_ALGORITHMS = """
import numpy as np

def fast_fourier_transform(x: np.ndarray) -> np.ndarray:
    \"\"\"Computes the Discrete Fourier Transform using the Cooley-Tukey Radix-2 FFT algorithm.
    
    Theorem:
        X[k] = E[k] + exp(-2pi * i * k / N) * O[k]
        where E[k] is FFT of even indices and O[k] is FFT of odd indices.
    \"\"\"
    N = len(x)
    if N <= 1:
        return x
    even = fast_fourier_transform(x[0::2])
    odd = fast_fourier_transform(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        even + factor[:N // 2] * odd,
        even + factor[N // 2:] * odd
    ])

def modular_exponentiation(base: int, exp: int, mod: int) -> int:
    \"\"\"Computes (base^exp) % mod in O(log exp) operations using binary exponentiation.
    
    Fermat's Little Theorem:
        If p is prime, a^(p-1) == 1 (mod p).
    \"\"\"
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def solve_linear_system_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"Solves A x = b for symmetric positive definite matrix A using Cholesky factor L L^T.\"\"\"
    L = np.linalg.cholesky(A)
    # Forward substitution: L y = b
    y = np.linalg.solve(L, b)
    # Backward substitution: L^T x = y
    x = np.linalg.solve(L.T, y)
    return x
""" * 20


@dataclass
class ContinualStage:
    """Represents a training stage in continual learning."""
    stage_idx: int
    name: str
    description: str
    train_tokens: np.ndarray
    val_tokens: np.ndarray
    train_dataset: TokenizedDataset
    val_dataset: TokenizedDataset


class ContinualDatasetPipeline:
    """Manages multi-stage datasets, tokenization, and validation splits."""

    def __init__(
        self,
        tokenizer: Optional[SparDATokenizer] = None,
        seq_len: int = 128,
        val_split: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.val_split = val_split

        if self.tokenizer is None:
            # Fallback to local tokenizer if available
            tok_path = "data/tokenizer.json"
            if os.path.exists(tok_path):
                self.tokenizer = SparDATokenizer.from_file(tok_path)

        self.stages_text: Dict[int, Dict[str, str]] = {
            1: {
                "name": "python_technical",
                "desc": "General Python & Technical Documentation",
                "text": STAGE1_PYTHON_DOCS,
            },
            2: {
                "name": "systems_cuda",
                "desc": "Systems Programming & CUDA C++ Kernels",
                "text": STAGE2_SYSTEMS_CUDA,
            },
            3: {
                "name": "math_algorithms",
                "desc": "Mathematical Reasoning & Algorithmic Code",
                "text": STAGE3_MATH_ALGORITHMS,
            },
        }

    def get_stage(self, stage_idx: int) -> ContinualStage:
        """Tokenizes and returns a ContinualStage instance."""
        if stage_idx not in self.stages_text:
            raise ValueError(f"Stage {stage_idx} not found. Available stages: {list(self.stages_text.keys())}")

        stage_info = self.stages_text[stage_idx]
        raw_text = stage_info["text"]

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(raw_text)
        else:
            # Simple byte-level fallback for testing without loaded tokenizer
            token_ids = list(raw_text.encode("utf-8"))

        arr = np.array(token_ids, dtype=np.uint16)
        split_idx = int(len(arr) * (1.0 - self.val_split))
        train_tokens = arr[:split_idx]
        val_tokens = arr[split_idx:]

        # Ensure minimum length for seq_len + 1
        min_tokens = (self.seq_len + 1) * 2
        if len(train_tokens) < min_tokens:
            train_tokens = np.tile(train_tokens, int(np.ceil(min_tokens / max(len(train_tokens), 1))))
        if len(val_tokens) < min_tokens:
            val_tokens = np.tile(val_tokens, int(np.ceil(min_tokens / max(len(val_tokens), 1))))

        train_ds = TokenizedDataset(train_tokens, seq_len=self.seq_len)
        val_ds = TokenizedDataset(val_tokens, seq_len=self.seq_len)

        return ContinualStage(
            stage_idx=stage_idx,
            name=stage_info["name"],
            description=stage_info["desc"],
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            train_dataset=train_ds,
            val_dataset=val_ds,
        )

    def get_all_stages(self) -> List[ContinualStage]:
        """Returns all configured continual learning stages."""
        return [self.get_stage(idx) for idx in sorted(self.stages_text.keys())]


# Convenient alias
ContinualDatasetManager = ContinualDatasetPipeline
