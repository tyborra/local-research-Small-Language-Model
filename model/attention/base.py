"""Base interface and KV Cache definition for modular attention backends."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from model.config import ModelConfig
from model.rope import RotaryEmbedding


class KVCache:
    """Manages Key-Value states for autoregressive decoding or offloading."""

    def __init__(self, max_batch_size: int, max_seq_len: int, n_kv_head: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.k = torch.zeros((max_batch_size, max_seq_len, n_kv_head, head_dim), device=device, dtype=dtype)
        self.v = torch.zeros((max_batch_size, max_seq_len, n_kv_head, head_dim), device=device, dtype=dtype)
        self.current_len = 0

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new k and v slices and returns full accumulated cache up to current length."""
        seq_len = xk.shape[1]
        start_idx = self.current_len
        end_idx = start_idx + seq_len
        
        self.k[:, start_idx:end_idx] = xk
        self.v[:, start_idx:end_idx] = xv
        self.current_len = end_idx
        
        return self.k[:, :self.current_len], self.v[:, :self.current_len]

    def reset(self):
        self.current_len = 0


class BaseAttention(nn.Module, ABC):
    """Abstract interface that all attention backends (Dense, Sparse, Forecast) must implement."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.num_queries_per_kv = config.num_queries_per_kv

        # Projections
        self.wq = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Executes attention.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            rotary_emb: RoPE module
            kv_cache: Optional persistent KV cache
            mask: Optional explicit attention mask
            is_causal: Whether to apply causal masking
        """
        pass
