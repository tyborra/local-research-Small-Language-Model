"""Model configuration for the SparDA-Style Local Research Model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    # Model architecture dimensions
    vocab_size: int = 32000
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    head_dim: int = 64
    ffn_dim: int = 2048
    
    # Forecast parameters
    forecast_heads: int = 4
    forecast_dim: int = 64
    
    # Context & Positional Encoding
    context_length: int = 1024
    rope_theta: float = 10000.0
    
    # Regularization & Initialization
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    
    # Attention backend selector
    # Options: "dense", "sparse", "forecast", "offload", "async_offload"
    attention_backend: str = "dense"
    sparse_mode: str = "forecast"
    
    # Sparse / Block settings
    block_size: int = 32
    top_k_blocks: int = 4
    num_queries_per_kv: int = 3

    def __post_init__(self):
        assert self.n_embd == self.n_head * self.head_dim, (
            f"n_embd ({self.n_embd}) must equal n_head ({self.n_head}) * head_dim ({self.head_dim})"
        )
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        )
        self.num_queries_per_kv = self.n_head // self.n_kv_head
