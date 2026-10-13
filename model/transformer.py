"""Transformer block with SwiGLU and pluggable attention backend."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig
from model.norm import RMSNorm
from model.rope import RotaryEmbedding
from model.forecast import ForecastProjection
from model.attention.base import BaseAttention, KVCache
from model.attention.dense import DenseAttention
from model.attention.sparse import SparseAttention


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.
    
    output = (SiLU(x @ w_gate) * (x @ w_up)) @ w_down
    """

    def __init__(self, n_embd: int, ffn_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, ffn_dim, bias=False)
        self.w_up = nn.Linear(n_embd, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def build_attention_backend(config: ModelConfig, layer_idx: int) -> BaseAttention:
    """Factory to instantiate the appropriate attention backend."""
    backend = config.attention_backend.lower()
    if backend == "dense":
        return DenseAttention(config, layer_idx)
    elif backend == "sparse":
        return SparseAttention(config, layer_idx)
    elif backend in ("forecast", "offload", "async_offload"):
        # Placeholders for future phases will be hooked here
        return DenseAttention(config, layer_idx)
    else:
        raise ValueError(f"Unknown attention backend: {config.attention_backend}")



class TransformerBlock(nn.Module):
    """Single Transformer layer containing RMSNorm, Attention, Forecast, and SwiGLU."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-attention normalization
        self.attn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        # Modular attention backend
        self.attention = build_attention_backend(config, layer_idx)
        # Forecast projection predicting layer L+1 requirements
        self.forecast = ForecastProjection(config)
        
        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        # SwiGLU FFN
        self.ffn = SwiGLU(config.n_embd, config.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        prev_layer_forecast: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            rotary_emb: Rotary position embeddings
            kv_cache: Optional KV cache
            mask: Optional attention mask
            is_causal: Whether to apply causal mask
            prev_layer_forecast: Forecast vector emitted by layer L-1
            
        Returns:
            x: Updated hidden states [batch, seq_len, n_embd]
            current_forecast: Forecast vector for layer L+1 [batch, seq_len, forecast_heads, forecast_dim]
        """
        # 1. Normalize input for attention and forecast
        normed_x = self.attn_norm(x)
        
        # 2. Compute forecast for the next layer (L+1)
        current_forecast = self.forecast(normed_x)
        
        # 3. Compute attention using backend
        attn_out = self.attention(
            normed_x,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            mask=mask,
            is_causal=is_causal,
            forecast_vector=prev_layer_forecast,
        )
        x = x + attn_out
        
        # 4. Normalize & SwiGLU FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, current_forecast
