"""Forecast Projection and Block Selection interfaces for SparDA architecture."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig


class ForecastProjection(nn.Module):
    """Forecast projection layer predicting next-layer KV requirements.
    
    wf = nn.Linear(n_embd, n_kv_head * forecast_dim, bias=False)
    Output shape: [B, T, forecast_heads, forecast_dim]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.forecast_heads = config.forecast_heads
        self.forecast_dim = config.forecast_dim
        
        self.wf = nn.Linear(
            self.n_embd,
            self.forecast_heads * self.forecast_dim,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects hidden state x into forecast representations.
        
        Args:
            x: [batch, seq_len, n_embd]
        Returns:
            forecast: [batch, seq_len, forecast_heads, forecast_dim]
        """
        batch_size, seq_len, _ = x.shape
        proj = self.wf(x)
        return proj.view(batch_size, seq_len, self.forecast_heads, self.forecast_dim)


class KVBlockCompressor(nn.Module):
    """Compresses KV blocks into representative prototype vectors for lightweight matching."""

    def __init__(self, block_size: int = 32, head_dim: int = 64, forecast_dim: int = 64):
        super().__init__()
        self.block_size = block_size
        self.head_dim = head_dim
        self.forecast_dim = forecast_dim
        
        # Linear map from key head_dim to forecast_dim if dimensions differ
        if head_dim != forecast_dim:
            self.proj = nn.Linear(head_dim, forecast_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Compresses full keys into block representations.
        
        Args:
            k: [batch, seq_len, n_kv_head, head_dim]
        Returns:
            block_reps: [batch, num_blocks, n_kv_head, forecast_dim]
        """
        batch_size, seq_len, n_kv_head, head_dim = k.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Pad to multiple of block_size if needed
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            
        # Reshape to [batch, num_blocks, block_size, n_kv_head, head_dim]
        k_blocks = k.view(batch_size, num_blocks, self.block_size, n_kv_head, head_dim)
        
        # Mean pooling across block_size tokens to form a block prototype
        block_mean = k_blocks.mean(dim=2)  # [batch, num_blocks, n_kv_head, head_dim]
        return self.proj(block_mean)
