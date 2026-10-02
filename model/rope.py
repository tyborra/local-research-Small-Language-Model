"""Rotary Position Embedding (RoPE) implementation."""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes complex exponential frequency bands for RoPE."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # theta_i = 1.0 / (theta ** (2i / d)) for i in [0, 1, ..., d/2 - 1]
    dim_indices = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    
    t = torch.arange(max_seq_len).float()
    # Outer product: [max_seq_len, head_dim // 2]
    freqs = torch.outer(t, inv_freq)
    
    # Precompute cos and sin: shapes [max_seq_len, head_dim // 2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    # Split the last dimension into pairs: [..., d//2], [..., d//2]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors.
    
    Args:
        xq: [batch, seq_len, num_q_heads, head_dim]
        xk: [batch, seq_len, num_kv_heads, head_dim]
        cos: [max_seq_len, head_dim // 2]
        sin: [max_seq_len, head_dim // 2]
        offset: starting position (for incremental KV-cache decoding)
    """
    seq_len = xq.shape[1]
    # Slice cos and sin for current sequence range: [seq_len, head_dim // 2]
    cos_seq = cos[offset : offset + seq_len, :]
    sin_seq = sin[offset : offset + seq_len, :]
    
    # Repeat along the half dimension so shape is [1, seq_len, 1, head_dim]
    cos_seq = torch.cat([cos_seq, cos_seq], dim=-1).unsqueeze(0).unsqueeze(2)
    sin_seq = torch.cat([sin_seq, sin_seq], dim=-1).unsqueeze(0).unsqueeze(2)
    
    # Cast to match query/key dtype
    cos_seq = cos_seq.type_as(xq)
    sin_seq = sin_seq.type_as(xq)
    
    xq_out = (xq * cos_seq) + (rotate_half(xq) * sin_seq)
    xk_out = (xk * cos_seq) + (rotate_half(xk) * sin_seq)
    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """Caches precomputed RoPE frequencies up to max_seq_len."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _extend_cache(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len:
            self.max_seq_len = max(seq_len, self.max_seq_len * 2)
            cos, sin = precompute_rope_freqs(self.head_dim, self.max_seq_len, self.theta)
            self.cos = cos.to(device)
            self.sin = sin.to(device)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = xq.shape[1]
        total_len = offset + seq_len
        if total_len > self.max_seq_len:
            self._extend_cache(total_len, xq.device)
            
        return apply_rotary_emb(xq, xk, self.cos, self.sin, offset=offset)
