"""Forecast Projection, KV Block Compression, and Block Selection interfaces for SparDA architecture."""

import math
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig


class ForecastProjection(nn.Module):
    """Forecast projection layer predicting next-layer (L+1) KV requirements from layer L hidden states.
    
    wf = nn.Linear(n_embd, forecast_heads * forecast_dim, bias=False)
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
    """Compresses key states into block prototype vectors for lightweight matching."""

    def __init__(self, block_size: int = 32, head_dim: int = 64, forecast_dim: int = 64):
        super().__init__()
        self.block_size = block_size
        self.head_dim = head_dim
        self.forecast_dim = forecast_dim
        
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
        
        # Mean pooling across tokens in block to form prototype
        block_mean = k_blocks.mean(dim=2)  # [batch, num_blocks, n_kv_head, head_dim]
        return self.proj(block_mean)


class ForecastBlockSelector(nn.Module):
    """Matches forecast vectors against compressed KV block prototypes to select Top-K blocks."""

    def __init__(self, block_size: int = 32, forecast_dim: int = 64):
        super().__init__()
        self.block_size = block_size
        self.forecast_dim = forecast_dim

    def forward(
        self,
        forecast: torch.Tensor,
        block_reps: torch.Tensor,
        top_k: int,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates block scores and selects Top-K blocks.
        
        Args:
            forecast: [batch, seq_len_q, forecast_heads, forecast_dim]
            block_reps: [batch, num_blocks, n_kv_head, forecast_dim]
            top_k: Number of blocks to select
            is_causal: Whether to apply causal masking to block selection
            
        Returns:
            scores: [batch, n_kv_head, seq_len_q, num_blocks]
            topk_indices: [batch, n_kv_head, seq_len_q, top_k]
        """
        batch_size, seq_len_q, n_f_heads, f_dim = forecast.shape
        _, num_blocks, n_kv_head, _ = block_reps.shape
        assert n_f_heads == n_kv_head, f"Mismatch: forecast_heads ({n_f_heads}) != n_kv_head ({n_kv_head})"

        # Permute to [batch, heads, seq, dim] for batched dot-product
        f = forecast.permute(0, 2, 1, 3)     # [B, H, T_q, D]
        b = block_reps.permute(0, 2, 1, 3)   # [B, H, N_b, D]

        # Scaled dot-product scores: [B, H, T_q, N_b]
        scale = 1.0 / math.sqrt(f_dim)
        scores = torch.matmul(f, b.transpose(-1, -2)) * scale

        if is_causal:
            # Token at index t belongs to block t // block_size
            # Block indices j > t // block_size are strictly in the future
            t_indices = torch.arange(seq_len_q, device=forecast.device).view(1, 1, seq_len_q, 1)
            b_indices = torch.arange(num_blocks, device=forecast.device).view(1, 1, 1, num_blocks)
            future_mask = b_indices > (t_indices // self.block_size)
            scores = scores.masked_fill(future_mask, float("-inf"))

        k_val = min(top_k, num_blocks)
        topk_indices = torch.topk(scores, k=k_val, dim=-1).indices
        return scores, topk_indices


class TeacherBlockSelector(nn.Module):
    """Extracts ground-truth block importance targets from full dense attention weights."""

    def __init__(self, block_size: int = 32, num_queries_per_kv: int = 3):
        super().__init__()
        self.block_size = block_size
        self.num_queries_per_kv = num_queries_per_kv

    def forward(
        self,
        attn_weights: torch.Tensor,
        top_k: int,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates teacher block importance from dense attention weights.
        
        Args:
            attn_weights: [batch, n_head, seq_len_q, seq_len_k] (post-softmax attention weights)
            top_k: Number of ground-truth blocks to extract
            is_causal: Whether attention is causal
            
        Returns:
            block_scores: [batch, n_kv_head, seq_len_q, num_blocks]
            topk_indices: [batch, n_kv_head, seq_len_q, top_k]
            target_distribution: [batch, n_kv_head, seq_len_q, num_blocks] (normalized probability)
        """
        batch_size, n_head, seq_len_q, seq_len_k = attn_weights.shape
        n_kv_head = n_head // self.num_queries_per_kv

        # 1. Group query heads by KV head and average
        # [B, n_kv_head, queries_per_kv, T_q, T_k] -> [B, n_kv_head, T_q, T_k]
        kv_attn = attn_weights.view(
            batch_size, n_kv_head, self.num_queries_per_kv, seq_len_q, seq_len_k
        ).mean(dim=2)

        # 2. Divide key length into blocks
        num_blocks = (seq_len_k + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - seq_len_k
        if pad_len > 0:
            kv_attn = F.pad(kv_attn, (0, pad_len))

        # [B, n_kv_head, T_q, num_blocks, block_size]
        kv_attn_blocks = kv_attn.view(batch_size, n_kv_head, seq_len_q, num_blocks, self.block_size)
        
        # Total attention weight assigned to each block
        block_scores = kv_attn_blocks.sum(dim=-1)  # [B, n_kv_head, T_q, num_blocks]

        # 3. Mask future blocks if causal
        if is_causal:
            t_indices = torch.arange(seq_len_q, device=attn_weights.device).view(1, 1, seq_len_q, 1)
            b_indices = torch.arange(num_blocks, device=attn_weights.device).view(1, 1, 1, num_blocks)
            future_mask = b_indices > (t_indices // self.block_size)
            block_scores = block_scores.masked_fill(future_mask, 0.0)

        # Compute normalized target probability distribution across valid blocks
        eps = 1e-8
        sums = block_scores.sum(dim=-1, keepdim=True) + eps
        target_distribution = block_scores / sums

        # Get Top-K block indices
        k_val = min(top_k, num_blocks)
        topk_indices = torch.topk(block_scores, k=k_val, dim=-1).indices

        return block_scores, topk_indices, target_distribution


def compute_forecast_loss(
    pred_scores: torch.Tensor,
    target_distribution: torch.Tensor,
    loss_type: str = "kl",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Computes auxiliary training loss between forecast block scores and teacher distribution.
    
    Args:
        pred_scores: [batch, n_kv_head, seq_len_q, num_blocks] (raw forecast logits)
        target_distribution: [batch, n_kv_head, seq_len_q, num_blocks] (teacher probability mass)
        loss_type: "kl" (KL-Divergence) or "bce" (Binary Cross Entropy with multi-hot targets)
        temperature: Softmax temperature for predicted logits
        
    Returns:
        Scalar loss tensor
    """
    valid_mask = torch.isfinite(pred_scores) & (target_distribution.sum(dim=-1, keepdim=True) > 0)
    
    if loss_type == "kl":
        # Log-softmax of predicted logits over valid block indices
        log_preds = F.log_softmax(pred_scores / temperature, dim=-1)
        # Replace non-finite log_preds with 0 to prevent NaN backpropagation
        log_preds = torch.where(torch.isfinite(log_preds), log_preds, torch.zeros_like(log_preds))
        # PyTorch F.kl_div expects input=log_preds, target=target_distribution
        kl = F.kl_div(log_preds, target_distribution, reduction="none")
        kl = torch.where(valid_mask, kl, torch.zeros_like(kl))
        num_valid = valid_mask.sum().clamp(min=1)
        return kl.sum() / num_valid
        
    elif loss_type == "bce":
        # Multi-hot target (e.g. non-zero probability blocks)
        target_binary = (target_distribution > 0).float()
        clean_preds = torch.where(torch.isfinite(pred_scores), pred_scores, torch.full_like(pred_scores, -100.0))
        bce = F.binary_cross_entropy_with_logits(clean_preds, target_binary, reduction="none")
        bce = torch.where(valid_mask, bce, torch.zeros_like(bce))
        num_valid = valid_mask.sum().clamp(min=1)
        return bce.sum() / num_valid
    else:
        raise ValueError(f"Unknown forecast loss type: {loss_type}")
