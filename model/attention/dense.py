"""Dense Grouped-Query Attention (GQA) implementation with PyTorch SDPA."""

from typing import Optional
import torch
import torch.nn.functional as F
from model.attention.base import BaseAttention, KVCache
from model.config import ModelConfig
from model.rope import RotaryEmbedding


class DenseAttention(BaseAttention):
    """Dense GQA Attention using PyTorch's optimized scaled_dot_product_attention (Flash/SDPA)."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projections: [B, T, H, D]
        xq = self.wq(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        # Apply RoPE
        cache_offset = kv_cache.current_len if kv_cache is not None else 0
        xq, xk = rotary_emb(xq, xk, offset=cache_offset)

        # Handle KV cache update during autoregressive decoding
        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)

        # Transpose to [B, H, T, D] for SDPA
        # xq: [B, n_head, T_q, head_dim]
        # xk: [B, n_kv_head, T_k, head_dim]
        # xv: [B, n_kv_head, T_k, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # GQA expansion: repeat KV heads to match query heads
        if self.num_queries_per_kv > 1:
            xk = torch.repeat_interleave(xk, repeats=self.num_queries_per_kv, dim=1)
            xv = torch.repeat_interleave(xv, repeats=self.num_queries_per_kv, dim=1)

        # Use PyTorch's hardware-accelerated SDPA (FlashAttention-2 / CuDNN / math)
        # If cache_offset > 0, we only generate 1 token at a time, so is_causal should be False if mask is not used
        causal_flag = is_causal if (seq_len > 1 and mask is None) else False
        
        output = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=causal_flag
        )

        # Reshape back to [B, T, n_head * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
        return self.wo(output)
