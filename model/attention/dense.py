"""Dense Grouped-Query Attention (GQA) implementation with PyTorch SDPA."""

import math
from typing import Optional, Tuple, Union
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

        # Transpose to [B, H, T, D] for attention computation
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

        causal_flag = is_causal if (seq_len > 1 and mask is None) else False
        return_attn_weights = kwargs.get("return_attn_weights", False)

        if return_attn_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(xq, xk.transpose(-1, -2)) * scale
            if causal_flag:
                t_k = xk.size(2)
                causal_mask = torch.triu(
                    torch.full((seq_len, t_k), float("-inf"), device=x.device, dtype=scores.dtype),
                    diagonal=1 + cache_offset,
                )
                scores = scores + causal_mask
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            output = torch.matmul(attn_weights, xv)
        else:
            # Use PyTorch's hardware-accelerated SDPA (FlashAttention-2 / CuDNN / math)
            output = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=causal_flag,
            )
            attn_weights = None

        # Reshape back to [B, T, n_head * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
        proj_out = self.wo(output)

        if return_attn_weights:
            return proj_out, attn_weights
        return proj_out

