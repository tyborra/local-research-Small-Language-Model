"""Sparse Grouped-Query Attention (GQA) backend with Forecast-guided Block Selection.

Implements block gathering, local causal preservation, attention sink preservation,
and forecast-guided Top-K block selection according to SparDA specification.
"""

import math
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention, KVCache
from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.forecast import KVBlockCompressor, ForecastBlockSelector


def gather_kv_blocks(
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gathers selected K and V blocks along the sequence dimension.
    
    Args:
        k: Key tensor [batch, n_kv_head, seq_len_k, head_dim]
        v: Value tensor [batch, n_kv_head, seq_len_k, head_dim]
        block_indices: Block indices [batch, n_kv_head, num_selected_blocks]
        block_size: Number of tokens per block
        
    Returns:
        k_gathered: [batch, n_kv_head, num_selected_blocks * block_size, head_dim]
        v_gathered: [batch, n_kv_head, num_selected_blocks * block_size, head_dim]
    """
    batch_size, n_kv_head, seq_len_k, head_dim = k.shape
    num_blocks = (seq_len_k + block_size - 1) // block_size
    pad_len = num_blocks * block_size - seq_len_k
    
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        
    num_selected = block_indices.shape[-1]
    
    # Reshape k and v to [batch, n_kv_head, num_blocks, block_size, head_dim]
    k_blocks = k.view(batch_size, n_kv_head, num_blocks, block_size, head_dim)
    v_blocks = v.view(batch_size, n_kv_head, num_blocks, block_size, head_dim)
    
    # Expand indices for gather: [batch, n_kv_head, num_selected, block_size, head_dim]
    idx_expanded = block_indices.view(batch_size, n_kv_head, num_selected, 1, 1).expand(
        batch_size, n_kv_head, num_selected, block_size, head_dim
    )
    
    k_gathered = torch.gather(k_blocks, dim=2, index=idx_expanded)
    v_gathered = torch.gather(v_blocks, dim=2, index=idx_expanded)
    
    k_gathered = k_gathered.view(batch_size, n_kv_head, num_selected * block_size, head_dim)
    v_gathered = v_gathered.view(batch_size, n_kv_head, num_selected * block_size, head_dim)
    
    return k_gathered, v_gathered


class SparseAttention(BaseAttention):
    """Sparse GQA Attention with forecast-guided Top-K block selection.
    
    Supports:
    1. Single-token decode with physical KV block gather (crucial for memory reduction & offload).
    2. Sequence prefill with block-sparse masked SDPA (fast vectorized training and prompt ingestion).
    3. Sequence prefill with chunked block gather (profiling physical block gather overhead).
    """

    def __init__(self, config: ModelConfig, layer_idx: int, mode: Optional[str] = None):
        super().__init__(config, layer_idx)
        self.block_size = config.block_size
        self.top_k_blocks = config.top_k_blocks
        
        if mode is not None:
            self.mode = mode
        elif config.attention_backend in ("static_sparse", "static"):
            self.mode = "static"
        elif getattr(config, "sparse_mode", "forecast") == "static":
            self.mode = "static"
        else:
            self.mode = "forecast"
        
        # Compressor and selector modules
        self.compressor = KVBlockCompressor(
            block_size=self.block_size,
            head_dim=self.head_dim,
            forecast_dim=config.forecast_dim,
        )
        self.selector = ForecastBlockSelector(
            block_size=self.block_size,
            forecast_dim=config.forecast_dim,
        )

    def _forward_decode_gather(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_len_k: int,
        forecast_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes decode attention using physical block gathering."""
        batch_size, n_kv_head, _, head_dim = xk.shape
        num_blocks = (seq_len_k + self.block_size - 1) // self.block_size
        curr_block = (seq_len_k - 1) // self.block_size

        # If total blocks are small enough, attend densely without dropping blocks
        max_sel_blocks = min(num_blocks, self.top_k_blocks + 2)
        if num_blocks <= max_sel_blocks:
            # Fall back to dense decode over accumulated cache
            xk_dense = xk[:, :, :seq_len_k, :]
            xv_dense = xv[:, :, :seq_len_k, :]
            if self.num_queries_per_kv > 1:
                xk_dense = torch.repeat_interleave(xk_dense, repeats=self.num_queries_per_kv, dim=1)
                xv_dense = torch.repeat_interleave(xv_dense, repeats=self.num_queries_per_kv, dim=1)
            out = F.scaled_dot_product_attention(xq, xk_dense, xv_dense, is_causal=False)
            out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
            return self.wo(out)

        # 1. Selection mask across blocks
        selected_mask = torch.zeros((batch_size, n_kv_head, num_blocks), dtype=torch.bool, device=xk.device)
        sink_idx = torch.zeros((batch_size, n_kv_head, 1), dtype=torch.long, device=xk.device)
        curr_idx = torch.full((batch_size, n_kv_head, 1), curr_block, dtype=torch.long, device=xk.device)
        selected_mask.scatter_(dim=-1, index=sink_idx, value=True)
        selected_mask.scatter_(dim=-1, index=curr_idx, value=True)

        if self.mode == "forecast" and forecast_vector is not None:
            # Compress keys into block representations: [B, num_blocks, n_kv_head, forecast_dim]
            xk_for_comp = xk.transpose(1, 2)  # [B, seq_len_k, n_kv_head, head_dim]
            block_reps = self.compressor(xk_for_comp)

            # Score blocks with forecast vector from layer L-1
            # forecast_vector: [B, 1, n_kv_head, forecast_dim]
            _, topk = self.selector(forecast_vector, block_reps, top_k=self.top_k_blocks, is_causal=True)
            topk_indices = topk[:, :, 0, :]  # [B, n_kv_head, top_k]
            selected_mask.scatter_(dim=-1, index=topk_indices.clamp(0, num_blocks - 1), value=True)
        else:
            # Static mode / no forecast: select local window blocks ending at curr_block
            local_start = max(0, curr_block - self.top_k_blocks + 1)
            for b in range(local_start, curr_block + 1):
                idx = torch.full((batch_size, n_kv_head, 1), b, dtype=torch.long, device=xk.device)
                selected_mask.scatter_(dim=-1, index=idx, value=True)

        # Mask future blocks if any
        future_mask = torch.arange(num_blocks, device=xk.device) > curr_block
        selected_mask[:, :, future_mask] = False

        # Extract fixed-size block indices per head
        val, sel_indices = torch.topk(selected_mask.float(), k=max_sel_blocks, dim=-1)

        # 4. Physically gather selected blocks from KV cache
        k_gat, v_gat = gather_kv_blocks(xk, xv, sel_indices, self.block_size)

        # 5. Build attention mask for gathered tokens
        offsets = torch.arange(self.block_size, device=xk.device).view(1, 1, 1, self.block_size)
        token_pos = (sel_indices.unsqueeze(-1) * self.block_size + offsets).view(
            batch_size, n_kv_head, 1, max_sel_blocks * self.block_size
        )
        block_valid = (val > 0.5).unsqueeze(-1).expand(-1, -1, -1, self.block_size).reshape(
            batch_size, n_kv_head, 1, max_sel_blocks * self.block_size
        )
        is_valid = (token_pos < seq_len_k) & block_valid

        attn_mask = torch.where(
            is_valid,
            torch.zeros(1, dtype=xq.dtype, device=xq.device),
            torch.full((1,), float("-inf"), dtype=xq.dtype, device=xq.device),
        )

        # Expand KV heads to match query heads
        if self.num_queries_per_kv > 1:
            k_gat = torch.repeat_interleave(k_gat, repeats=self.num_queries_per_kv, dim=1)
            v_gat = torch.repeat_interleave(v_gat, repeats=self.num_queries_per_kv, dim=1)
            attn_mask = torch.repeat_interleave(attn_mask, repeats=self.num_queries_per_kv, dim=1)

        out = F.scaled_dot_product_attention(xq, k_gat, v_gat, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
        return self.wo(out)

    def _forward_masked_sdpa(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_len: int,
        forecast_vector: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Executes full-sequence sparse attention using block-sparse causal masking."""
        batch_size = xq.shape[0]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # If context is within local/sink range, execute dense attention
        if num_blocks <= 2 or (self.mode == "forecast" and (forecast_vector is None or self.layer_idx == 0)):
            if self.num_queries_per_kv > 1:
                xk = torch.repeat_interleave(xk, repeats=self.num_queries_per_kv, dim=1)
                xv = torch.repeat_interleave(xv, repeats=self.num_queries_per_kv, dim=1)
            
            if return_attn_weights:
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(xq, xk.transpose(-1, -2)) * scale
                if is_causal:
                    causal_mask = torch.triu(
                        torch.full((seq_len, seq_len), float("-inf"), device=xq.device, dtype=scores.dtype),
                        diagonal=1,
                    )
                    scores = scores + causal_mask
                if mask is not None:
                    scores = scores + mask
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
                out = torch.matmul(attn_weights, xv)
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
                return self.wo(out), attn_weights

            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=is_causal)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
            return self.wo(out)

        t_q = torch.arange(seq_len, device=xq.device).view(1, 1, seq_len, 1)
        t_k = torch.arange(seq_len, device=xq.device).view(1, 1, 1, seq_len)
        b_q = t_q // self.block_size
        b_k = t_k // self.block_size

        sink_mask = (b_k == 0)

        if self.mode == "forecast" and forecast_vector is not None:
            # 1. Compress keys into block representations
            xk_for_comp = xk.transpose(1, 2)  # [B, T, n_kv_head, head_dim]
            block_reps = self.compressor(xk_for_comp)

            # 2. Select Top-K blocks using forecast vector
            _, topk = self.selector(forecast_vector, block_reps, top_k=self.top_k_blocks, is_causal=is_causal)
            # topk: [batch, n_kv_head, seq_len, top_k]

            # 3. Construct block-sparse causal mask
            local_mask = (b_k == b_q)
            topk_mask = torch.zeros((batch_size, self.n_kv_head, seq_len, num_blocks), dtype=torch.bool, device=xq.device)
            topk_mask.scatter_(dim=-1, index=topk.clamp(0, num_blocks - 1), value=True)
            topk_selected = topk_mask.gather(dim=-1, index=b_k.expand(batch_size, self.n_kv_head, seq_len, seq_len))
            allowed = local_mask | sink_mask | topk_selected
        else:
            # Static sliding-window + attention sink mask
            sliding_window_mask = (b_k <= b_q) & (b_q - b_k < self.top_k_blocks + 1)
            allowed = sliding_window_mask | sink_mask
        if is_causal:
            allowed = allowed & (t_k <= t_q)

        attn_mask = torch.where(
            allowed,
            torch.zeros(1, dtype=xq.dtype, device=xq.device),
            torch.full((1,), float("-inf"), dtype=xq.dtype, device=xq.device),
        )

        if mask is not None:
            attn_mask = attn_mask + mask

        if self.num_queries_per_kv > 1:
            xk = torch.repeat_interleave(xk, repeats=self.num_queries_per_kv, dim=1)
            xv = torch.repeat_interleave(xv, repeats=self.num_queries_per_kv, dim=1)
            attn_mask = torch.repeat_interleave(attn_mask, repeats=self.num_queries_per_kv, dim=1)

        if return_attn_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(xq, xk.transpose(-1, -2)) * scale + attn_mask
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
            out = torch.matmul(attn_weights, xv)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
            return self.wo(out), attn_weights

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
        return self.wo(out)

    def _forward_chunked_gather(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_len: int,
        forecast_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes full sequence prefill by physical block-by-block gathering.
        
        Used for profiling actual wall-clock gather operations and memory reduction.
        """
        batch_size = xq.shape[0]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []

        xk_for_comp = xk.transpose(1, 2)
        block_reps = self.compressor(xk_for_comp)

        for b_q in range(num_blocks):
            start_q = b_q * self.block_size
            end_q = min(seq_len, (b_q + 1) * self.block_size)
            q_len = end_q - start_q

            xq_slice = xq[:, :, start_q:end_q, :]
            
            # Local block + sink block + (top-k if forecast else sliding window)
            max_sel = min(num_blocks, self.top_k_blocks + 2)
            selected_mask = torch.zeros((batch_size, self.n_kv_head, num_blocks), dtype=torch.bool, device=xq.device)
            sink_idx = torch.zeros((batch_size, self.n_kv_head, 1), dtype=torch.long, device=xq.device)
            curr_idx = torch.full((batch_size, self.n_kv_head, 1), b_q, dtype=torch.long, device=xq.device)

            selected_mask.scatter_(dim=-1, index=sink_idx, value=True)
            selected_mask.scatter_(dim=-1, index=curr_idx, value=True)

            if self.mode == "forecast" and forecast_vector is not None and b_q > 0:
                fc_slice = forecast_vector[:, start_q:end_q, :, :]
                _, topk = self.selector(fc_slice, block_reps, top_k=self.top_k_blocks, is_causal=True)
                topk_indices = topk.view(batch_size, self.n_kv_head, -1)
                selected_mask.scatter_(dim=-1, index=topk_indices.clamp(0, num_blocks - 1), value=True)
            else:
                local_start = max(0, b_q - self.top_k_blocks + 1)
                for b in range(local_start, b_q + 1):
                    idx = torch.full((batch_size, self.n_kv_head, 1), b, dtype=torch.long, device=xq.device)
                    selected_mask.scatter_(dim=-1, index=idx, value=True)

            future_mask = torch.arange(num_blocks, device=xq.device) > b_q
            selected_mask[:, :, future_mask] = False

            val, sel_indices = torch.topk(selected_mask.float(), k=max_sel, dim=-1)
            k_gat, v_gat = gather_kv_blocks(xk, xv, sel_indices, self.block_size)

            offsets = torch.arange(self.block_size, device=xq.device).view(1, 1, 1, self.block_size)
            token_pos = (sel_indices.unsqueeze(-1) * self.block_size + offsets).view(
                batch_size, self.n_kv_head, 1, max_sel * self.block_size
            )
            block_valid = (val > 0.5).unsqueeze(-1).expand(-1, -1, -1, self.block_size).reshape(
                batch_size, self.n_kv_head, 1, max_sel * self.block_size
            )

            # Causal mask relative to current slice
            q_pos = (torch.arange(start_q, end_q, device=xq.device)).view(1, 1, q_len, 1)
            is_valid = (token_pos < seq_len) & (token_pos <= q_pos) & block_valid

            attn_mask = torch.where(
                is_valid,
                torch.zeros(1, dtype=xq.dtype, device=xq.device),
                torch.full((1,), float("-inf"), dtype=xq.dtype, device=xq.device),
            )

            if self.num_queries_per_kv > 1:
                k_gat = torch.repeat_interleave(k_gat, repeats=self.num_queries_per_kv, dim=1)
                v_gat = torch.repeat_interleave(v_gat, repeats=self.num_queries_per_kv, dim=1)
                attn_mask = torch.repeat_interleave(attn_mask, repeats=self.num_queries_per_kv, dim=1)

            out_slice = F.scaled_dot_product_attention(xq_slice, k_gat, v_gat, attn_mask=attn_mask, is_causal=False)
            outputs.append(out_slice)

        full_out = torch.cat(outputs, dim=2)
        full_out = full_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
        return self.wo(full_out)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        forecast_vector: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Executes Sparse GQA Attention.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            rotary_emb: RotaryEmbedding module
            kv_cache: Optional persistent KV cache for autoregressive decoding
            mask: Optional explicit attention mask
            is_causal: Whether attention is causal
            forecast_vector: Optional forecast tensor [batch, seq_len, forecast_heads, forecast_dim] from layer L-1
            **kwargs: Extra flags (e.g. return_attn_weights, use_chunked_gather)
        """
        batch_size, seq_len, _ = x.shape

        # Projections: [B, T, H, D]
        xq = self.wq(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        cache_offset = kv_cache.current_len if kv_cache is not None else 0
        xq, xk = rotary_emb(xq, xk, offset=cache_offset)

        # Autoregressive decoding path (T=1 with kv_cache)
        if kv_cache is not None and seq_len == 1:
            xk, xv = kv_cache.update(xk, xv)
            seq_len_k = kv_cache.current_len
            xq = xq.transpose(1, 2)  # [B, n_head, 1, head_dim]
            xk = xk.transpose(1, 2)  # [B, n_kv_head, seq_len_k, head_dim]
            xv = xv.transpose(1, 2)  # [B, n_kv_head, seq_len_k, head_dim]
            return self._forward_decode_gather(
                xq=xq,
                xk=xk,
                xv=xv,
                seq_len_k=seq_len_k,
                forecast_vector=forecast_vector,
            )

        # Sequence prefill / training path
        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        use_chunked_gather = kwargs.get("use_chunked_gather", False)
        if use_chunked_gather:
            return self._forward_chunked_gather(
                xq=xq,
                xk=xk,
                xv=xv,
                seq_len=seq_len,
                forecast_vector=forecast_vector,
            )

        return self._forward_masked_sdpa(
            xq=xq,
            xk=xk,
            xv=xv,
            seq_len=seq_len,
            forecast_vector=forecast_vector,
            mask=mask,
            is_causal=is_causal,
            return_attn_weights=kwargs.get("return_attn_weights", False),
        )
