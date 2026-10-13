"""Forecast-Guided KV Cache Offloading Attention Backend (CPU <-> GPU).

Implements Section 11 of the SparDA specification:
- Authoritative full KV cache is stored in Pinned Host CPU RAM (pin_memory).
- GPU holds only active/selected KV blocks for current computation.
- Tracks PCIe transfer volume, transfer latency, selection latency, and attention latency.
"""

import math
import time
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention, KVCache
from model.attention.sparse import gather_kv_blocks
from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.forecast import KVBlockCompressor, ForecastBlockSelector


class OffloadedKVCache:
    """Authoritative KV Cache maintained in Pinned Host CPU Memory.
    
    Streams only selected blocks to GPU on demand, drastically reducing GPU VRAM.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        gpu_device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        pin_memory: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.gpu_device = gpu_device
        self.dtype = dtype

        # Allocate pinned CPU host memory for fast DMA transfers
        self.cpu_k = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_head, head_dim),
            dtype=dtype,
            device="cpu",
        )
        self.cpu_v = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_head, head_dim),
            dtype=dtype,
            device="cpu",
        )
        if pin_memory and torch.cuda.is_available() and self.gpu_device.type == "cuda":
            self.cpu_k = self.cpu_k.pin_memory()
            self.cpu_v = self.cpu_v.pin_memory()

        self.current_len = 0

        # Transfer accounting
        self.total_bytes_transferred = 0
        self.total_transfer_calls = 0

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> int:
        """Appends new token keys and values from GPU to the CPU pinned cache."""
        seq_len = xk.shape[1]
        start_idx = self.current_len
        end_idx = start_idx + seq_len

        # Copy from GPU to CPU host memory
        self.cpu_k[:, start_idx:end_idx] = xk.detach().to("cpu", non_blocking=True)
        self.cpu_v[:, start_idx:end_idx] = xv.detach().to("cpu", non_blocking=True)
        self.current_len = end_idx
        return self.current_len

    def get_full_cpu_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the slice of the CPU cache up to the current length."""
        return self.cpu_k[:, :self.current_len], self.cpu_v[:, :self.current_len]

    def transfer_selected_blocks(
        self,
        block_indices: torch.Tensor,
        block_size: int,
        non_blocking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Gathers selected blocks from CPU memory and transfers them to the GPU.
        
        Args:
            block_indices: [batch, n_kv_head, num_selected_blocks]
            block_size: Number of tokens per block
            non_blocking: Whether to use asynchronous PCIe transfer
            
        Returns:
            gpu_k: [batch, n_kv_head, num_selected_blocks * block_size, head_dim] on GPU
            gpu_v: [batch, n_kv_head, num_selected_blocks * block_size, head_dim] on GPU
            bytes_transferred: Volume in bytes transferred across PCIe
        """
        # 1. Gather on CPU
        # k, v inputs to gather_kv_blocks: [B, H, T, D]
        k_slice = self.cpu_k[:, :self.current_len].transpose(1, 2)
        v_slice = self.cpu_v[:, :self.current_len].transpose(1, 2)
        
        # Ensure indices are CPU-side for CPU gather
        cpu_indices = block_indices.to("cpu")
        cpu_k_gathered, cpu_v_gathered = gather_kv_blocks(
            k=k_slice,
            v=v_slice,
            block_indices=cpu_indices,
            block_size=block_size,
        )

        # 2. Transfer gathered blocks from CPU pinned memory to GPU
        gpu_k = cpu_k_gathered.to(self.gpu_device, non_blocking=non_blocking)
        gpu_v = cpu_v_gathered.to(self.gpu_device, non_blocking=non_blocking)

        bytes_transferred = (gpu_k.numel() + gpu_v.numel()) * gpu_k.element_size()
        self.total_bytes_transferred += bytes_transferred
        self.total_transfer_calls += 1

        return gpu_k, gpu_v, bytes_transferred

    def reset(self):
        self.current_len = 0
        self.total_bytes_transferred = 0
        self.total_transfer_calls = 0


class ForecastOffloadAttention(BaseAttention):
    """Attention backend with Forecast-guided CPU -> GPU KV cache block offloading."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_size = config.block_size
        self.top_k_blocks = config.top_k_blocks

        self.compressor = KVBlockCompressor(
            block_size=self.block_size,
            head_dim=self.head_dim,
            forecast_dim=config.forecast_dim,
        )
        self.selector = ForecastBlockSelector(
            block_size=self.block_size,
            forecast_dim=config.forecast_dim,
        )

        # Profiling stats
        self.last_profile: Dict[str, float] = {
            "selection_ms": 0.0,
            "transfer_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "bytes_transferred": 0,
        }

    def _forward_offload_decode(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        offload_cache: OffloadedKVCache,
        forecast_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive decode step offloading KV to CPU and streaming Top-K blocks to GPU."""
        device = xq.device
        is_cuda = device.type == "cuda"
        batch_size = xq.shape[0]

        t_start = time.perf_counter()

        # 1. Update authoritative CPU pinned cache with new token
        curr_len = offload_cache.update(xk, xv)
        num_blocks = (curr_len + self.block_size - 1) // self.block_size
        curr_block = (curr_len - 1) // self.block_size
        max_sel_blocks = min(num_blocks, self.top_k_blocks + 2)

        # Small context fallback (if <= max_sel_blocks, transfer full slice)
        if num_blocks <= max_sel_blocks or forecast_vector is None or self.layer_idx == 0:
            if is_cuda:
                torch.cuda.synchronize(device)
            t_sel_end = time.perf_counter()

            # Transfer accumulated slice from CPU to GPU
            cpu_k_full, cpu_v_full = offload_cache.get_full_cpu_kv()
            gpu_k = cpu_k_full.to(device, non_blocking=True).transpose(1, 2)
            gpu_v = cpu_v_full.to(device, non_blocking=True).transpose(1, 2)
            bytes_transferred = (gpu_k.numel() + gpu_v.numel()) * gpu_k.element_size()
            
            if is_cuda:
                torch.cuda.synchronize(device)
            t_trans_end = time.perf_counter()

            if self.num_queries_per_kv > 1:
                gpu_k = torch.repeat_interleave(gpu_k, repeats=self.num_queries_per_kv, dim=1)
                gpu_v = torch.repeat_interleave(gpu_v, repeats=self.num_queries_per_kv, dim=1)

            out = F.scaled_dot_product_attention(xq, gpu_k, gpu_v, is_causal=False)
            out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
            proj_out = self.wo(out)

            if is_cuda:
                torch.cuda.synchronize(device)
            t_attn_end = time.perf_counter()

            self.last_profile = {
                "selection_ms": (t_sel_end - t_start) * 1000.0,
                "transfer_ms": (t_trans_end - t_sel_end) * 1000.0,
                "attention_ms": (t_attn_end - t_trans_end) * 1000.0,
                "total_ms": (t_attn_end - t_start) * 1000.0,
                "bytes_transferred": bytes_transferred,
            }
            return proj_out

        # 2. Block Selection via Forecast Vector
        # We need block representations. Slicing CPU keys for compressor:
        cpu_k_full, _ = offload_cache.get_full_cpu_kv()
        # Transfer compressed keys (or compress on GPU)
        # Slicing keys to GPU for prototype compression
        # Note: In Phase 6, prototypes are updated incrementally.
        gpu_k_for_comp = cpu_k_full.to(device, non_blocking=True)
        block_reps = self.compressor(gpu_k_for_comp)

        _, topk = self.selector(forecast_vector, block_reps, top_k=self.top_k_blocks, is_causal=True)
        topk_indices = topk[:, :, 0, :]  # [B, n_kv_head, top_k]

        # Combine Sink (0) + Local (curr_block) + Top-K
        selected_mask = torch.zeros((batch_size, self.n_kv_head, num_blocks), dtype=torch.bool, device=device)
        sink_idx = torch.zeros((batch_size, self.n_kv_head, 1), dtype=torch.long, device=device)
        curr_idx = torch.full((batch_size, self.n_kv_head, 1), curr_block, dtype=torch.long, device=device)

        selected_mask.scatter_(dim=-1, index=sink_idx, value=True)
        selected_mask.scatter_(dim=-1, index=curr_idx, value=True)
        selected_mask.scatter_(dim=-1, index=topk_indices.clamp(0, num_blocks - 1), value=True)

        future_mask = torch.arange(num_blocks, device=device) > curr_block
        selected_mask[:, :, future_mask] = False

        val, sel_indices = torch.topk(selected_mask.float(), k=max_sel_blocks, dim=-1)

        if is_cuda:
            torch.cuda.synchronize(device)
        t_sel_end = time.perf_counter()

        # 3. Selective Block Transfer from CPU Pinned Memory to GPU
        gpu_k_gat, gpu_v_gat, bytes_transferred = offload_cache.transfer_selected_blocks(
            block_indices=sel_indices,
            block_size=self.block_size,
            non_blocking=True,
        )

        if is_cuda:
            torch.cuda.synchronize(device)
        t_trans_end = time.perf_counter()

        # 4. Construct validity mask on GPU
        offsets = torch.arange(self.block_size, device=device).view(1, 1, 1, self.block_size)
        token_pos = (sel_indices.unsqueeze(-1) * self.block_size + offsets).view(
            batch_size, self.n_kv_head, 1, max_sel_blocks * self.block_size
        )
        block_valid = (val > 0.5).unsqueeze(-1).expand(-1, -1, -1, self.block_size).reshape(
            batch_size, self.n_kv_head, 1, max_sel_blocks * self.block_size
        )
        is_valid = (token_pos < curr_len) & block_valid

        attn_mask = torch.where(
            is_valid,
            torch.zeros(1, dtype=xq.dtype, device=device),
            torch.full((1,), float("-inf"), dtype=xq.dtype, device=device),
        )

        if self.num_queries_per_kv > 1:
            gpu_k_gat = torch.repeat_interleave(gpu_k_gat, repeats=self.num_queries_per_kv, dim=1)
            gpu_v_gat = torch.repeat_interleave(gpu_v_gat, repeats=self.num_queries_per_kv, dim=1)
            attn_mask = torch.repeat_interleave(attn_mask, repeats=self.num_queries_per_kv, dim=1)

        out = F.scaled_dot_product_attention(xq, gpu_k_gat, gpu_v_gat, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
        proj_out = self.wo(out)

        if is_cuda:
            torch.cuda.synchronize(device)
        t_attn_end = time.perf_counter()

        self.last_profile = {
            "selection_ms": (t_sel_end - t_start) * 1000.0,
            "transfer_ms": (t_trans_end - t_sel_end) * 1000.0,
            "attention_ms": (t_attn_end - t_trans_end) * 1000.0,
            "total_ms": (t_attn_end - t_start) * 1000.0,
            "bytes_transferred": bytes_transferred,
        }

        return proj_out

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[Union[KVCache, OffloadedKVCache]] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        forecast_vector: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Executes Forecast-guided Offload Attention."""
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        cache_offset = kv_cache.current_len if kv_cache is not None else 0
        xq, xk = rotary_emb(xq, xk, offset=cache_offset)

        # Offloaded decode path
        if isinstance(kv_cache, OffloadedKVCache) and seq_len == 1:
            xq = xq.transpose(1, 2)  # [B, n_head, 1, head_dim]
            return self._forward_offload_decode(
                xq=xq,
                xk=xk,
                xv=xv,
                offload_cache=kv_cache,
                forecast_vector=forecast_vector,
            )

        # Standard KVCache or prefill path: update cache if present
        if kv_cache is not None:
            if isinstance(kv_cache, OffloadedKVCache):
                kv_cache.update(xk, xv)
                cpu_k, cpu_v = kv_cache.get_full_cpu_kv()
                xk = cpu_k.to(x.device, non_blocking=True)
                xv = cpu_v.to(x.device, non_blocking=True)
            else:
                xk, xv = kv_cache.update(xk, xv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.num_queries_per_kv > 1:
            xk = torch.repeat_interleave(xk, repeats=self.num_queries_per_kv, dim=1)
            xv = torch.repeat_interleave(xv, repeats=self.num_queries_per_kv, dim=1)

        causal_flag = is_causal if (seq_len > 1 and mask is None) else False
        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=causal_flag)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.head_dim)
        return self.wo(out)
