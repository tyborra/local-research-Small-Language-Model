"""Asynchronous Forecast-Guided KV Cache Prefetching Attention Backend (CPU <-> GPU).

Implements Section 12 of the SparDA specification:
- Overlaps CPU -> GPU KV PCIe DMA transfers on a dedicated CUDA Stream
  with GPU compute (Layer L Attention and SwiGLU FFN).
- Layer L produces Forecast vector F_L for Layer L+1.
- Asynchronous transfer of selected Layer L+1 KV blocks is queued immediately on
  the background transfer stream.
- When Layer L+1 runs, the compute stream synchronizes with the transfer event via
  CUDA event (torch.cuda.Event), hiding transfer latency behind Layer L computation.
"""

import math
import time
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.base import BaseAttention, KVCache
from model.attention.offload import OffloadedKVCache
from model.attention.sparse import gather_kv_blocks
from model.config import ModelConfig
from model.rope import RotaryEmbedding
from model.forecast import KVBlockCompressor, ForecastBlockSelector


class AsyncOffloadedKVCache(OffloadedKVCache):
    """Authoritative KV Cache in Pinned Host CPU RAM with dedicated Async CUDA Transfer Stream."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        gpu_device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        pin_memory: bool = True,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            gpu_device=gpu_device,
            dtype=dtype,
            pin_memory=pin_memory,
        )
        self.transfer_stream = transfer_stream
        if self.transfer_stream is None and torch.cuda.is_available() and self.gpu_device.type == "cuda":
            self.transfer_stream = torch.cuda.Stream(device=self.gpu_device)

    def async_transfer_selected_blocks(
        self,
        block_indices: torch.Tensor,
        block_size: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event], int]:
        """Asynchronously gathers selected blocks on CPU and queues DMA transfer to GPU.

        Args:
            block_indices: [batch, n_kv_head, num_selected_blocks]
            block_size: Number of tokens per block
            stream: Target CUDA stream for DMA transfer (defaults to self.transfer_stream)

        Returns:
            gpu_k: GPU tensor receiving the transferred keys
            gpu_v: GPU tensor receiving the transferred values
            event: CUDA event recorded immediately after queuing the transfer
            bytes_transferred: Volume in bytes transferred
        """
        target_stream = stream if stream is not None else self.transfer_stream
        is_cuda = self.gpu_device.type == "cuda" and torch.cuda.is_available()

        # 1. Gather on CPU from pinned authoritative store
        cpu_indices = block_indices.to("cpu")
        max_idx = int(cpu_indices.max().item()) if cpu_indices.numel() > 0 else 0
        req_len = min(self.max_seq_len, max(self.current_len, (max_idx + 1) * block_size))
        k_slice = self.cpu_k[:, :req_len].transpose(1, 2)
        v_slice = self.cpu_v[:, :req_len].transpose(1, 2)

        cpu_k_gathered, cpu_v_gathered = gather_kv_blocks(
            k=k_slice,
            v=v_slice,
            block_indices=cpu_indices,
            block_size=block_size,
        )

        # Ensure gathered tensors are pinned for true asynchronous DMA
        if is_cuda and not cpu_k_gathered.is_pinned():
            cpu_k_gathered = cpu_k_gathered.pin_memory()
            cpu_v_gathered = cpu_v_gathered.pin_memory()

        bytes_transferred = (cpu_k_gathered.numel() + cpu_v_gathered.numel()) * cpu_k_gathered.element_size()
        self.total_bytes_transferred += bytes_transferred
        self.total_transfer_calls += 1

        # 2. Queue asynchronous DMA transfer on dedicated transfer stream
        event: Optional[torch.cuda.Event] = None
        if is_cuda and target_stream is not None:
            with torch.cuda.stream(target_stream):
                gpu_k = cpu_k_gathered.to(self.gpu_device, non_blocking=True)
                gpu_v = cpu_v_gathered.to(self.gpu_device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(target_stream)
        else:
            gpu_k = cpu_k_gathered.to(self.gpu_device)
            gpu_v = cpu_v_gathered.to(self.gpu_device)

        return gpu_k, gpu_v, event, bytes_transferred


class AsyncForecastOffloadAttention(BaseAttention):
    """Attention backend featuring asynchronous forecast-driven KV block prefetching."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        transfer_stream: Optional[torch.cuda.Stream] = None,
    ):
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

        # Dedicated CUDA transfer stream
        self.transfer_stream = transfer_stream

        # Prefetched state container
        self.prefetch_event: Optional[torch.cuda.Event] = None
        self.prefetched_gpu_k: Optional[torch.Tensor] = None
        self.prefetched_gpu_v: Optional[torch.Tensor] = None
        self.prefetched_sel_indices: Optional[torch.Tensor] = None
        self.prefetched_val: Optional[torch.Tensor] = None
        self.prefetched_max_sel_blocks: int = 0
        self.prefetched_bytes: int = 0
        self.prefetched_curr_len_at_launch: int = 0
        self.prefetched_is_small_context: bool = False
        self.prefetch_launch_time: float = 0.0

        # Fine-grained latency & overlap metrics
        self.last_profile: Dict[str, float] = {
            "selection_ms": 0.0,
            "transfer_ms": 0.0,
            "transfer_launch_ms": 0.0,
            "wait_ms": 0.0,
            "attention_ms": 0.0,
            "total_ms": 0.0,
            "bytes_transferred": 0,
            "latency_hidden_ms": 0.0,
            "overlap_pct": 0.0,
        }

    def _ensure_transfer_stream(self, device: torch.device):
        if self.transfer_stream is None and device.type == "cuda" and torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream(device=device)

    def prefetch(
        self,
        forecast_vector: torch.Tensor,
        offload_cache: Union[AsyncOffloadedKVCache, OffloadedKVCache],
    ) -> bool:
        """Asynchronously preselects and issues DMA transfer for this layer's KV blocks.

        Called at Layer L to prefetch Layer L+1 blocks while Layer L executes compute.

        Args:
            forecast_vector: Forecast vector emitted by layer L [batch, 1, forecast_heads, forecast_dim]
            offload_cache: Authoritative CPU KV cache for this layer (Layer L+1)

        Returns:
            True if prefetch was successfully queued, False otherwise
        """
        device = forecast_vector.device
        self._ensure_transfer_stream(device)
        is_cuda = device.type == "cuda" and torch.cuda.is_available()

        curr_len = offload_cache.current_len
        if curr_len == 0:
            return False

        t0 = time.perf_counter()
        batch_size = forecast_vector.shape[0]
        # Lookahead 1 token to anticipate upcoming decode step
        target_len = curr_len + 1
        num_blocks = (target_len + self.block_size - 1) // self.block_size
        curr_block = curr_len // self.block_size
        max_sel_blocks = min(num_blocks, self.top_k_blocks + 2)

        # Small context fallback (if <= max_sel_blocks or layer 0)
        is_small_context = (num_blocks <= max_sel_blocks or forecast_vector is None or self.layer_idx == 0)

        if is_small_context:
            # Transfer contiguous slice up to target_len on transfer stream
            cpu_k_full, cpu_v_full = offload_cache.cpu_k[:, :target_len], offload_cache.cpu_v[:, :target_len]
            if is_cuda and not cpu_k_full.is_pinned():
                cpu_k_full = cpu_k_full.pin_memory()
                cpu_v_full = cpu_v_full.pin_memory()

            bytes_trans = (cpu_k_full.numel() + cpu_v_full.numel()) * cpu_k_full.element_size()
            offload_cache.total_bytes_transferred += bytes_trans
            offload_cache.total_transfer_calls += 1

            event: Optional[torch.cuda.Event] = None
            if is_cuda and self.transfer_stream is not None:
                with torch.cuda.stream(self.transfer_stream):
                    gpu_k = cpu_k_full.to(device, non_blocking=True).transpose(1, 2)
                    gpu_v = cpu_v_full.to(device, non_blocking=True).transpose(1, 2)
                    event = torch.cuda.Event()
                    event.record(self.transfer_stream)
            else:
                gpu_k = cpu_k_full.to(device).transpose(1, 2)
                gpu_v = cpu_v_full.to(device).transpose(1, 2)

            t1 = time.perf_counter()
            self.prefetch_event = event
            self.prefetched_gpu_k = gpu_k
            self.prefetched_gpu_v = gpu_v
            self.prefetched_sel_indices = None
            self.prefetched_val = None
            self.prefetched_max_sel_blocks = max_sel_blocks
            self.prefetched_bytes = bytes_trans
            self.prefetched_curr_len_at_launch = curr_len
            self.prefetched_is_small_context = True
            self.prefetch_launch_time = t0
            self.last_profile["transfer_launch_ms"] = (t1 - t0) * 1000.0
            return True

        # 1. Block Selection via Forecast Vector
        cpu_k_full, _ = offload_cache.get_full_cpu_kv()
        gpu_k_for_comp = cpu_k_full.to(device, non_blocking=True)
        block_reps = self.compressor(gpu_k_for_comp)

        _, topk = self.selector(forecast_vector, block_reps, top_k=self.top_k_blocks, is_causal=True)
        topk_indices = topk[:, :, 0, :]  # [B, n_kv_head, top_k]

        selected_mask = torch.zeros((batch_size, self.n_kv_head, num_blocks), dtype=torch.bool, device=device)
        sink_idx = torch.zeros((batch_size, self.n_kv_head, 1), dtype=torch.long, device=device)
        curr_idx = torch.full((batch_size, self.n_kv_head, 1), curr_block, dtype=torch.long, device=device)

        selected_mask.scatter_(dim=-1, index=sink_idx, value=True)
        selected_mask.scatter_(dim=-1, index=curr_idx, value=True)
        selected_mask.scatter_(dim=-1, index=topk_indices.clamp(0, num_blocks - 1), value=True)

        future_mask = torch.arange(num_blocks, device=device) > curr_block
        selected_mask[:, :, future_mask] = False

        val, sel_indices = torch.topk(selected_mask.float(), k=max_sel_blocks, dim=-1)

        # 2. Issue DMA transfer on dedicated transfer stream
        if isinstance(offload_cache, AsyncOffloadedKVCache):
            gpu_k, gpu_v, event, bytes_trans = offload_cache.async_transfer_selected_blocks(
                block_indices=sel_indices,
                block_size=self.block_size,
                stream=self.transfer_stream,
            )
        else:
            gpu_k, gpu_v, bytes_trans = offload_cache.transfer_selected_blocks(
                block_indices=sel_indices,
                block_size=self.block_size,
                non_blocking=True,
            )
            event = torch.cuda.Event() if is_cuda else None
            if event is not None and self.transfer_stream is not None:
                event.record(self.transfer_stream)

        t1 = time.perf_counter()

        # 3. Store prefetched state
        self.prefetch_event = event
        self.prefetched_gpu_k = gpu_k
        self.prefetched_gpu_v = gpu_v
        self.prefetched_sel_indices = sel_indices
        self.prefetched_val = val
        self.prefetched_max_sel_blocks = max_sel_blocks
        self.prefetched_bytes = bytes_trans
        self.prefetched_curr_len_at_launch = curr_len
        self.prefetched_is_small_context = False
        self.prefetch_launch_time = t0
        self.last_profile["transfer_launch_ms"] = (t1 - t0) * 1000.0

        return True

    def _forward_async_decode(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        offload_cache: Union[AsyncOffloadedKVCache, OffloadedKVCache],
        forecast_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes decode step with asynchronous stream synchronization and latency hiding."""
        device = xq.device
        is_cuda = device.type == "cuda" and torch.cuda.is_available()
        batch_size = xq.shape[0]

        t_start = time.perf_counter()

        has_prefetch = (
            self.prefetch_event is not None
            and self.prefetched_gpu_k is not None
            and self.prefetched_gpu_v is not None
        )

        wait_ms = 0.0
        sel_ms = 0.0
        transfer_ms = 0.0

        if has_prefetch:
            # Synchronize compute stream with background transfer event
            t_wait_start = time.perf_counter()
            if is_cuda and self.prefetch_event is not None:
                torch.cuda.current_stream().wait_event(self.prefetch_event)
            t_wait_end = time.perf_counter()
            wait_ms = (t_wait_end - t_wait_start) * 1000.0

            gpu_k = self.prefetched_gpu_k
            gpu_v = self.prefetched_gpu_v
            sel_indices = self.prefetched_sel_indices
            val = self.prefetched_val
            max_sel_blocks = self.prefetched_max_sel_blocks
            bytes_transferred = self.prefetched_bytes
            prev_len = self.prefetched_curr_len_at_launch
            is_small = self.prefetched_is_small_context

            # Update authoritative CPU cache with new token
            curr_len = offload_cache.update(xk, xv)

            # Reset prefetch state
            self.prefetch_event = None
            self.prefetched_gpu_k = None
            self.prefetched_gpu_v = None

            if is_small:
                # Insert current token into the pre-allocated slice slot
                gpu_k[:, :, prev_len:prev_len + 1, :] = xk.transpose(1, 2)
                gpu_v[:, :, prev_len:prev_len + 1, :] = xv.transpose(1, 2)

                if self.num_queries_per_kv > 1:
                    gpu_k = torch.repeat_interleave(gpu_k, repeats=self.num_queries_per_kv, dim=1)
                    gpu_v = torch.repeat_interleave(gpu_v, repeats=self.num_queries_per_kv, dim=1)

                t_attn_start = time.perf_counter()
                out = F.scaled_dot_product_attention(xq, gpu_k, gpu_v, is_causal=False)
                out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
                proj_out = self.wo(out)

                if is_cuda:
                    torch.cuda.synchronize(device)
                t_attn_end = time.perf_counter()
                attn_ms = (t_attn_end - t_attn_start) * 1000.0
                total_ms = (t_attn_end - t_start) * 1000.0

                self.last_profile = {
                    "selection_ms": 0.0,
                    "transfer_ms": (bytes_transferred / (25 * 1024 * 1024 * 1024)) * 1000.0,
                    "transfer_launch_ms": self.last_profile.get("transfer_launch_ms", 0.0),
                    "wait_ms": wait_ms,
                    "attention_ms": attn_ms,
                    "total_ms": total_ms,
                    "bytes_transferred": bytes_transferred,
                    "latency_hidden_ms": max(0.0, ((bytes_transferred / (25 * 1024 * 1024 * 1024)) * 1000.0) - wait_ms),
                    "overlap_pct": 100.0 if wait_ms < 0.1 else 50.0,
                }
                return proj_out

            else:
                # Insert current token into the gathered block buffer
                offsets = torch.arange(self.block_size, device=device).view(1, 1, 1, self.block_size)
                token_pos = (sel_indices.unsqueeze(-1) * self.block_size + offsets).view(
                    batch_size, self.n_kv_head, 1, max_sel_blocks * self.block_size
                )
                curr_token_slot = (token_pos == prev_len).squeeze(2)
                if curr_token_slot.any():
                    xk_t = xk.transpose(1, 2)
                    xv_t = xv.transpose(1, 2)
                    gpu_k[curr_token_slot] = xk_t.reshape(-1, self.head_dim)
                    gpu_v[curr_token_slot] = xv_t.reshape(-1, self.head_dim)

        else:
            # Fallback on-demand synchronous path
            curr_len = offload_cache.update(xk, xv)
            num_blocks = (curr_len + self.block_size - 1) // self.block_size
            curr_block = (curr_len - 1) // self.block_size
            max_sel_blocks = min(num_blocks, self.top_k_blocks + 2)

            if num_blocks <= max_sel_blocks or forecast_vector is None or self.layer_idx == 0:
                t_sel_end = time.perf_counter()
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
                    "transfer_launch_ms": 0.0,
                    "wait_ms": 0.0,
                    "attention_ms": (t_attn_end - t_trans_end) * 1000.0,
                    "total_ms": (t_attn_end - t_start) * 1000.0,
                    "bytes_transferred": bytes_transferred,
                    "latency_hidden_ms": 0.0,
                    "overlap_pct": 0.0,
                }
                return proj_out

            # Selection on GPU
            cpu_k_full, _ = offload_cache.get_full_cpu_kv()
            gpu_k_for_comp = cpu_k_full.to(device, non_blocking=True)
            block_reps = self.compressor(gpu_k_for_comp)

            _, topk = self.selector(forecast_vector, block_reps, top_k=self.top_k_blocks, is_causal=True)
            topk_indices = topk[:, :, 0, :]

            selected_mask = torch.zeros((batch_size, self.n_kv_head, num_blocks), dtype=torch.bool, device=device)
            sink_idx = torch.zeros((batch_size, self.n_kv_head, 1), dtype=torch.long, device=device)
            curr_idx = torch.full((batch_size, self.n_kv_head, 1), curr_block, dtype=torch.long, device=device)

            selected_mask.scatter_(dim=-1, index=sink_idx, value=True)
            selected_mask.scatter_(dim=-1, index=curr_idx, value=True)
            selected_mask.scatter_(dim=-1, index=topk_indices.clamp(0, num_blocks - 1), value=True)

            future_mask = torch.arange(num_blocks, device=device) > curr_block
            selected_mask[:, :, future_mask] = False

            val, sel_indices = torch.topk(selected_mask.float(), k=max_sel_blocks, dim=-1)
            t_sel_end = time.perf_counter()
            sel_ms = (t_sel_end - t_start) * 1000.0

            gpu_k, gpu_v, bytes_transferred = offload_cache.transfer_selected_blocks(
                block_indices=sel_indices,
                block_size=self.block_size,
                non_blocking=True,
            )
            if is_cuda:
                torch.cuda.synchronize(device)
            t_trans_end = time.perf_counter()
            transfer_ms = (t_trans_end - t_sel_end) * 1000.0

        # Construct validity mask on GPU
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
            gpu_k = torch.repeat_interleave(gpu_k, repeats=self.num_queries_per_kv, dim=1)
            gpu_v = torch.repeat_interleave(gpu_v, repeats=self.num_queries_per_kv, dim=1)
            attn_mask = torch.repeat_interleave(attn_mask, repeats=self.num_queries_per_kv, dim=1)

        t_attn_start = time.perf_counter()
        out = F.scaled_dot_product_attention(xq, gpu_k, gpu_v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.n_head * self.head_dim)
        proj_out = self.wo(out)

        if is_cuda:
            torch.cuda.synchronize(device)
        t_attn_end = time.perf_counter()
        attn_ms = (t_attn_end - t_attn_start) * 1000.0
        total_ms = (t_attn_end - t_start) * 1000.0

        est_transfer_time_ms = (bytes_transferred / (25 * 1024 * 1024 * 1024)) * 1000.0 if bytes_transferred > 0 else 0.0
        latency_hidden_ms = max(0.0, est_transfer_time_ms - wait_ms) if has_prefetch else 0.0
        overlap_pct = (
            min(100.0, (latency_hidden_ms / max(est_transfer_time_ms, 1e-4)) * 100.0)
            if (has_prefetch and est_transfer_time_ms > 0)
            else 0.0
        )

        self.last_profile = {
            "selection_ms": sel_ms,
            "transfer_ms": transfer_ms if not has_prefetch else est_transfer_time_ms,
            "transfer_launch_ms": self.last_profile.get("transfer_launch_ms", 0.0),
            "wait_ms": wait_ms,
            "attention_ms": attn_ms,
            "total_ms": total_ms,
            "bytes_transferred": bytes_transferred,
            "latency_hidden_ms": latency_hidden_ms,
            "overlap_pct": overlap_pct,
        }

        return proj_out

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        kv_cache: Optional[Union[KVCache, OffloadedKVCache, AsyncOffloadedKVCache]] = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        forecast_vector: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward entrypoint for Asynchronous Forecast-Guided Offload Attention."""
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        cache_offset = kv_cache.current_len if kv_cache is not None else 0
        xq, xk = rotary_emb(xq, xk, offset=cache_offset)

        # Autoregressive decode step with async offloaded KV
        if isinstance(kv_cache, (AsyncOffloadedKVCache, OffloadedKVCache)) and seq_len == 1:
            xq = xq.transpose(1, 2)  # [B, n_head, 1, head_dim]
            return self._forward_async_decode(
                xq=xq,
                xk=xk,
                xv=xv,
                offload_cache=kv_cache,
                forecast_vector=forecast_vector,
            )

        # Prefill path: populate CPU/GPU cache and compute standard attention
        if kv_cache is not None:
            if isinstance(kv_cache, (AsyncOffloadedKVCache, OffloadedKVCache)):
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
