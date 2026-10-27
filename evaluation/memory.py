"""CUDA VRAM profiler for model weights, activations, and KV cache footprints."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from model.attention.base import KVCache
from model.config import ModelConfig


def calculate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    n_layer: Optional[int] = None,
    n_kv_head: Optional[int] = None,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    n_layers: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
) -> int:
    """Calculates theoretical analytical KV cache size in bytes.
    
    Formula: 2 (Key + Value) * batch_size * seq_len * n_kv_head * head_dim * bytes_per_elem * n_layer
    """
    n_l = n_layer if n_layer is not None else (n_layers if n_layers is not None else 12)
    n_kv_h = n_kv_head if n_kv_head is not None else (n_kv_heads if n_kv_heads is not None else 4)
    bytes_per_elem = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    return 2 * batch_size * seq_len * n_kv_h * head_dim * bytes_per_elem * n_l


class MemoryProfiler:
    """Profiles GPU VRAM allocation across weights, KV cache, activations, and peak usage."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.total_device_memory_mb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
            if self.is_cuda else 0.0
        )

    def _get_allocated_mb(self) -> float:
        if not self.is_cuda:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def _get_reserved_mb(self) -> float:
        if not self.is_cuda:
            return 0.0
        return torch.cuda.memory_reserved(self.device) / (1024 ** 2)

    def _get_max_allocated_mb(self) -> float:
        if not self.is_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def _reset_peak(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def profile_static_weights(self, model: nn.Module) -> Dict[str, float]:
        """Calculates theoretical and actual VRAM occupied by model weights."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_weight_bytes = param_bytes + buffer_bytes
        
        return {
            "theoretical_weights_mb": total_weight_bytes / (1024 ** 2),
            "allocated_weights_mb": self._get_allocated_mb(),
        }

    def profile_context_vram(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
        include_backward: bool = False,
    ) -> Dict[str, Any]:
        """Profiles VRAM consumption during prefill and decoding with KV cache at a given context length.
        
        Args:
            model: SparDAResearchModel instance
            batch_size: Batch size
            seq_len: Target sequence length (context length)
            dtype: Tensor precision
            include_backward: Whether to also profile training backward pass activations
            
        Returns:
            Dict containing detailed memory breakdown in megabytes and gigabytes.
        """
        if not self.is_cuda:
            return {
                "context_length": seq_len,
                "batch_size": batch_size,
                "error": "Memory profiling requires CUDA device",
            }

        torch.cuda.empty_cache()
        self._reset_peak()
        cfg = getattr(model, "config", None)
        assert cfg is not None, "Model must possess a ModelConfig attribute"

        # 1. Weights baseline
        weights_mb = self._get_allocated_mb()

        # 2. KV-Cache Allocation (instantiate KVCache for all layers)
        analytical_kv_bytes = calculate_kv_cache_bytes(
            batch_size=batch_size,
            seq_len=seq_len,
            n_layer=cfg.n_layer,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            dtype=dtype,
        )
        analytical_kv_mb = analytical_kv_bytes / (1024 ** 2)

        kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=seq_len,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                device=self.device,
                dtype=dtype,
            )
            for _ in range(cfg.n_layer)
        ]
        kv_cache_actual_mb = max(self._get_allocated_mb() - weights_mb, 0.0)

        # 3. Prefill Forward Pass (no KV cache during initial prompt processing)
        # Clear KV cache to measure pure prefill
        del kv_caches
        torch.cuda.empty_cache()
        self._reset_peak()
        prefill_base_mb = self._get_allocated_mb()

        input_ids = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=(batch_size, seq_len),
            device=self.device,
            dtype=torch.long,
        )

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                _ = model(input_ids)

        prefill_peak_mb = self._get_max_allocated_mb()
        prefill_activation_mb = max(prefill_peak_mb - prefill_base_mb, 0.0)

        # 4. Decode Step Profiling (with active KV cache)
        torch.cuda.empty_cache()
        self._reset_peak()
        
        # Instantiate fresh KV cache and perform 1 prefill token + 1 decode step
        kv_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=seq_len,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                device=self.device,
                dtype=dtype,
            )
            for _ in range(cfg.n_layer)
        ]
        
        decode_input = torch.randint(
            low=0,
            high=cfg.vocab_size,
            size=(batch_size, 1),
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                _ = model(decode_input, kv_caches=kv_caches)

        decode_peak_mb = self._get_max_allocated_mb()
        peak_total_mb = max(prefill_peak_mb, decode_peak_mb)
        reserved_mb = self._get_reserved_mb()
        headroom_mb = max(self.total_device_memory_mb - peak_total_mb, 0.0)

        # Cleanup
        del kv_caches, input_ids, decode_input
        torch.cuda.empty_cache()

        return {
            "context_length": seq_len,
            "batch_size": batch_size,
            "weights_mb": float(weights_mb),
            "weights_gb": float(weights_mb / 1024),
            "kv_cache_analytical_mb": float(analytical_kv_mb),
            "kv_cache_actual_mb": float(kv_cache_actual_mb),
            "prefill_activation_mb": float(prefill_activation_mb),
            "prefill_peak_mb": float(prefill_peak_mb),
            "decode_peak_mb": float(decode_peak_mb),
            "peak_vram_mb": float(peak_total_mb),
            "peak_vram_gb": float(peak_total_mb / 1024),
            "reserved_vram_mb": float(reserved_mb),
            "total_vram_mb": float(self.total_device_memory_mb),
            "headroom_vram_mb": float(headroom_mb),
            "headroom_vram_gb": float(headroom_mb / 1024),
        }


def profile_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Helper function to profile memory for a given model and context length."""
    if device is None:
        device = next(model.parameters()).device
    profiler = MemoryProfiler(device)
    return profiler.profile_context_vram(model, batch_size=batch_size, seq_len=seq_len, dtype=dtype)
