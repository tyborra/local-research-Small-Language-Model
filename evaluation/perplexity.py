"""Validation perplexity and loss evaluation harness for SparDA research model."""

import math
import time
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import TokenizedDataset, create_dataloader


def compute_perplexity(
    model: nn.Module,
    val_data: Union[DataLoader, np.ndarray, torch.Tensor],
    batch_size: int = 4,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes exact cross-entropy loss, perplexity, and bits-per-token over validation data.
    
    Args:
        model: SparDAResearchModel or equivalent PyTorch module
        val_data: DataLoader or raw 1D token array/tensor
        batch_size: Batch size to use if val_data is a raw token array
        seq_len: Sequence length to use if val_data is a raw token array
        device: Device to run evaluation on (defaults to next(model.parameters()).device)
        dtype: Data type for AMP autocast (defaults to torch.bfloat16 on CUDA if supported)
        max_batches: Optional cap on the number of evaluation batches
        
    Returns:
        Dict with keys:
            - eval_loss: float (cross-entropy in nats)
            - perplexity: float (exp(eval_loss))
            - bits_per_token: float (loss / ln(2))
            - total_tokens: int
            - total_batches: int
            - eval_time_sec: float
            - tokens_per_sec: float
    """
    if device is None:
        device = next(model.parameters()).device

    if dtype is None:
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Prepare DataLoader if raw tokens are provided
    if isinstance(val_data, (np.ndarray, torch.Tensor)):
        if seq_len is None:
            if hasattr(model, "config") and hasattr(model.config, "context_length"):
                seq_len = model.config.context_length
            else:
                seq_len = 1024
        dataset = TokenizedDataset(val_data, seq_len=seq_len)
        data_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    else:
        data_loader = val_data

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_tokens = input_ids.numel()

            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids, targets=labels)
                loss = outputs["loss"]

            loss_val = loss.item()
            total_loss += loss_val * batch_tokens
            total_tokens += batch_tokens
            total_batches += 1

    eval_time = max(time.perf_counter() - start_time, 1e-6)

    if total_tokens == 0:
        return {
            "eval_loss": float("nan"),
            "perplexity": float("nan"),
            "bits_per_token": float("nan"),
            "total_tokens": 0,
            "total_batches": 0,
            "eval_time_sec": eval_time,
            "tokens_per_sec": 0.0,
        }

    avg_loss = total_loss / total_tokens
    # Safe exponentiation to avoid overflow
    ppl = math.exp(min(avg_loss, 100.0))
    bits_per_token = avg_loss / math.log(2.0)
    tok_per_sec = total_tokens / eval_time

    return {
        "eval_loss": float(avg_loss),
        "perplexity": float(ppl),
        "bits_per_token": float(bits_per_token),
        "total_tokens": int(total_tokens),
        "total_batches": int(total_batches),
        "eval_time_sec": float(eval_time),
        "tokens_per_sec": float(tok_per_sec),
    }
