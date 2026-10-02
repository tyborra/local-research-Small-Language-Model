"""Atomic checkpoint saving and loading utilities."""

import os
import tempfile
from typing import Dict, Any, Optional
import torch

from model.config import ModelConfig
from model.model import SparDAResearchModel


def save_checkpoint(
    path: str,
    model: SparDAResearchModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: ModelConfig,
    extra: Optional[Dict[str, Any]] = None,
):
    """Saves checkpoint atomically to prevent partial write corruption."""
    abs_path = os.path.normpath(os.path.abspath(path))
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tmp_path = f"{abs_path}.tmp.{os.getpid()}"

    checkpoint_data = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "extra": extra or {},
    }

    try:
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, abs_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_checkpoint(
    path: str,
    model: SparDAResearchModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads checkpoint into model and optional optimizer."""
    map_location = device if device is not None else torch.device("cpu")
    checkpoint_data = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint_data["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    return {
        "step": checkpoint_data.get("step", 0),
        "loss": checkpoint_data.get("loss", float("inf")),
        "config": checkpoint_data.get("config"),
        "extra": checkpoint_data.get("extra", {}),
    }
