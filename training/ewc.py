"""Elastic Weight Consolidation (EWC) Parameter Regularization.

Implements Section 14 of the SparDA specification:
- Computes empirical Fisher Information Matrix (diagonal approximation) for completed stages.
- Penalizes parameter shifts proportional to their Fisher importance:
  L_EWC(θ) = (λ / 2) * Σ F_i * (θ_i - θ*_i)^2
- Supports selective protection:
  - 'all': Protects all model parameters.
  - 'backbone': Protects attention and MLP weights, leaving forecast heads flexible.
  - 'forecast': Protects forecast projection weights specifically to preserve KV selection accuracy.
"""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:
    """Manages Fisher Information Matrix calculation and quadratic parameter penalties across stages."""

    def __init__(
        self,
        model: nn.Module,
        importance_lambda: float = 400.0,
        protect_target: str = "all",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: SparDAResearchModel or equivalent module.
            importance_lambda: Strength multiplier for the quadratic penalty.
            protect_target: 'all', 'backbone', or 'forecast'.
            device: Device to hold optimal parameter copies and Fisher matrices.
        """
        self.model = model
        self.importance_lambda = float(importance_lambda)
        self.protect_target = protect_target.lower()
        self.device = device if device is not None else next(model.parameters()).device

        # Stored optimal weights and Fisher information per parameter name
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.recorded_stages: List[int] = []

    def _should_protect(self, name: str) -> bool:
        """Determines if a named parameter should be regularized under the chosen target."""
        if self.protect_target == "all":
            return True
        elif self.protect_target == "forecast":
            return "forecast" in name
        elif self.protect_target == "backbone":
            return "forecast" not in name
        return True

    def register_stage(
        self,
        dataloader: DataLoader,
        stage_id: int,
        max_batches: int = 50,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, float]:
        """Computes empirical Fisher Information on the current stage and records optimal weights.
        
        Args:
            dataloader: DataLoader with stage data
            stage_id: Integer identifier of the completed stage
            max_batches: Maximum number of batches used to estimate Fisher
            dtype: Autocast dtype (bf16/fp32)
        
        Returns:
            Dict mapping parameter name to its mean Fisher importance value
        """
        self.model.eval()

        # Save optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_protect(name):
                self.optimal_params[name] = param.detach().clone().to(self.device)

        # Initialize or accumulate Fisher matrix
        new_fisher: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_protect(name):
                new_fisher[name] = torch.zeros_like(param, device=self.device)

        batch_count = 0
        autocast_dtype = dtype if dtype is not None else torch.float32

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            self.model.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
                outputs = self.model(input_ids, targets=labels)
                loss = outputs["loss"]

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and self._should_protect(name) and param.grad is not None:
                    new_fisher[name] += param.grad.detach() ** 2

            batch_count += 1

        # Normalize across batches
        if batch_count > 0:
            for name in new_fisher:
                new_fisher[name] /= float(batch_count)

        # Accumulate or set Fisher matrix
        if not self.fisher_matrix:
            self.fisher_matrix = new_fisher
        else:
            for name in new_fisher:
                if name in self.fisher_matrix:
                    self.fisher_matrix[name] = 0.5 * self.fisher_matrix[name] + 0.5 * new_fisher[name]
                else:
                    self.fisher_matrix[name] = new_fisher[name]

        self.recorded_stages.append(stage_id)
        self.model.train()

        return {name: float(f.mean().item()) for name, f in self.fisher_matrix.items()}

    def penalty(self, current_model: Optional[nn.Module] = None) -> torch.Tensor:
        """Computes quadratic EWC penalty across protected parameters:
        
        L_penalty = (λ / 2) * Σ F_i * (θ_i - θ*_i)^2
        """
        if not self.fisher_matrix or not self.optimal_params or self.importance_lambda <= 0.0:
            return torch.tensor(0.0, device=self.device)

        model = current_model if current_model is not None else self.model
        total_penalty = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if name in self.fisher_matrix and name in self.optimal_params:
                diff = param - self.optimal_params[name]
                fisher = self.fisher_matrix[name]
                total_penalty = total_penalty + (fisher * (diff ** 2)).sum()

        return (self.importance_lambda / 2.0) * total_penalty


# Convenient alias
EWCRegularizer = ElasticWeightConsolidation
