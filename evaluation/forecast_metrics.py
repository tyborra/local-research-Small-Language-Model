"""Evaluation metrics for Forecast Block Prediction (Recall@K, Precision@K, KL Divergence)."""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def compute_recall_at_k(
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
    k: int,
) -> float:
    """Computes Recall@K between predicted top-K block indices and teacher target top-K blocks.
    
    Recall@K = |Pred_K ∩ Target_K| / |Target_K|
    
    Args:
        pred_indices: [..., num_pred] tensor of predicted block indices
        target_indices: [..., num_target] tensor of ground-truth block indices
        k: Threshold K to evaluate
        
    Returns:
        float: Mean Recall@K across all tokens and heads
    """
    k_pred = min(k, pred_indices.size(-1))
    k_target = min(k, target_indices.size(-1))
    if k_target == 0 or k_pred == 0:
        return 0.0

    p_sub = pred_indices[..., :k_pred]      # [..., k_pred]
    t_sub = target_indices[..., :k_target]  # [..., k_target]

    # Expand for pairwise equality match: [..., k_pred, 1] == [..., 1, k_target]
    p_expanded = p_sub.unsqueeze(-1)
    t_expanded = t_sub.unsqueeze(-2)
    matches = (p_expanded == t_expanded).any(dim=-1).float()  # [..., k_pred]

    # Sum matching elements and divide by target set size
    intersection = matches.sum(dim=-1)  # [...]
    recall = (intersection / float(k_target)).mean().item()
    return float(recall)


def compute_precision_at_k(
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
    k: int,
) -> float:
    """Computes Precision@K between predicted top-K block indices and teacher target top-K blocks.
    
    Precision@K = |Pred_K ∩ Target_K| / |Pred_K|
    """
    k_pred = min(k, pred_indices.size(-1))
    k_target = min(k, target_indices.size(-1))
    if k_target == 0 or k_pred == 0:
        return 0.0

    p_sub = pred_indices[..., :k_pred]
    t_sub = target_indices[..., :k_target]

    p_expanded = p_sub.unsqueeze(-1)
    t_expanded = t_sub.unsqueeze(-2)
    matches = (p_expanded == t_expanded).any(dim=-1).float()

    intersection = matches.sum(dim=-1)
    precision = (intersection / float(k_pred)).mean().item()
    return float(precision)


def compute_block_selection_accuracy(
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> float:
    """Computes top-1 exact match accuracy: whether the #1 predicted block is the #1 teacher block."""
    top1_pred = pred_indices[..., 0]
    top1_target = target_indices[..., 0]
    acc = (top1_pred == top1_target).float().mean().item()
    return float(acc)


def compute_kl_divergence(
    pred_scores: torch.Tensor,
    target_distribution: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Computes KL-Divergence D_KL(Target || Pred) over valid (finite) block score distributions."""
    valid_mask = torch.isfinite(pred_scores) & (target_distribution.sum(dim=-1, keepdim=True) > 0)
    
    # Target distribution
    target_p = target_distribution.clamp(min=eps)
    target_p = target_p / target_p.sum(dim=-1, keepdim=True).clamp(min=eps)
    
    # Predicted distribution
    pred_p = F.softmax(pred_scores, dim=-1).clamp(min=eps)
    pred_p = pred_p / pred_p.sum(dim=-1, keepdim=True).clamp(min=eps)
    
    # KL = sum( target_p * log(target_p / pred_p) )
    kl = (target_p * (torch.log(target_p) - torch.log(pred_p))).sum(dim=-1)
    
    # Average over valid positions
    pos_mask = valid_mask.any(dim=-1)
    if pos_mask.any():
        return float(kl[pos_mask].mean().item())
    return float(kl.mean().item())


def evaluate_forecast_metrics(
    pred_scores: torch.Tensor,
    pred_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_distribution: Optional[torch.Tensor] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Computes full suite of forecast prediction metrics.
    
    Returns dict containing:
        - "accuracy_top1": Exact match for top-1 block
        - "recall@K": For each K in k_values
        - "precision@K": For each K in k_values
        - "kl_divergence": (If target_distribution is provided)
    """
    if k_values is None:
        k_values = [1, 2, 4, 8]

    metrics: Dict[str, float] = {
        "accuracy_top1": compute_block_selection_accuracy(pred_indices, target_indices),
    }

    for k in k_values:
        metrics[f"recall@{k}"] = compute_recall_at_k(pred_indices, target_indices, k=k)
        metrics[f"precision@{k}"] = compute_precision_at_k(pred_indices, target_indices, k=k)

    if target_distribution is not None:
        metrics["kl_divergence"] = compute_kl_divergence(pred_scores, target_distribution)

    return metrics
