"""Continual Learning & Catastrophic Forgetting Evaluation Suite.

Implements Section 14 of the SparDA specification:
- Cross-stage evaluation matrix R_{i,j} (Performance on stage j after training on stage i).
- Perplexity and Cross-Entropy loss tracking across stages.
- Backward Transfer (BWT) and Average Forgetting metrics.
- Forward Transfer (FWT) metrics.
- Forecast prediction retention (Recall@K, Precision@K) across training stages.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.perplexity import compute_perplexity
from evaluation.forecast_metrics import evaluate_forecast_metrics, compute_recall_at_k, compute_precision_at_k


def evaluate_cross_stage_performance(
    model: nn.Module,
    stage_loaders: Dict[int, DataLoader],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_batches: int = 20,
    evaluate_forecasts: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Evaluates model performance across all stage validation loaders.
    
    Args:
        model: SparDAResearchModel or equivalent
        stage_loaders: Mapping of stage_idx -> DataLoader
        device: Device to run evaluation
        dtype: Autocast dtype (bf16/fp32)
        max_batches: Number of batches per stage to evaluate
        evaluate_forecasts: If True, evaluates forecast Recall@K if model supports it
        
    Returns:
        Dict mapping stage_idx -> Dict containing perplexity, loss, and forecast metrics
    """
    model.eval()
    dev = device if device is not None else next(model.parameters()).device
    autocast_dtype = dtype if dtype is not None else (torch.bfloat16 if dev.type == "cuda" else torch.float32)

    results: Dict[int, Dict[str, Any]] = {}

    for stage_idx, loader in sorted(stage_loaders.items()):
        # 1. Compute validation loss and perplexity
        ppl_metrics = compute_perplexity(
            model=model,
            val_data=loader,
            device=dev,
            dtype=autocast_dtype,
            max_batches=max_batches,
        )

        stage_res: Dict[str, Any] = {
            "eval_loss": ppl_metrics["eval_loss"],
            "perplexity": ppl_metrics["perplexity"],
            "bits_per_token": ppl_metrics["bits_per_token"],
        }

        # 2. Evaluate Forecast retention if supported and requested
        has_forecast = hasattr(model, "forecast_projections") and model.config.forecast_heads > 0
        if evaluate_forecasts and has_forecast:
            top1_accs = []
            recalls_k = {2: [], 4: []}

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= max_batches:
                        break
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    labels = batch["labels"].to(dev, non_blocking=True)

                    with torch.amp.autocast(dev.type, dtype=autocast_dtype, enabled=dev.type == "cuda"):
                        out = model(input_ids, targets=labels, return_forecasts=True)

                    # Check if teacher selection and forecast predictions are present
                    if "forecast_predictions" in out and out["forecast_predictions"]:
                        for layer_idx, fc_out in out["forecast_predictions"].items():
                            if isinstance(fc_out, dict) and "pred_indices" in fc_out and "target_indices" in fc_out:
                                p_idx = fc_out["pred_indices"]
                                t_idx = fc_out["target_indices"]
                                if p_idx is not None and t_idx is not None:
                                    recalls_k[2].append(compute_recall_at_k(p_idx, t_idx, k=2))
                                    recalls_k[4].append(compute_recall_at_k(p_idx, t_idx, k=4))

            stage_res["forecast_recall@2"] = float(np.mean(recalls_k[2])) if recalls_k[2] else 0.0
            stage_res["forecast_recall@4"] = float(np.mean(recalls_k[4])) if recalls_k[4] else 0.0

        results[stage_idx] = stage_res

    model.train()
    return results


def compute_continual_metrics(
    trajectory: List[Dict[int, Dict[str, Any]]],
    metric_key: str = "eval_loss",
) -> Dict[str, float]:
    """Calculates Backward Transfer (BWT) and Average Forgetting from evaluation trajectory.
    
    trajectory: List of cross-stage evaluations recorded after completing each stage.
      trajectory[i][j] is the performance on stage j after training on stage i.
      Indices are 0-based for training steps, 1-based for stage IDs.
    
    Returns:
        Dict containing:
            - backward_transfer: Negative value indicates catastrophic forgetting.
            - average_forgetting: Mean increase in loss on previous stages.
            - max_forgetting: Maximum observed loss increase on any historical stage.
    """
    num_stages = len(trajectory)
    if num_stages <= 1:
        return {
            "backward_transfer": 0.0,
            "average_forgetting": 0.0,
            "max_forgetting": 0.0,
        }

    # BWT = (1 / (N - 1)) * sum_{j=1}^{N-1} (R_{N, j} - R_{j, j})
    # Note: For loss, R_{N, j} > R_{j, j} means loss increased (forgotten).
    # We define BWT such that negative values imply forgetting: (R_{j,j} - R_{N,j}) for loss.
    final_step = trajectory[-1]
    bwt_sum = 0.0
    forgetting_list = []

    for stage_j in range(1, num_stages):
        r_jj = trajectory[stage_j - 1][stage_j][metric_key]
        r_nj = final_step[stage_j][metric_key]

        if metric_key in ("eval_loss", "perplexity"):
            # Higher is worse; forgetting is positive increase in loss/ppl
            loss_delta = r_nj - r_jj
            bwt_sum -= loss_delta  # Negative BWT when loss increases
            
            # Forgetting = max_{l in {j...N-1}} R_{l, j} - R_{N, j} (or loss degradation from best)
            best_prev = min(trajectory[l - 1][stage_j][metric_key] for l in range(stage_j, num_stages))
            f_j = max(0.0, r_nj - best_prev)
            forgetting_list.append(f_j)
        else:
            # Higher is better (e.g. accuracy or recall)
            delta = r_nj - r_jj
            bwt_sum += delta
            best_prev = max(trajectory[l - 1][stage_j][metric_key] for l in range(stage_j, num_stages))
            f_j = max(0.0, best_prev - r_nj)
            forgetting_list.append(f_j)

    denom = float(num_stages - 1)
    bwt = bwt_sum / denom
    avg_forgetting = float(np.mean(forgetting_list)) if forgetting_list else 0.0
    max_forgetting = float(np.max(forgetting_list)) if forgetting_list else 0.0

    return {
        "backward_transfer": float(bwt),
        "average_forgetting": float(avg_forgetting),
        "max_forgetting": float(max_forgetting),
    }


def generate_continual_report(
    trajectories: Dict[str, List[Dict[int, Dict[str, Any]]]],
    stage_names: Optional[Dict[int, str]] = None,
) -> str:
    """Generates an aesthetic Markdown comparison report across continual learning methods.
    
    Args:
        trajectories: Dict of strategy_name -> list of stage evaluation dicts.
        stage_names: Optional mapping of stage_idx -> human readable name.
    """
    if stage_names is None:
        stage_names = {
            1: "Stage 1 (Python Docs)",
            2: "Stage 2 (Systems/CUDA)",
            3: "Stage 3 (Math/Algorithms)",
        }

    lines = []
    lines.append("# Continual Learning & Catastrophic Forgetting Analysis")
    lines.append("")
    lines.append("## Cross-Stage Perplexity & Retention Matrix")
    lines.append("")

    for strat_name, traj in trajectories.items():
        lines.append(f"### Strategy: **{strat_name}**")
        lines.append("")
        
        # Table Header
        headers = ["Stage Completed"] + [stage_names.get(s, f"Stage {s}") for s in sorted(stage_names.keys())]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for step_idx, step_eval in enumerate(traj):
            completed_stage = step_idx + 1
            row = [f"After Stage {completed_stage}"]
            for s in sorted(stage_names.keys()):
                if s in step_eval:
                    ppl = step_eval[s]["perplexity"]
                    loss = step_eval[s]["eval_loss"]
                    row.append(f"PPL: {ppl:.2f} (Loss: {loss:.3f})")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        # Summary Metrics
        loss_metrics = compute_continual_metrics(traj, metric_key="eval_loss")
        ppl_metrics = compute_continual_metrics(traj, metric_key="perplexity")

        lines.append("")
        lines.append(f"- **Max Stage Forgetting (PPL Delta)**: `{ppl_metrics['max_forgetting']:.2f}`")
        lines.append("")

    return "\n".join(lines)


def compute_backward_transfer(
    trajectory: List[Dict[int, Dict[str, Any]]],
    metric_key: str = "eval_loss",
) -> float:
    """Computes Backward Transfer across stages."""
    return compute_continual_metrics(trajectory, metric_key=metric_key)["backward_transfer"]


def compute_forgetting(
    trajectory: List[Dict[int, Dict[str, Any]]],
    metric_key: str = "eval_loss",
) -> float:
    """Computes Average Forgetting across stages."""
    return compute_continual_metrics(trajectory, metric_key=metric_key)["average_forgetting"]


class ContinualEvaluator:
    """Orchestrates cross-stage evaluation and tracks metrics across training stages."""

    def __init__(
        self,
        model: nn.Module,
        stage_loaders: Dict[int, DataLoader],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ):
        self.model = model
        self.stage_loaders = stage_loaders
        self.device = device
        self.dtype = dtype
        self.max_batches = max_batches
        self.trajectory: List[Dict[int, Dict[str, Any]]] = []

    def evaluate_stage(self, completed_stage: int) -> Dict[int, Dict[str, Any]]:
        """Evaluates model across all stages and appends to trajectory."""
        stage_results = evaluate_cross_stage_performance(
            model=self.model,
            stage_loaders=self.stage_loaders,
            device=self.device,
            dtype=self.dtype,
            max_batches=self.max_batches,
        )
        self.trajectory.append(stage_results)
        return stage_results

    def get_metrics(self, metric_key: str = "eval_loss") -> Dict[str, float]:
        """Calculates BWT and forgetting on the current trajectory."""
        return compute_continual_metrics(self.trajectory, metric_key=metric_key)

    def generate_report(self, strategy_name: str = "Current Run") -> str:
        """Generates markdown report."""
        return generate_continual_report({strategy_name: self.trajectory})

