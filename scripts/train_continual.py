"""CLI and Experiment Runner for Continual Learning & Catastrophic Forgetting.

Implements Section 14 of the SparDA specification:
- Runs multi-stage continual learning:
  - Stage 1: General Python & Documentation
  - Stage 2: Systems Programming & CUDA Kernels
  - Stage 3: Mathematical Reasoning & Algorithms
- Compares:
  1. Naive sequential fine-tuning (catastrophic forgetting baseline)
  2. Experience Replay (5%–20% rehearsal ratio)
  3. Elastic Weight Consolidation (EWC) parameter regularizer
  4. Selective EWC protecting Forecast projection heads
- Evaluates Backward Transfer (BWT), Average Forgetting, and Forecast Retention.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
import yaml
import torch

from model.config import ModelConfig
from model.model import SparDAResearchModel
from data.tokenizer import SparDATokenizer
from data.continual import ContinualDatasetManager
from training.continual_trainer import ContinualTrainer
from evaluation.continual import generate_continual_report


def run_continual_experiment(
    config_path: str = "configs/base.yaml",
    strategy: str = "all",
    attention_backend: str = "dense",
    steps_per_stage: int = 50,
    replay_ratio: float = 0.15,
    ewc_lambda: float = 400.0,
    batch_size: int = 4,
    device_str: str = "cuda",
    output_dir: str = "docs/continual",
) -> Dict[str, Any]:
    """Runs single or comparative continual learning experiment suite."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Executing Continual Learning Experiment on device: {device} (Backend: {attention_backend})")

    # 1. Load Model Config
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    model_kwargs = raw_cfg["model"]
    model_kwargs["attention_backend"] = attention_backend

    # Ensure forecast heads are configured for forecast evaluation
    if "forecast_heads" not in model_kwargs or model_kwargs["forecast_heads"] == 0:
        model_kwargs["forecast_heads"] = 4
        model_kwargs["forecast_dim"] = 64

    model_config = ModelConfig(**model_kwargs)

    # 2. Prepare Data Stages
    tok_path = raw_cfg.get("paths", {}).get("tokenizer_path", "data/tokenizer.json")
    tokenizer = SparDATokenizer.from_file(tok_path) if os.path.exists(tok_path) else None
    data_mgr = ContinualDatasetManager(tokenizer=tokenizer, seq_len=128)
    stages = data_mgr.get_all_stages()

    print(f"Configured {len(stages)} continual learning stages:")
    for s in stages:
        print(f"  - Stage {s.stage_idx}: {s.name} ({s.description}) - {len(s.train_dataset)} train samples")

    # 3. Determine Strategies to Run
    if strategy == "all":
        strategies_to_test = [
            ("naive", "naive", 0.0, 0.0, "all"),
            ("replay_15pct", "replay", replay_ratio, 0.0, "all"),
            ("ewc_backbone", "ewc", 0.0, ewc_lambda, "backbone"),
            ("ewc_forecast", "ewc", 0.0, ewc_lambda, "forecast"),
            ("replay_and_ewc", "replay_and_ewc", replay_ratio, ewc_lambda, "all"),
        ]
    elif strategy == "naive":
        strategies_to_test = [("naive", "naive", 0.0, 0.0, "all")]
    elif strategy == "replay":
        strategies_to_test = [(f"replay_{int(replay_ratio*100)}pct", "replay", replay_ratio, 0.0, "all")]
    elif strategy == "ewc":
        strategies_to_test = [("ewc_all", "ewc", 0.0, ewc_lambda, "all")]
    elif strategy == "ewc_selective_forecast":
        strategies_to_test = [("ewc_forecast", "ewc", 0.0, ewc_lambda, "forecast")]
    else:
        strategies_to_test = [(strategy, strategy, replay_ratio, ewc_lambda, "all")]

    results: Dict[str, Any] = {}
    trajectories: Dict[str, Any] = {}

    for strat_key, strat_mode, r_ratio, e_lambda, e_target in strategies_to_test:
        print(f"\n=======================================================")
        print(f" Running Continual Strategy: {strat_key.upper()}")
        print(f" Mode: {strat_mode}, Replay: {r_ratio*100:.1f}%, EWC lambda: {e_lambda}, Target: {e_target}")
        print(f"=======================================================")

        # Initialize fresh model copy for each strategy
        torch.manual_seed(42)
        model = SparDAResearchModel(model_config)

        trainer = ContinualTrainer(
            model=model,
            stages=stages,
            device=device,
            strategy=strat_mode,
            replay_ratio=r_ratio,
            replay_capacity=500,
            ewc_lambda=e_lambda,
            ewc_target=e_target,
            learning_rate=raw_cfg.get("training", {}).get("learning_rate", 3e-4),
            steps_per_stage=steps_per_stage,
            batch_size=batch_size,
            grad_accum_steps=2,
            mixed_precision="bf16" if device.type == "cuda" else "fp32",
        )

        strat_res = trainer.train_all_stages()
        results[strat_key] = strat_res
        trajectories[strat_key] = strat_res["trajectory"]

    # 4. Generate Comparative Report
    stage_names = {s.stage_idx: f"{s.name}" for s in stages}
    report_md = generate_continual_report(trajectories, stage_names=stage_names)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "continual_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSaved Continual Learning Report to {report_path}")

    # Serialize results
    json_path = os.path.join(output_dir, "continual_results.json")
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {
            "metrics_loss": v["metrics_loss"],
            "metrics_ppl": v["metrics_ppl"],
            "trajectory": v["trajectory"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved Continual Learning JSON metrics to {json_path}")

    return {
        "results": serializable_results,
        "report_md": report_md,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SparDA Continual Learning & Catastrophic Forgetting Benchmark")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to model config")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "naive", "replay", "ewc", "ewc_selective_forecast", "replay_and_ewc"],
        help="Continual learning regularization strategy",
    )
    parser.add_argument("--attention_backend", type=str, default="dense", choices=["dense", "sparse", "forecast", "offload", "async_offload"])
    parser.add_argument("--steps_per_stage", type=int, default=30, help="Training steps per stage")
    parser.add_argument("--replay_ratio", type=float, default=0.15, help="Rehearsal ratio (5-20%)")
    parser.add_argument("--ewc_lambda", type=float, default=400.0, help="EWC quadratic loss penalty multiplier")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="docs/continual", help="Output directory for reports")
    args = parser.parse_args()

    run_continual_experiment(
        config_path=args.config,
        strategy=args.strategy,
        attention_backend=args.attention_backend,
        steps_per_stage=args.steps_per_stage,
        replay_ratio=args.replay_ratio,
        ewc_lambda=args.ewc_lambda,
        batch_size=args.batch_size,
        device_str=args.device,
        output_dir=args.output_dir,
    )
