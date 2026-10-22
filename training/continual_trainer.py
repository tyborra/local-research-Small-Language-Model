"""Continual Learning Trainer supporting Staged Training, Replay, and EWC.

Implements Section 14 of the SparDA specification:
- Multi-stage sequential training loop.
- Dynamic rehearsal mixing: (1 - α) * D_new + α * D_replay.
- Elastic Weight Consolidation (EWC) loss regularization.
- Cross-stage evaluation and catastrophic forgetting tracking.
- Forecast projection accuracy and recall preservation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import SparDAResearchModel
from data.continual import ContinualStage, ContinualDatasetManager
from training.replay import ReplayBuffer, ContinualMixedDataLoader
from training.ewc import ElasticWeightConsolidation
from training.trainer import get_cosine_lr
from evaluation.continual import ContinualEvaluator, compute_continual_metrics


class ContinualTrainer:
    """Manages multi-stage continual learning, replay memory, EWC regularization, and cross-stage evaluation."""

    def __init__(
        self,
        model: SparDAResearchModel,
        stages: List[ContinualStage],
        device: torch.device,
        strategy: str = "replay",
        replay_ratio: float = 0.15,
        replay_capacity: int = 1000,
        ewc_lambda: float = 400.0,
        ewc_target: str = "all",
        learning_rate: float = 3.0e-4,
        min_lr: float = 3.0e-5,
        weight_decay: float = 0.05,
        warmup_steps: int = 20,
        steps_per_stage: int = 100,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        mixed_precision: str = "bf16",
    ):
        """
        Args:
            model: SparDAResearchModel instance
            stages: Ordered list of ContinualStage objects
            device: Compute device (cuda or cpu)
            strategy: 'naive', 'replay', 'ewc', or 'replay_and_ewc'
            replay_ratio: Fraction of replay samples in each batch (e.g. 0.15)
            replay_capacity: Maximum samples in ReplayBuffer
            ewc_lambda: EWC loss multiplier
            ewc_target: 'all', 'backbone', or 'forecast'
            learning_rate: Peak learning rate per stage
            min_lr: Minimum learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Warmup steps per stage
            steps_per_stage: Number of optimization steps per stage
            batch_size: Batch size
            grad_accum_steps: Gradient accumulation steps
            mixed_precision: 'bf16' or 'fp32'
        """
        self.model = model.to(device)
        self.stages = stages
        self.device = device
        self.strategy = strategy.lower()
        self.replay_ratio = replay_ratio
        self.steps_per_stage = steps_per_stage
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Precision
        self.dtype = (
            torch.bfloat16
            if mixed_precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        # Setup Replay Buffer
        use_replay = "replay" in self.strategy
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, stratified=True) if use_replay else None

        # Setup EWC Regularizer
        use_ewc = "ewc" in self.strategy
        self.ewc = (
            ElasticWeightConsolidation(
                model=self.model,
                importance_lambda=ewc_lambda,
                protect_target=ewc_target,
                device=self.device,
            )
            if use_ewc
            else None
        )

        # Build Stage Validation Loaders
        self.val_loaders: Dict[int, DataLoader] = {
            stage.stage_idx: DataLoader(stage.val_dataset, batch_size=batch_size, shuffle=False)
            for stage in self.stages
        }

        # Evaluator
        self.evaluator = ContinualEvaluator(
            model=self.model,
            stage_loaders=self.val_loaders,
            device=self.device,
            dtype=self.dtype,
            max_batches=15,
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.AdamW:
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

    def train_stage(self, stage: ContinualStage) -> Dict[str, Any]:
        """Trains the model on a single stage with active replay and/or EWC."""
        print(f"\n--- [Stage {stage.stage_idx}: {stage.description}] Strategy: {self.strategy} ---")
        self.model.train()

        base_train_loader = DataLoader(
            stage.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Create mixed data loader if replay is active
        if self.replay_buffer is not None and not self.replay_buffer.is_empty():
            train_loader = ContinualMixedDataLoader(
                current_loader=base_train_loader,
                replay_buffer=self.replay_buffer,
                replay_ratio=self.replay_ratio,
                device=self.device,
            )
        else:
            train_loader = base_train_loader

        train_iter = iter(train_loader)
        stage_loss = 0.0
        start_time = time.perf_counter()

        for step in range(self.steps_per_stage):
            lr = get_cosine_lr(step, self.warmup_steps, self.steps_per_stage, self.learning_rate, self.min_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_ewc = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                    outputs = self.model(input_ids, targets=labels)
                    ce_loss = outputs["loss"]
                    total_loss = ce_loss

                    # Apply EWC regularization penalty
                    if self.ewc is not None and self.ewc.recorded_stages:
                        ewc_pen = self.ewc.penalty(self.model)
                        total_loss = total_loss + ewc_pen
                        accum_ewc += ewc_pen.item() / self.grad_accum_steps

                    loss_step = total_loss / self.grad_accum_steps

                loss_step.backward()
                accum_loss += ce_loss.item() / self.grad_accum_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            stage_loss += accum_loss

            if (step + 1) % max(1, self.steps_per_stage // 4) == 0 or step == self.steps_per_stage - 1:
                ewc_str = f" | EWC Pen: {accum_ewc:.4f}" if accum_ewc > 0 else ""
                print(f"  Step {step + 1:3d}/{self.steps_per_stage:3d} | CE Loss: {accum_loss:.4f}{ewc_str} | LR: {lr:.2e}")

        elapsed = time.perf_counter() - start_time

        # Post-stage actions:
        # 1. Update Replay Buffer with samples from this stage
        if self.replay_buffer is not None:
            added = self.replay_buffer.add_from_dataset(
                dataset=stage.train_dataset,
                stage_id=stage.stage_idx,
                max_samples=self.replay_buffer.capacity // len(self.stages),
            )
            print(f"  [Replay] Added {added} samples from Stage {stage.stage_idx} to Replay Buffer (Total: {len(self.replay_buffer)})")

        # 2. Update EWC Fisher matrix if EWC is active
        if self.ewc is not None:
            print(f"  [EWC] Computing Fisher Information Matrix for Stage {stage.stage_idx}...")
            self.ewc.register_stage(
                dataloader=DataLoader(stage.train_dataset, batch_size=self.batch_size, shuffle=True),
                stage_id=stage.stage_idx,
                max_batches=20,
                dtype=self.dtype,
            )

        # 3. Cross-stage evaluation across all stages
        print(f"  [Evaluation] Evaluating cross-stage performance...")
        cross_eval = self.evaluator.evaluate_stage(completed_stage=stage.stage_idx)

        for s_idx, metrics in cross_eval.items():
            fc_str = f" | FC Recall@2: {metrics.get('forecast_recall@2', 0.0):.3f}" if "forecast_recall@2" in metrics else ""
            print(f"    -> Stage {s_idx} ({self.stages[s_idx - 1].name}): PPL={metrics['perplexity']:.2f}, Loss={metrics['eval_loss']:.3f}{fc_str}")

        return {
            "stage_idx": stage.stage_idx,
            "avg_loss": stage_loss / self.steps_per_stage,
            "elapsed_sec": elapsed,
            "cross_eval": cross_eval,
        }

    def train_all_stages(self) -> Dict[str, Any]:
        """Runs continual learning across all configured stages."""
        stage_summaries = []
        for stage in self.stages:
            summary = self.train_stage(stage)
            stage_summaries.append(summary)

        # Compute continual metrics
        bwt_loss = self.evaluator.get_metrics("eval_loss")
        bwt_ppl = self.evaluator.get_metrics("perplexity")

        print(f"\n========================================================")
        print(f" Continual Learning Summary: Strategy '{self.strategy}'")
        print(f"========================================================")
        print(f" Backward Transfer (Loss):     {bwt_loss['backward_transfer']:+.4f}")
        print(f" Average Forgetting (Loss):   {bwt_loss['average_forgetting']:.4f}")
        print(f" Backward Transfer (PPL):      {bwt_ppl['backward_transfer']:+.2f}")
        print(f" Max Stage Forgetting (PPL):  {bwt_ppl['max_forgetting']:.2f}")

        return {
            "strategy": self.strategy,
            "stage_summaries": stage_summaries,
            "trajectory": self.evaluator.trajectory,
            "metrics_loss": bwt_loss,
            "metrics_ppl": bwt_ppl,
            "report": self.evaluator.generate_report(strategy_name=self.strategy),
        }
