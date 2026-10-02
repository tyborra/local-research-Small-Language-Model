"""Trainer implementation with BF16, gradient accumulation, and telemetry."""

import os
import time
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.model import SparDAResearchModel
from training.checkpoint import save_checkpoint


def get_cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class Trainer:
    """Manages training lifecycle, optimization, telemetry, and checkpointing."""

    def __init__(
        self,
        model: SparDAResearchModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.learning_rate = float(train_cfg.get("learning_rate", 6e-4))
        self.min_lr = float(train_cfg.get("min_lr", 6e-5))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 100))
        self.max_steps = int(train_cfg.get("max_steps", 1000))
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.eval_interval = int(train_cfg.get("eval_interval", 100))
        self.eval_steps = int(train_cfg.get("eval_steps", 20))
        self.save_interval = int(train_cfg.get("save_interval", 200))
        self.checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")

        # Mixed precision setup
        self.mixed_precision = train_cfg.get("mixed_precision", "bf16")
        self.dtype = torch.bfloat16 if self.mixed_precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float32

        # Enable gradient checkpointing if requested
        if train_cfg.get("gradient_checkpointing", False):
            self.model.enable_gradient_checkpointing()

        # Optimizer: separate weight decay for 2D weight tensors vs 1D biases/norms
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
            {"params": decay_params, "weight_decay": float(train_cfg.get("weight_decay", 0.1))},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.learning_rate,
            betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        )

        self.step = 0
        self.tokens_processed = 0

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluates validation loss."""
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_steps:
                break
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
                outputs = self.model(input_ids, targets=labels)
                total_loss += outputs["loss"].item()
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.model.train()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        train_iter = iter(self.train_loader)
        running_loss = 0.0
        start_time = time.perf_counter()
        step_start_time = time.perf_counter()

        print(f"Starting training run on {self.device} (dtype: {self.dtype})")
        print(f"Total steps: {self.max_steps}, Grad Accum: {self.grad_accum_steps}")

        history = []

        while self.step < self.max_steps:
            # Update learning rate
            lr = get_cosine_lr(self.step, self.warmup_steps, self.max_steps, self.learning_rate, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
                    outputs = self.model(input_ids, targets=labels)
                    loss = outputs["loss"] / self.grad_accum_steps

                loss.backward()
                accum_loss += loss.item()
                self.tokens_processed += input_ids.numel()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.step += 1
            running_loss += accum_loss

            # Telemetry every 10 steps
            if self.step % 10 == 0 or self.step == self.max_steps:
                step_elapsed = time.perf_counter() - step_start_time
                tok_per_sec = (input_ids.numel() * self.grad_accum_steps * 10) / max(step_elapsed, 1e-6)
                step_start_time = time.perf_counter()

                peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3) if self.device.type == "cuda" else 0.0
                avg_train_loss = running_loss / (10 if self.step % 10 == 0 else (self.step % 10))
                running_loss = 0.0

                val_loss_str = ""
                val_loss = None
                if self.step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    val_loss_str = f" | val_loss: {val_loss:.4f}"

                print(
                    f"Step {self.step:5d}/{self.max_steps:5d} | "
                    f"loss: {avg_train_loss:.4f}{val_loss_str} | "
                    f"lr: {lr:.2e} | "
                    f"tok/s: {tok_per_sec:7.0f} | "
                    f"VRAM: {peak_vram_gb:.2f} GB"
                )

                history.append({
                    "step": self.step,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tok_per_sec": tok_per_sec,
                    "vram_gb": peak_vram_gb,
                })

            # Checkpoint save
            if self.step % self.save_interval == 0 or self.step == self.max_steps:
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_step_{self.step}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.step,
                    loss=accum_loss,
                    config=self.model.config,
                )
                print(f"Saved checkpoint to {ckpt_path}")

        total_time = time.perf_counter() - start_time
        print(f"Training completed in {total_time:.1f}s ({self.tokens_processed:,} total tokens processed)")
        return {"history": history, "total_time": total_time, "tokens_processed": self.tokens_processed}
