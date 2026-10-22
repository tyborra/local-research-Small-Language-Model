"""Dynamic Experience Replay Buffer for Continual Learning.

Implements Section 14 of the SparDA specification:
- Stratified and ring experience replay buffers.
- Maintains past stage data at configurable rehearsal ratios (5%–20%).
- Blends current stage data with past stage memories: (1 - α) * D_new + α * D_replay.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from training.ewc import ElasticWeightConsolidation, EWCRegularizer


class ReplayBuffer:
    """Experience replay buffer for storing token sequences from earlier training stages.
    
    Supports:
    - Stratified storage (partitioned by stage ID) so older stages aren't evicted by newer ones.
    - Uniform or reservoir sampling across past stages.
    - Fast batch tensor generation.
    """

    def __init__(
        self,
        capacity: int = 1000,
        stratified: bool = True,
    ):
        """
        Args:
            capacity: Maximum total number of sequence samples to retain.
            stratified: If True, balances capacity evenly across recorded stages.
        """
        self.capacity = capacity
        self.stratified = stratified
        self.stage_buffers: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        self.flat_buffer: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.stage_buffers.values()) if self.stratified else len(self.flat_buffer)

    def is_empty(self) -> bool:
        return len(self) == 0

    def add_sample(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        stage_id: int = 1,
    ) -> None:
        """Adds a single sample (input_ids, labels) to the replay buffer."""
        sample = {
            "input_ids": input_ids.detach().cpu().clone(),
            "labels": labels.detach().cpu().clone(),
            "stage_id": stage_id,
        }

        if self.stratified:
            if stage_id not in self.stage_buffers:
                self.stage_buffers[stage_id] = []
            buf = self.stage_buffers[stage_id]
            max_per_stage = max(1, self.capacity // max(len(self.stage_buffers), 1))

            if len(buf) < max_per_stage:
                buf.append(sample)
            else:
                # Random replacement (reservoir sampling within stage)
                idx = random.randint(0, len(buf) - 1)
                buf[idx] = sample

            # Re-balance existing stages if new stage reduced max_per_stage
            for s_id, s_buf in self.stage_buffers.items():
                if len(s_buf) > max_per_stage:
                    self.stage_buffers[s_id] = s_buf[:max_per_stage]
        else:
            if len(self.flat_buffer) < self.capacity:
                self.flat_buffer.append(sample)
            else:
                idx = random.randint(0, len(self.flat_buffer) - 1)
                self.flat_buffer[idx] = sample

    def add_from_dataset(
        self,
        dataset: Dataset,
        stage_id: int,
        max_samples: Optional[int] = None,
    ) -> int:
        """Populates the buffer from a Dataset.
        
        Args:
            dataset: PyTorch dataset yielding dict with 'input_ids' and 'labels'
            stage_id: Identifier for the current stage
            max_samples: Maximum number of samples to import (defaults to capacity / expected stages)
        
        Returns:
            Number of samples added.
        """
        num_to_add = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        added = 0
        for idx in indices[:num_to_add]:
            item = dataset[idx]
            self.add_sample(
                input_ids=item["input_ids"],
                labels=item["labels"],
                stage_id=stage_id,
            )
            added += 1

        return added

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples a batch of replay memories.
        
        Balances representation equally across all stored stages if stratified.
        
        Returns:
            Dict containing:
                - 'input_ids': [batch_size, seq_len]
                - 'labels': [batch_size, seq_len]
        """
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        selected: List[Dict[str, Any]] = []

        if self.stratified:
            active_stages = [s for s, b in self.stage_buffers.items() if len(b) > 0]
            per_stage = max(1, batch_size // len(active_stages))
            
            for stage in active_stages:
                buf = self.stage_buffers[stage]
                count = min(per_stage, len(buf))
                selected.extend(random.sample(buf, count))

            # If under batch_size due to integer division, sample remaining randomly
            while len(selected) < batch_size:
                stage = random.choice(active_stages)
                selected.append(random.choice(self.stage_buffers[stage]))
            
            # Truncate to exact batch_size
            selected = selected[:batch_size]
        else:
            k = min(batch_size, len(self.flat_buffer))
            selected = random.sample(self.flat_buffer, k)
            while len(selected) < batch_size:
                selected.append(random.choice(self.flat_buffer))

        input_ids = torch.stack([s["input_ids"] for s in selected], dim=0)
        labels = torch.stack([s["labels"] for s in selected], dim=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class ContinualMixedDataLoader:
    """Combines a current-stage DataLoader with a ReplayBuffer to produce rehearsal batches.
    
    For each step, a batch of size B contains:
        B_replay = round(B * replay_ratio) samples from ReplayBuffer
        B_current = B - B_replay samples from current stage DataLoader
    
    If ReplayBuffer is empty (e.g. Stage 1), yields 100% current stage data.
    """

    def __init__(
        self,
        current_loader: DataLoader,
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_ratio: float = 0.15,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            current_loader: PyTorch DataLoader for new stage data.
            replay_buffer: ReplayBuffer holding memories of past stages.
            replay_ratio: Rehearsal ratio alpha in [0.0, 1.0] (typical: 0.05 to 0.20).
            device: Target device for tensor transfer.
        """
        self.current_loader = current_loader
        self.replay_buffer = replay_buffer
        self.replay_ratio = max(0.0, min(1.0, float(replay_ratio)))
        self.device = device
        self.batch_size = current_loader.batch_size or 4

    def __len__(self) -> int:
        return len(self.current_loader)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for current_batch in self.current_loader:
            curr_input = current_batch["input_ids"]
            curr_labels = current_batch["labels"]
            batch_sz = curr_input.size(0)

            # Determine number of replay samples
            if (
                self.replay_buffer is None
                or self.replay_buffer.is_empty()
                or self.replay_ratio <= 0.0
            ):
                final_input = curr_input
                final_labels = curr_labels
            else:
                n_replay = int(round(batch_sz * self.replay_ratio))
                n_replay = max(1, min(n_replay, batch_sz - 1)) if batch_sz > 1 else 0

                if n_replay > 0:
                    replay_batch = self.replay_buffer.sample_batch(n_replay)
                    n_curr = batch_sz - n_replay

                    # Slice current batch
                    curr_sub_input = curr_input[:n_curr]
                    curr_sub_labels = curr_labels[:n_curr]

                    # Mix
                    final_input = torch.cat([curr_sub_input, replay_batch["input_ids"]], dim=0)
                    final_labels = torch.cat([curr_sub_labels, replay_batch["labels"]], dim=0)
                else:
                    final_input = curr_input
                    final_labels = curr_labels

            if self.device is not None:
                final_input = final_input.to(self.device, non_blocking=True)
                final_labels = final_labels.to(self.device, non_blocking=True)

            yield {
                "input_ids": final_input,
                "labels": final_labels,
            }
