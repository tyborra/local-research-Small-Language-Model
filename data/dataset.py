"""Dataset and DataLoader wrappers for pretraining and fine-tuning."""

from typing import Union, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenizedDataset(Dataset):
    """Memory-efficient chunked token dataset for causal language modeling."""

    def __init__(
        self,
        token_ids: Union[torch.Tensor, np.ndarray, List[int]],
        seq_len: int = 1024,
    ):
        super().__init__()
        self.seq_len = seq_len
        
        if isinstance(token_ids, list):
            self.tokens = torch.tensor(token_ids, dtype=torch.long)
        elif isinstance(token_ids, np.ndarray):
            self.tokens = torch.from_numpy(token_ids).long()
        else:
            self.tokens = token_ids.long()

        # Each sample needs seq_len + 1 tokens (input_ids of length seq_len, labels of length seq_len shifted by 1)
        self.chunk_size = seq_len + 1
        self.num_samples = len(self.tokens) // self.chunk_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start_idx = idx * self.chunk_size
        chunk = self.tokens[start_idx : start_idx + self.chunk_size]
        
        input_ids = chunk[:-1]
        labels = chunk[1:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
