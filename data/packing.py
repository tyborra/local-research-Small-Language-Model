"""Sequence packing and document segmentation utilities."""

from typing import List, Dict
import torch


def pack_sequences(
    tokenized_docs: List[List[int]],
    max_seq_len: int = 1024,
    pad_token_id: int = 3,
    eos_token_id: int = 2,
) -> List[Dict[str, torch.Tensor]]:
    """Packs variable length token sequences into fixed max_seq_len buffers.
    
    Concatenates documents with eos_token_id to maximize token density during pretraining.
    """
    packed_samples = []
    current_pack = []

    for doc in tokenized_docs:
        # Ensure document ends with eos_token_id
        if len(doc) == 0:
            continue
        if doc[-1] != eos_token_id:
            doc = doc + [eos_token_id]

        current_pack.extend(doc)

        while len(current_pack) >= max_seq_len + 1:
            chunk = current_pack[: max_seq_len + 1]
            current_pack = current_pack[max_seq_len + 1 :]
            
            chunk_tensor = torch.tensor(chunk, dtype=torch.long)
            packed_samples.append({
                "input_ids": chunk_tensor[:-1],
                "labels": chunk_tensor[1:],
            })

    return packed_samples
