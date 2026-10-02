"""Tied Embedding and LM Head module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedEmbedding(nn.Module):
    """Shares weights between token input embedding and language model output projection.
    
    This preserves parameter efficiency (~84M params instead of ~108M for an untied 32K vocab).
    """

    def __init__(self, vocab_size: int, n_embd: int, initializer_range: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        # Master weight tensor shared between input embedding and output projection
        self.weight = nn.Parameter(torch.empty(vocab_size, n_embd))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds integer token IDs to continuous vectors."""
        return F.embedding(input_ids, self.weight)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Projects continuous vectors back to vocabulary logits."""
        return F.linear(hidden_states, self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)
