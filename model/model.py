"""Full SparDA-Style Local Research Language Model implementation."""

from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import ModelConfig
from model.embeddings import TiedEmbedding
from model.norm import RMSNorm
from model.rope import RotaryEmbedding
from model.transformer import TransformerBlock
from model.attention.base import KVCache


class SparDAResearchModel(nn.Module):
    """SparDA-Style Research Language Model built from scratch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared token embedding and output LM head
        self.embeddings = TiedEmbedding(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            initializer_range=config.initializer_range,
        )

        # Precomputed Rotary Position Embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )

        # Transformer layers stack
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])

        # Final layer normalization
        self.final_norm = RMSNorm(config.n_embd, eps=config.norm_eps)

        # Flag for activation gradient checkpointing
        self.gradient_checkpointing = False

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def count_parameters(self) -> Dict[str, int]:
        """Returns detailed parameter breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = self.embeddings.weight.numel()
        return {
            "total": total,
            "trainable": trainable,
            "embeddings": emb_params,
            "layers": total - emb_params,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        return_forecasts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            targets: Optional [batch_size, seq_len] for cross-entropy loss computation
            kv_caches: Optional list of KVCache objects, one per layer
            return_forecasts: If True, returns forecast representations emitted by each layer
            
        Returns:
            Dict containing "logits", optional "loss", and optional "forecasts"
        """
        batch_size, seq_len = input_ids.shape
        x = self.embeddings.embed(input_ids)

        forecasts: List[torch.Tensor] = []
        prev_forecast: Optional[torch.Tensor] = None

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None
            next_cache = kv_caches[i + 1] if (kv_caches is not None and i + 1 < len(kv_caches)) else None

            if self.gradient_checkpointing and self.training and kv_cache is None:
                # Custom forward wrapper for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x, current_forecast = checkpoint(
                    create_custom_forward(layer),
                    x,
                    self.rotary_emb,
                    None,
                    None,
                    True,
                    prev_forecast,
                    use_reentrant=False,
                )
            else:
                x, current_forecast = layer(
                    x=x,
                    rotary_emb=self.rotary_emb,
                    kv_cache=kv_cache,
                    mask=None,
                    is_causal=True,
                    prev_layer_forecast=prev_forecast,
                    next_layer_block=next_layer,
                    next_kv_cache=next_cache,
                )

            if return_forecasts:
                forecasts.append(current_forecast)
            prev_forecast = current_forecast

        x = self.final_norm(x)
        logits = self.embeddings.project(x)

        output = {"logits": logits}
        if return_forecasts:
            output["forecasts"] = forecasts

        if targets is not None:
            # Flatten for cross-entropy calculation
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation using prompt and iterative sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop input context if it exceeds model context length
            context_ids = input_ids if input_ids.shape[1] <= self.config.context_length else input_ids[:, -self.config.context_length:]
            
            outputs = self.forward(context_ids)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
