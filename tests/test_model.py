"""Unit tests for the SparDAResearchModel architecture."""

import pytest
import torch
from model.config import ModelConfig
from model.model import SparDAResearchModel


def test_model_parameter_count():
    """Verify that the model parameter count matches the exact specification."""
    config = ModelConfig()
    model = SparDAResearchModel(config)
    
    counts = model.count_parameters()
    total = counts["total"]
    
    # Mathematical breakdown with SwiGLU (w_gate, w_up, w_down = 3 * 768 * 2048):
    # Embeddings: 32000 * 768 = 24,576,000
    # Per layer:
    #   Attn: wq(768*768) + wk(768*256) + wv(768*256) + wo(768*768) + wf(768*256) = 1,769,472
    #   SwiGLU: w_gate(768*2048) + w_up(768*2048) + w_down(2048*768) = 4,718,592
    #   Norms: attn_norm(768) + ffn_norm(768) = 1,536
    #   Total per layer = 6,489,600
    # 12 layers: 12 * 6,489,600 = 77,875,200
    # Final norm: 768
    # Total tied model: 24,576,000 + 77,875,200 + 768 = 102,451,968
    assert total == 102_451_968, f"Expected 102,451,968 parameters, got {total}"
    assert counts["trainable"] == 102_451_968


def test_tied_embeddings():
    """Verify that input embeddings and output projection share the identical memory."""
    config = ModelConfig()
    model = SparDAResearchModel(config)
    
    # Verify master weight is identical
    assert model.embeddings.weight.shape == (32000, 768)


def test_forward_pass_and_loss():
    """Test forward pass dimensions and loss computation."""
    config = ModelConfig(context_length=128)
    model = SparDAResearchModel(config)
    model.eval()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids=input_ids, targets=targets, return_forecasts=True)

    # Verify logits shape
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert "loss" in outputs
    assert outputs["loss"].item() > 0

    # Verify forecast representations shape across all 12 layers
    assert len(outputs["forecasts"]) == config.n_layer
    for f in outputs["forecasts"]:
        assert f.shape == (batch_size, seq_len, config.forecast_heads, config.forecast_dim)


def test_backward_pass():
    """Verify backpropagation and gradient flow through all layers and forecast projections."""
    config = ModelConfig(n_layer=2, context_length=64)  # Small config for fast test
    model = SparDAResearchModel(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    outputs = model(input_ids=input_ids, targets=targets, return_forecasts=True)
    # Combine LM loss with forecast loss to verify gradient flow through all paths
    total_loss = outputs["loss"] + 0.01 * sum(f.sum() for f in outputs["forecasts"])
    total_loss.backward()

    # Verify gradients exist on embeddings, attention, forecast, and FFN
    assert model.embeddings.weight.grad is not None
    assert model.layers[0].attention.wq.weight.grad is not None
    assert model.layers[0].forecast.wf.weight.grad is not None
    assert model.layers[0].ffn.w_gate.weight.grad is not None


def test_generate():
    """Verify autoregressive generation method."""
    config = ModelConfig(n_layer=2, context_length=64)
    model = SparDAResearchModel(config)
    model.eval()

    prompt = torch.tensor([[101, 204, 305]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.7)

    assert generated.shape == (1, 8)
