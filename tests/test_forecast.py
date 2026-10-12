"""Unit tests for Phase 3: Forecast projection, block selection, teacher extraction, and metrics."""

import math
import pytest
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.forecast import (
    ForecastProjection,
    KVBlockCompressor,
    ForecastBlockSelector,
    TeacherBlockSelector,
    compute_forecast_loss,
)
from evaluation.forecast_metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_block_selection_accuracy,
    compute_kl_divergence,
    evaluate_forecast_metrics,
)


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=256,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        head_dim=16,
        ffn_dim=128,
        forecast_heads=2,
        forecast_dim=16,
        context_length=64,
    )


def test_forecast_projection_shape_and_grad(config):
    fp = ForecastProjection(config)
    x = torch.randn(2, 32, config.n_embd, requires_grad=True)
    out = fp(x)

    assert out.shape == (2, 32, config.forecast_heads, config.forecast_dim)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_kv_block_compressor():
    batch, seq_len, n_kv_head, head_dim = 2, 64, 4, 32
    block_size = 16
    compressor = KVBlockCompressor(block_size=block_size, head_dim=head_dim, forecast_dim=32)

    k = torch.randn(batch, seq_len, n_kv_head, head_dim)
    blocks = compressor(k)

    expected_num_blocks = seq_len // block_size  # 4
    assert blocks.shape == (batch, expected_num_blocks, n_kv_head, 32)

    # Test padding when seq_len is not divisible by block_size
    k_odd = torch.randn(batch, 50, n_kv_head, head_dim)
    blocks_odd = compressor(k_odd)
    # ceil(50 / 16) = 4
    assert blocks_odd.shape == (batch, 4, n_kv_head, 32)


def test_forecast_block_selector_causal():
    batch, seq_len_q, n_kv_head, forecast_dim = 2, 32, 2, 16
    num_blocks = 4
    block_size = 8

    selector = ForecastBlockSelector(block_size=block_size, forecast_dim=forecast_dim)
    forecast = torch.randn(batch, seq_len_q, n_kv_head, forecast_dim)
    block_reps = torch.randn(batch, num_blocks, n_kv_head, forecast_dim)

    scores, topk = selector(forecast, block_reps, top_k=2, is_causal=True)

    assert scores.shape == (batch, n_kv_head, seq_len_q, num_blocks)
    assert topk.shape == (batch, n_kv_head, seq_len_q, 2)

    # For token 0 (block 0), blocks 1, 2, 3 must be masked with -inf
    assert torch.isneginf(scores[:, :, 0, 1:]).all()
    # For token 7 (block 0), blocks 1, 2, 3 must be masked with -inf
    assert torch.isneginf(scores[:, :, 7, 1:]).all()
    # For token 8 (block 1), block 0 and 1 are accessible, blocks 2, 3 must be -inf
    assert torch.isfinite(scores[:, :, 8, 0]).all()
    assert torch.isfinite(scores[:, :, 8, 1]).all()
    assert torch.isneginf(scores[:, :, 8, 2:]).all()


def test_teacher_block_selector():
    batch, n_head, seq_len_q, seq_len_k = 2, 6, 16, 16
    num_queries_per_kv = 3  # n_kv_head = 2
    block_size = 4
    num_blocks = seq_len_k // block_size  # 4

    teacher = TeacherBlockSelector(block_size=block_size, num_queries_per_kv=num_queries_per_kv)

    # Construct mock attention weights (softmax-normalized across keys)
    raw_attn = torch.randn(batch, n_head, seq_len_q, seq_len_k)
    causal_mask = torch.triu(torch.full((seq_len_q, seq_len_k), float("-inf")), diagonal=1)
    attn_weights = F.softmax(raw_attn + causal_mask, dim=-1)

    block_scores, topk, target_dist = teacher(attn_weights, top_k=2, is_causal=True)

    assert block_scores.shape == (batch, 2, seq_len_q, num_blocks)
    assert topk.shape == (batch, 2, seq_len_q, 2)
    assert target_dist.shape == (batch, 2, seq_len_q, num_blocks)

    # Future blocks should have 0.0 in target distribution
    assert (target_dist[:, :, 0, 1:] == 0.0).all()

    # Target distribution across valid blocks should sum to approximately 1.0
    row_sums = target_dist.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_compute_forecast_loss():
    batch, n_kv_head, seq_len_q, num_blocks = 2, 2, 16, 4
    pred_scores = torch.randn(batch, n_kv_head, seq_len_q, num_blocks, requires_grad=True)

    # Causal mask on pred_scores
    t_idx = torch.arange(seq_len_q).view(1, 1, seq_len_q, 1)
    b_idx = torch.arange(num_blocks).view(1, 1, 1, num_blocks)
    pred_scores_masked = pred_scores.masked_fill(b_idx > (t_idx // 4), float("-inf"))

    target_dist = F.softmax(torch.randn(batch, n_kv_head, seq_len_q, num_blocks), dim=-1)
    target_dist = target_dist.masked_fill(b_idx > (t_idx // 4), 0.0)
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Test KL loss
    loss_kl = compute_forecast_loss(pred_scores_masked, target_dist, loss_type="kl")
    assert torch.isfinite(loss_kl)
    assert loss_kl.item() >= 0.0
    loss_kl.backward()
    assert pred_scores.grad is not None

    # Test BCE loss with fresh tensor
    pred_scores_2 = torch.randn(batch, n_kv_head, seq_len_q, num_blocks, requires_grad=True)
    pred_scores_masked_2 = pred_scores_2.masked_fill(b_idx > (t_idx // 4), float("-inf"))
    loss_bce = compute_forecast_loss(pred_scores_masked_2, target_dist, loss_type="bce")
    assert torch.isfinite(loss_bce)
    assert loss_bce.item() >= 0.0
    loss_bce.backward()
    assert pred_scores_2.grad is not None


def test_forecast_evaluation_metrics():
    # Synthetic predictions and targets
    # Suppose we have 4 tokens, k=2
    # Token 0: pred=[0, 1], target=[0, 1] -> perfect match
    # Token 1: pred=[0, 2], target=[0, 1] -> 1 match
    # Token 2: pred=[2, 3], target=[0, 1] -> 0 matches
    pred = torch.tensor([
        [[[0, 1], [0, 2], [2, 3]]]
    ])
    target = torch.tensor([
        [[[0, 1], [0, 1], [0, 1]]]
    ])

    recall_2 = compute_recall_at_k(pred, target, k=2)
    # Token 0: 2/2 = 1.0, Token 1: 1/2 = 0.5, Token 2: 0/2 = 0.0 -> mean = 0.5
    assert math.isclose(recall_2, 0.5, rel_tol=1e-5)

    prec_2 = compute_precision_at_k(pred, target, k=2)
    assert math.isclose(prec_2, 0.5, rel_tol=1e-5)

    acc = compute_block_selection_accuracy(pred, target)
    # Token 0: 0==0 (1), Token 1: 0==0 (1), Token 2: 2==0 (0) -> mean = 2/3 = 0.6667
    assert math.isclose(acc, 2.0 / 3.0, rel_tol=1e-4)

    # Test evaluate_forecast_metrics aggregator
    metrics = evaluate_forecast_metrics(
        pred_scores=torch.randn(1, 1, 3, 4),
        pred_indices=pred,
        target_indices=target,
        k_values=[1, 2],
    )
    assert "accuracy_top1" in metrics
    assert "recall@1" in metrics
    assert "recall@2" in metrics
    assert "precision@1" in metrics
    assert "precision@2" in metrics
