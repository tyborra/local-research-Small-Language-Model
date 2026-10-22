"""Unit tests for Continual Learning, Replay Buffer, EWC, and Forgetting Metrics."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.model import SparDAResearchModel
from data.continual import ContinualDatasetManager, ContinualStage
from training.replay import ReplayBuffer, ContinualMixedDataLoader
from training.ewc import ElasticWeightConsolidation, EWCRegularizer
from training.continual_trainer import ContinualTrainer
from evaluation.continual import (
    evaluate_cross_stage_performance,
    compute_continual_metrics,
    compute_backward_transfer,
    compute_forgetting,
    generate_continual_report,
    ContinualEvaluator,
)


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=32000,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        head_dim=16,
        ffn_dim=128,
        forecast_heads=2,
        forecast_dim=16,
        context_length=64,
        block_size=8,
        top_k_blocks=2,
        attention_backend="dense",
    )


def test_continual_dataset_stages():
    mgr = ContinualDatasetManager(tokenizer=None, seq_len=32)
    stages = mgr.get_all_stages()

    assert len(stages) == 3
    for stage in stages:
        assert stage.stage_idx in [1, 2, 3]
        assert len(stage.train_tokens) > 0
        assert len(stage.val_tokens) > 0
        assert len(stage.train_dataset) > 0
        assert len(stage.val_dataset) > 0

        # Sample check
        sample = stage.train_dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)


def test_replay_buffer_and_stratification():
    buf = ReplayBuffer(capacity=50, stratified=True)
    assert buf.is_empty()
    assert len(buf) == 0

    # Add samples from stage 1 and stage 2
    for _ in range(30):
        buf.add_sample(torch.randint(0, 100, (16,)), torch.randint(0, 100, (16,)), stage_id=1)
    for _ in range(30):
        buf.add_sample(torch.randint(0, 100, (16,)), torch.randint(0, 100, (16,)), stage_id=2)

    assert not buf.is_empty()
    assert len(buf) <= 50

    # Sample batch
    batch = buf.sample_batch(batch_size=8)
    assert batch["input_ids"].shape == (8, 16)
    assert batch["labels"].shape == (8, 16)


def test_continual_mixed_dataloader():
    mgr = ContinualDatasetManager(tokenizer=None, seq_len=16)
    stage1 = mgr.get_stage(1)
    stage2 = mgr.get_stage(2)

    buf = ReplayBuffer(capacity=100)
    buf.add_from_dataset(stage1.train_dataset, stage_id=1, max_samples=20)

    loader2 = DataLoader(stage2.train_dataset, batch_size=4, shuffle=False)
    mixed_loader = ContinualMixedDataLoader(
        current_loader=loader2,
        replay_buffer=buf,
        replay_ratio=0.25,  # 1 out of 4 is replay
    )

    batch = next(iter(mixed_loader))
    assert batch["input_ids"].shape == (4, 16)
    assert batch["labels"].shape == (4, 16)


def test_ewc_fisher_and_penalty(tiny_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparDAResearchModel(tiny_config).to(device)

    mgr = ContinualDatasetManager(tokenizer=None, seq_len=16)
    stage1 = mgr.get_stage(1)
    loader = DataLoader(stage1.train_dataset, batch_size=2, shuffle=True)

    ewc = ElasticWeightConsolidation(model=model, importance_lambda=200.0, protect_target="all")
    assert len(ewc.fisher_matrix) == 0

    # Register stage 1
    mean_fishers = ewc.register_stage(dataloader=loader, stage_id=1, max_batches=5)
    assert len(ewc.fisher_matrix) > 0
    assert len(ewc.optimal_params) > 0

    # Immediately after registering, penalty should be zero (parameters haven't moved)
    init_penalty = ewc.penalty(model)
    assert torch.isclose(init_penalty, torch.tensor(0.0, device=device), atol=1e-5)

    # Perturb weights slightly
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)

    pert_penalty = ewc.penalty(model)
    assert pert_penalty.item() > 0.0


def test_ewc_selective_forecast_protection(tiny_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparDAResearchModel(tiny_config).to(device)

    mgr = ContinualDatasetManager(tokenizer=None, seq_len=16)
    stage1 = mgr.get_stage(1)
    loader = DataLoader(stage1.train_dataset, batch_size=2)

    # Protect only forecast heads
    ewc_fc = ElasticWeightConsolidation(model=model, importance_lambda=100.0, protect_target="forecast")
    ewc_fc.register_stage(dataloader=loader, stage_id=1, max_batches=3)

    for name in ewc_fc.fisher_matrix:
        assert "forecast" in name


def test_continual_evaluation_metrics():
    # Synthetic trajectory: 3 stages
    # Stage 1: learns well, Stage 2: partial forgetting on Stage 1, Stage 3: further forgetting
    traj = [
        {
            1: {"eval_loss": 2.0, "perplexity": 7.39},
            2: {"eval_loss": 4.5, "perplexity": 90.0},
            3: {"eval_loss": 5.0, "perplexity": 148.0},
        },
        {
            1: {"eval_loss": 2.8, "perplexity": 16.44},
            2: {"eval_loss": 2.1, "perplexity": 8.16},
            3: {"eval_loss": 4.8, "perplexity": 121.0},
        },
        {
            1: {"eval_loss": 3.2, "perplexity": 24.53},
            2: {"eval_loss": 2.7, "perplexity": 14.88},
            3: {"eval_loss": 2.2, "perplexity": 9.02},
        },
    ]

    metrics = compute_continual_metrics(traj, metric_key="eval_loss")
    # Loss increased on stage 1 (2.0 -> 3.2) and stage 2 (2.1 -> 2.7) => BWT is negative
    assert metrics["backward_transfer"] < 0.0
    assert metrics["average_forgetting"] > 0.0

    bwt = compute_backward_transfer(traj, metric_key="eval_loss")
    fgt = compute_forgetting(traj, metric_key="eval_loss")
    assert bwt == metrics["backward_transfer"]
    assert fgt == metrics["average_forgetting"]

    # Test report generation
    report = generate_continual_report({"TestStrategy": traj})
    assert "# Continual Learning & Catastrophic Forgetting Analysis" in report
    assert "TestStrategy" in report


def test_end_to_end_continual_trainer_step(tiny_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparDAResearchModel(tiny_config).to(device)

    mgr = ContinualDatasetManager(tokenizer=None, seq_len=16)
    stages = [mgr.get_stage(1), mgr.get_stage(2)]

    trainer = ContinualTrainer(
        model=model,
        stages=stages,
        device=device,
        strategy="replay_and_ewc",
        replay_ratio=0.20,
        replay_capacity=50,
        ewc_lambda=50.0,
        steps_per_stage=3,
        batch_size=2,
        grad_accum_steps=1,
    )

    summary = trainer.train_all_stages()
    assert summary["strategy"] == "replay_and_ewc"
    assert len(summary["stage_summaries"]) == 2
    assert "metrics_loss" in summary
    assert "metrics_ppl" in summary
    assert "backward_transfer" in summary["metrics_loss"]
