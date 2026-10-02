"""Unit test for atomic checkpoint saving and loading."""

import os
import torch
from model.config import ModelConfig
from model.model import SparDAResearchModel
from training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    config = ModelConfig(n_layer=2, context_length=64)
    model = SparDAResearchModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = str(tmp_path / "model_test.pt")
    save_checkpoint(ckpt_path, model, optimizer, step=42, loss=3.14, config=config)

    assert os.path.exists(ckpt_path)

    # Load into a new model
    new_model = SparDAResearchModel(config)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    info = load_checkpoint(ckpt_path, model=new_model, optimizer=new_opt)

    assert info["step"] == 42
    assert abs(info["loss"] - 3.14) < 1e-4

    # Compare weights
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
