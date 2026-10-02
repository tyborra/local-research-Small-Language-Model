"""Main training entrypoint for SparDA Local Research Model."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import numpy as np
import torch

from model.config import ModelConfig
from model.model import SparDAResearchModel
from data.dataset import TokenizedDataset, create_dataloader
from training.trainer import Trainer


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SparDA Local Research Model")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--context_length", type=int, default=None, help="Override context length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml_config(args.config)
    
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.context_length is not None:
        cfg["model"]["context_length"] = args.context_length

    # Device selection
    if args.device is not None:
        device_str = args.device
    else:
        device_str = cfg.get("training", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Set seeds
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load token data
    paths = cfg.get("paths", {})
    train_bin_path = paths.get("train_data_path", "data/train.bin")
    val_bin_path = paths.get("val_data_path", "data/val.bin")

    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"Training data not found at {train_bin_path}. Run scripts/prepare_data.py first.")

    print(f"Loading training data from {train_bin_path}...")
    train_tokens = np.fromfile(train_bin_path, dtype=np.uint16)
    print(f"Loaded {len(train_tokens):,} training tokens.")

    val_tokens = None
    if os.path.exists(val_bin_path):
        val_tokens = np.fromfile(val_bin_path, dtype=np.uint16)
        print(f"Loaded {len(val_tokens):,} validation tokens.")

    seq_len = cfg["model"]["context_length"]
    batch_size = cfg["training"]["batch_size"]

    train_dataset = TokenizedDataset(train_tokens, seq_len=seq_len)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_tokens is not None:
        val_dataset = TokenizedDataset(val_tokens, seq_len=seq_len)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_cfg = ModelConfig(**cfg["model"])
    model = SparDAResearchModel(model_cfg)
    param_counts = model.count_parameters()
    print(f"Initialized SparDAResearchModel:")
    print(f"  Total parameters:     {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Embeddings (tied):    {param_counts['embeddings']:,}")
    print(f"  Layers ({model_cfg.n_layer}x):         {param_counts['layers']:,}")

    # Initialize trainer and run
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
    )

    results = trainer.train()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
