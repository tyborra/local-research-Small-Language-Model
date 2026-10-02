"""Evaluation script to measure validation loss, perplexity, and throughput."""

import os
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import SparDAResearchModel
from data.dataset import TokenizedDataset, create_dataloader
from evaluation.perplexity import compute_perplexity
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate SparDA Model Validation Perplexity")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--val_data", type=str, default=None, help="Path to val.bin")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Instantiate model
    model_cfg = ModelConfig(**cfg["model"])
    model = SparDAResearchModel(model_cfg).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/base"), "model_step_100.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt_meta = load_checkpoint(ckpt_path, model=model, device=device)
        print(f"Loaded checkpoint from step {ckpt_meta.get('step', '?')}.")
    else:
        print(f"No checkpoint found at {ckpt_path}. Evaluating untrained randomly initialized model.")

    # Load validation data
    val_path = args.val_data or cfg.get("paths", {}).get("val_data_path", "data/val.bin")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    val_tokens = np.fromfile(val_path, dtype=np.uint16)
    print(f"Loaded {len(val_tokens):,} validation tokens from {val_path}.")

    val_dataset = TokenizedDataset(val_tokens, seq_len=model_cfg.context_length)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {len(val_loader)} batches (batch_size={args.batch_size}, seq_len={model_cfg.context_length})...")

    results = compute_perplexity(
        model=model,
        val_data=val_loader,
        device=device,
        dtype=dtype,
        max_batches=args.max_batches,
    )

    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION METRICS:")
    print("=" * 60)
    print(f"  Cross-Entropy Loss: {results['eval_loss']:.4f} nats")
    print(f"  Perplexity (PPL):   {results['perplexity']:.2f}")
    print(f"  Bits per Token:     {results['bits_per_token']:.4f} bits/tok")
    print(f"  Total Tokens:       {results['total_tokens']:,}")
    print(f"  Total Batches:      {results['total_batches']:,}")
    print(f"  Eval Duration:      {results['eval_time_sec']:.2f} s")
    print(f"  Eval Throughput:    {results['tokens_per_sec']:,.0f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
