"""Evaluation script for Forecast Block Prediction accuracy against Teacher Attention."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import yaml
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import SparDAResearchModel
from model.forecast import (
    KVBlockCompressor,
    ForecastBlockSelector,
    TeacherBlockSelector,
    compute_forecast_loss,
)
from evaluation.forecast_metrics import evaluate_forecast_metrics
from data.dataset import TokenizedDataset, create_dataloader
from training.checkpoint import load_checkpoint


def run_forecast_evaluation(
    model: SparDAResearchModel,
    dataloader,
    block_size: int = 32,
    top_k_values: List[int] = [1, 2, 4, 8],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """Evaluates one-layer-ahead forecast prediction accuracy against teacher attention weights."""
    model.eval()
    cfg = model.config

    compressor = KVBlockCompressor(
        block_size=block_size,
        head_dim=cfg.head_dim,
        forecast_dim=cfg.forecast_dim,
    ).to(device)

    forecast_selector = ForecastBlockSelector(
        block_size=block_size,
        forecast_dim=cfg.forecast_dim,
    ).to(device)

    teacher_selector = TeacherBlockSelector(
        block_size=block_size,
        num_queries_per_kv=cfg.num_queries_per_kv,
    ).to(device)

    layer_metrics: Dict[int, List[Dict[str, float]]] = {
        i: [] for i in range(cfg.n_layer - 1)
    }

    print(f"\nEvaluating Forecast Block Selection across {cfg.n_layer - 1} layer transitions...")
    print(f"Block size: {block_size} tokens | K thresholds: {top_k_values}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            batch_size, seq_len = input_ids.shape

            # Step through layers manually to hook Layer L forecast -> Layer L+1 keys/attention
            x = model.embeddings.embed(input_ids)
            prev_forecast = None

            for layer_idx in range(cfg.n_layer):
                layer = model.layers[layer_idx]
                normed_x = layer.attn_norm(x)
                
                # Emit forecast from current layer
                current_forecast = layer.forecast(normed_x)  # [B, T, 4, 64]

                # Extract key projections for block compression
                xk = layer.attention.wk(normed_x).view(batch_size, seq_len, cfg.n_kv_head, cfg.head_dim)
                xq = layer.attention.wq(normed_x).view(batch_size, seq_len, cfg.n_head, cfg.head_dim)
                xq, xk_rot = model.rotary_emb(xq, xk, offset=0)

                # Compute dense attention output and attention weights for teacher extraction
                attn_out, attn_weights = layer.attention(
                    normed_x,
                    rotary_emb=model.rotary_emb,
                    return_attn_weights=True,
                )

                # Residual + FFN
                x = x + attn_out
                x = x + layer.ffn(layer.ffn_norm(x))

                # Evaluate forecast emitted by Layer L-1 predicting Layer L
                if layer_idx > 0 and prev_forecast is not None:
                    # 1. Compress keys of current layer (L)
                    block_reps = compressor(xk)  # [B, N_b, n_kv_head, forecast_dim]

                    # 2. Forecast selection from Layer L-1
                    max_k = max(top_k_values)
                    pred_scores, pred_topk = forecast_selector(
                        forecast=prev_forecast,
                        block_reps=block_reps,
                        top_k=max_k,
                        is_causal=True,
                    )

                    # 3. Ground-truth teacher selection from current layer (L) attention weights
                    _, teacher_topk, target_dist = teacher_selector(
                        attn_weights=attn_weights,
                        top_k=max_k,
                        is_causal=True,
                    )

                    # 4. Evaluate metrics
                    metrics = evaluate_forecast_metrics(
                        pred_scores=pred_scores,
                        pred_indices=pred_topk,
                        target_indices=teacher_topk,
                        target_distribution=target_dist,
                        k_values=top_k_values,
                    )
                    layer_metrics[layer_idx - 1].append(metrics)

                prev_forecast = current_forecast

    # Aggregate across batches
    summary = {}
    for l_idx, batch_list in layer_metrics.items():
        if not batch_list:
            continue
        agg = {}
        for k in batch_list[0].keys():
            agg[k] = float(np.mean([m[k] for m in batch_list]))
        summary[f"Layer_{l_idx}_to_{l_idx+1}"] = agg

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Forecast Accuracy vs Teacher")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--val_data", type=str, default="data/val.bin")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    model_cfg = ModelConfig(**cfg["model"])
    model = SparDAResearchModel(model_cfg).to(device)

    ckpt_path = args.checkpoint or os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints/base"), "model_step_100.pt")
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model=model, device=device)
        print(f"Loaded checkpoint from {ckpt_path}")

    val_tokens = np.fromfile(args.val_data, dtype=np.uint16)
    dataset = TokenizedDataset(val_tokens, seq_len=model_cfg.context_length)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    results = run_forecast_evaluation(
        model=model,
        dataloader=dataloader,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        max_batches=args.max_batches,
    )

    print("\n" + "=" * 80)
    print("FORECAST ACCURACY VS TEACHER ATTENTION (Randomly Initialized Forecast Heads)")
    print("=" * 80)
    for transition, m in results.items():
        print(f"[{transition}]")
        print(f"  Top-1 Accuracy: {m['accuracy_top1']*100:.2f}%")
        print(f"  Recall@1:       {m['recall@1']*100:.2f}% | Precision@1: {m['precision@1']*100:.2f}%")
        print(f"  Recall@4:       {m['recall@4']*100:.2f}% | Precision@4: {m['precision@4']*100:.2f}%")
        print(f"  Recall@8:       {m['recall@8']*100:.2f}% | Precision@8: {m['precision@8']*100:.2f}%")
        if "kl_divergence" in m:
            print(f"  KL Divergence:  {m['kl_divergence']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
