"""Interactive text generation script for SparDA Local Research Model."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from model.config import ModelConfig
from model.model import SparDAResearchModel
from data.tokenizer import SparDATokenizer
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from SparDA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load tokenizer
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer}")
    tokenizer = SparDATokenizer(args.tokenizer)

    # Load checkpoint and model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    # Inspect checkpoint config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = SparDAResearchModel(config)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint} (step {ckpt.get('step', '?')})")
    print(f"Prompt: {args.prompt}")

    # Encode prompt
    input_ids = tokenizer.encode_torch(args.prompt, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.EOS_ID,
        )

    # Decode and print
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print("\n--- Generated Output ---")
    print(generated_text)
    print("------------------------\n")


if __name__ == "__main__":
    main()
