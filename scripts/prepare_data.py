"""Downloads text corpus, trains 32K Byte-fallback BPE tokenizer, and tokenizes into binary files."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import urllib.request
import numpy as np
from tqdm import tqdm

from data.tokenizer import SparDATokenizer

CORPUS_URLS = {
    "wikitext2_train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "wikitext2_valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download_file(url: str, dest_path: str):
    """Downloads a file with a progress bar."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url} -> {dest_path}")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.info().get("Content-Length", 0))
        block_size = 8192
        with open(dest_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                pbar.update(len(buffer))


def prepare_data(data_dir: str = "data", vocab_size: int = 32000):
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Download raw texts
    train_raw_path = os.path.join(raw_dir, "train_raw.txt")
    val_raw_path = os.path.join(raw_dir, "val_raw.txt")

    # Download source components
    wiki_train = os.path.join(raw_dir, "wikitext2_train.txt")
    wiki_val = os.path.join(raw_dir, "wikitext2_valid.txt")
    shakespeare = os.path.join(raw_dir, "shakespeare.txt")

    download_file(CORPUS_URLS["wikitext2_train"], wiki_train)
    download_file(CORPUS_URLS["wikitext2_valid"], wiki_val)
    download_file(CORPUS_URLS["shakespeare"], shakespeare)

    # Combine training files
    print("Combining raw corpora for training...")
    with open(train_raw_path, "w", encoding="utf-8") as outfile:
        for fpath in [wiki_train, shakespeare]:
            with open(fpath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    with open(val_raw_path, "w", encoding="utf-8") as outfile:
        with open(wiki_val, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())

    # 2. Train 32K BPE Tokenizer
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    print(f"Training Byte-fallback BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = SparDATokenizer.train_from_files(
        files=[train_raw_path],
        vocab_size=vocab_size,
        min_frequency=2,
        save_path=tokenizer_path,
    )
    print(f"Tokenizer trained with vocab_size={tokenizer.vocab_size} and saved to {tokenizer_path}")

    # 3. Tokenize train and validation datasets into uint16 arrays
    print("Tokenizing training text...")
    with open(train_raw_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    train_ids = tokenizer.encode(train_text)
    train_arr = np.array(train_ids, dtype=np.uint16)
    train_bin_path = os.path.join(data_dir, "train.bin")
    train_arr.tofile(train_bin_path)
    print(f"Saved {len(train_arr):,} training tokens to {train_bin_path}")

    print("Tokenizing validation text...")
    with open(val_raw_path, "r", encoding="utf-8") as f:
        val_text = f.read()

    val_ids = tokenizer.encode(val_text)
    val_arr = np.array(val_ids, dtype=np.uint16)
    val_bin_path = os.path.join(data_dir, "val.bin")
    val_arr.tofile(val_bin_path)
    print(f"Saved {len(val_arr):,} validation tokens to {val_bin_path}")

    print("Data preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and tokenizer for SparDA model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store datasets")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()

    prepare_data(data_dir=args.data_dir, vocab_size=args.vocab_size)
