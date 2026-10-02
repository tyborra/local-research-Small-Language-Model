"""Unit tests for Tokenizer, Dataset, and Sequence Packing."""

import os
import pytest
import torch
from data.tokenizer import SparDATokenizer
from data.dataset import TokenizedDataset, create_dataloader
from data.packing import pack_sequences


def test_tokenizer_training_and_encoding(tmp_path):
    corpus = [
        "Language models predict the next token in a sequence.",
        "Sparse attention mechanisms reduce KV cache memory footprints.",
        "Asynchronous CPU to GPU prefetch overlaps data movement with tensor compute.",
        "Continual learning investigates catastrophic forgetting across sequential domains.",
    ] * 20

    save_path = str(tmp_path / "tokenizer.json")
    tokenizer = SparDATokenizer.train_from_iterator(corpus, vocab_size=300, min_frequency=1, save_path=save_path)

    assert os.path.exists(save_path)
    assert tokenizer.vocab_size <= 300

    text = "Language models predict tokens."
    encoded = tokenizer.encode(text)
    assert encoded[0] == tokenizer.BOS_ID
    assert encoded[-1] == tokenizer.EOS_ID

    decoded = tokenizer.decode(encoded)
    assert "Language" in decoded
    assert "tokens" in decoded


def test_tokenized_dataset():
    tokens = list(range(100))
    dataset = TokenizedDataset(tokens, seq_len=10)

    # 100 // 11 = 9 samples
    assert len(dataset) == 9
    sample = dataset[0]

    assert len(sample["input_ids"]) == 10
    assert len(sample["labels"]) == 10
    # Labels should be shifted by 1 relative to input_ids
    assert (sample["input_ids"][1:] == sample["labels"][:-1]).all()


def test_pack_sequences():
    docs = [
        [10, 11, 12],
        [20, 21, 22, 23],
        [30, 31, 32, 33, 34, 35],
    ]
    packed = pack_sequences(docs, max_seq_len=6, eos_token_id=2)
    assert len(packed) >= 1
    for sample in packed:
        assert sample["input_ids"].shape == (6,)
        assert sample["labels"].shape == (6,)
