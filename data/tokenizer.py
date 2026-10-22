"""Custom Byte-fallback BPE Tokenizer for the SparDA Research Model."""

import os
from pathlib import Path
from typing import List, Union, Optional
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class SparDATokenizer:
    """Byte-fallback BPE Tokenizer with 32,000 vocabulary size."""

    SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
    UNK_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    PAD_ID = 3

    def __init__(self, tokenizer_file: Optional[Union[str, Path]] = None):
        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        else:
            # Initialize empty BPE tokenizer with byte-level fallback
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDecoder()

    @classmethod
    def from_file(cls, tokenizer_file: Union[str, Path]) -> "SparDATokenizer":
        """Loads a trained SparDATokenizer from a JSON file."""
        return cls(tokenizer_file=tokenizer_file)

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        save_path: Optional[str] = None,
    ) -> "SparDATokenizer":
        """Trains a new BPE tokenizer from an iterator of strings."""
        instance = cls()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        instance.tokenizer.train_from_iterator(iterator, trainer=trainer)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            instance.save(save_path)

        return instance

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        save_path: Optional[str] = None,
    ) -> "SparDATokenizer":
        """Trains a new BPE tokenizer from raw text files."""
        instance = cls()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        instance.tokenizer.train(files, trainer=trainer)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            instance.save(save_path)

        return instance

    def save(self, path: str):
        self.tokenizer.save(path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            return [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in (self.UNK_ID, self.BOS_ID, self.EOS_ID, self.PAD_ID)]
        return self.tokenizer.decode(token_ids)

    def encode_torch(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
