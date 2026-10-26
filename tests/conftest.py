import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from core.config import ModelConfig
from core.data import BPE_DIR, Data
from core.model import GPT
from core.tokenizers import BPETokenizer

VOCAB = 64

# Small stand-ins for bpe-small, so the tests run in seconds on a CPU.
TINY = ModelConfig("test-tiny", n_layer=2, n_head=2, d_model=32, d_mlp=128)
SMALL = ModelConfig("test-small", n_layer=4, n_head=4, d_model=64, d_mlp=256)

STORIES = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
    "One day, a big dog ran to the tree. The dog was happy and wagged its tail.",
    "Tom had a red ball. He threw the ball to his friend Sam, and they laughed together.",
    "The cat sat on the mat. It was warm and sunny, so the cat fell asleep.",
]


@pytest.fixture
def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY.with_vocab(VOCAB))


@pytest.fixture
def default_model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL.with_vocab(VOCAB))


def make_corpus(path: Path, splits=("train", "val"), repeat: int = 20) -> Path:
    """The tracked tokenizer and a few stories as token streams: a corpus that needs no download."""
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(BPE_DIR / "tokenizer.json", path)
    tok = BPETokenizer.load(path / "tokenizer.json")
    ids = [i for _ in range(repeat) for s in STORIES for i in [*tok.encode(s), tok.eos_id]]
    for split in splits:
        np.array(ids, dtype=np.uint16).tofile(path / f"{split}.bin")
    return path


@pytest.fixture
def data(tmp_path) -> Data:
    return Data(make_corpus(tmp_path / "corpus"))
