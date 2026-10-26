"""TinyStories BPE corpus: download a slice, normalise, train a tokenizer, pack token streams.

Only the first part of the 1.9 GB train file is fetched, using an HTTP range request.

    uv run python -m core.data            # prepares data/bpe4096/ if missing
"""

import argparse
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .tokenizers import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
SEPARATOR = "<|endoftext|>"

# Typographic characters that GPT-written stories use, mapped to plain ASCII.
REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def fetch_head(name: str, n_bytes: int) -> str:
    raw = DATA_DIR / "raw" / f"{name}.head{n_bytes}"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BASE_URL + name, headers={"Range": f"bytes=0-{n_bytes - 1}"})
        print(f"downloading first {n_bytes / 1e6:.0f} MB of {name}")
        with urllib.request.urlopen(req) as r:
            raw.write_bytes(r.read())
    return raw.read_bytes().decode("utf-8", errors="ignore")


def normalise(story: str) -> str:
    story = unicodedata.normalize("NFKC", story)
    for a, b in REPLACEMENTS.items():
        story = story.replace(a, b)
    return story.strip()


def split_stories(text: str) -> list[str]:
    parts = text.split(SEPARATOR)[:-1]   # last piece is cut off by the byte range
    return [s for s in (normalise(p) for p in parts) if s]


BPE_VOCAB = 4096
BPE_DIR = DATA_DIR / f"bpe{BPE_VOCAB}"


def build_bpe(train_mb: float = 250, val_mb: float = 5, tokenizer_mb: float = 60) -> None:
    """A TinyStories slice, a 4,096-token BPE trained on part of it, and uint16 streams."""
    train = split_stories(fetch_head("TinyStories-train.txt", int(train_mb * 1e6)))
    val = split_stories(fetch_head("TinyStories-valid.txt", int(val_mb * 1e6)))
    sample, size = [], 0
    for s in train:
        if size > tokenizer_mb * 1e6:
            break
        sample.append(s)
        size += len(s)
    print(f"training a {BPE_VOCAB}-token BPE on {len(sample):,} stories")
    tok = BPETokenizer.train(sample, BPE_VOCAB)
    BPE_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(BPE_DIR / "tokenizer.json")

    def pack(stories: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(stories), 10_000):
            encoded = tok.encode_batch(stories[start:start + 10_000])
            chunks.append(np.fromiter((i for story in encoded for i in [*story, tok.eos_id]), dtype=np.uint16))
        return np.concatenate(chunks)

    train_ids, val_ids = pack(train), pack(val)
    train_ids.tofile(BPE_DIR / "train.bin")
    val_ids.tofile(BPE_DIR / "val.bin")
    print(f"train {len(train_ids):,} tokens ({len(train):,} stories), val {len(val_ids):,} tokens")


class Data:
    """Memory-mapped train/val token streams with random-window batching."""

    def __init__(self, data_dir: Path = BPE_DIR, need_train: bool = True):
        if self._missing(data_dir, need_train):
            build_bpe()
        self.tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
        self.splits = {s: np.memmap(data_dir / f"{s}.bin", dtype=np.uint16, mode="r")
                       for s in ("train", "val") if (data_dir / f"{s}.bin").exists()}

    @staticmethod
    def _missing(data_dir: Path, need_train: bool) -> bool:
        """Only val.bin is needed to run a model (neuron examples); train.bin only to train it."""
        return not (data_dir / "val.bin").exists() or (need_train and not (data_dir / "train.bin").exists())

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


def data_dir() -> Path:
    return BPE_DIR


def load_data(need_train: bool = True) -> Data:
    """need_train=False: a model that only runs, so a corpus is built only if val.bin is missing."""
    return Data(BPE_DIR, need_train)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-mb", type=float, default=None)
    p.add_argument("--val-mb", type=float, default=None)
    args = p.parse_args()
    build_bpe(args.train_mb or 250, args.val_mb or 5)
