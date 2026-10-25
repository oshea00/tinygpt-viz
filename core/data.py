"""TinyStories character corpus: download a slice, normalise, build the charset.

Only the first few MB of the 1.9 GB train file are fetched, using an HTTP range request.

    uv run python -m core.data            # prepares data/ if missing
"""

import argparse
import collections
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .tokenizers import EOS, CharTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
SEPARATOR = "<|endoftext|>"

# Typographic characters that GPT-written stories use, mapped to plain ASCII.
REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}
MIN_CHAR_COUNT = 50   # characters rarer than this in the slice are dropped with their story


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


def build(train_mb: float = 10, val_mb: float = 1) -> None:
    train = split_stories(fetch_head("TinyStories-train.txt", int(train_mb * 1.6e6)))
    val = split_stories(fetch_head("TinyStories-valid.txt", int(val_mb * 1.6e6)))

    counts = collections.Counter("".join(train))
    keep = {c for c, n in counts.items() if n >= MIN_CHAR_COUNT and (c.isascii() or c == "\n")}
    chars = sorted(keep | {EOS})
    tok = CharTokenizer(chars)

    def pack(stories: list[str], limit: int) -> np.ndarray:
        clean = [s for s in stories if all(c in keep for c in s)]
        text = "".join(s + EOS for s in clean)[:limit]
        return np.array(tok.encode(text), dtype=np.uint8)

    train_ids = pack(train, int(train_mb * 1e6))
    val_ids = pack(val, int(val_mb * 1e6))
    DATA_DIR.mkdir(exist_ok=True)
    tok.save(DATA_DIR / "char_vocab.json")
    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")
    print(f"vocab {tok.vocab_size}: {''.join(tok.token_text(i) for i in range(tok.vocab_size))}")
    print(f"train {len(train_ids):,} chars, val {len(val_ids):,} chars")


class CharData:
    """Memory-mapped train/val token streams with random-window batching."""

    def __init__(self, data_dir: Path = DATA_DIR):
        if not (data_dir / "train.bin").exists():
            build()
        self.tokenizer = CharTokenizer.load(data_dir / "char_vocab.json")
        self.splits = {
            s: np.memmap(data_dir / f"{s}.bin", dtype=np.uint8, mode="r") for s in ("train", "val")
        }

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-mb", type=float, default=10)
    p.add_argument("--val-mb", type=float, default=1)
    args = p.parse_args()
    build(args.train_mb, args.val_mb)
