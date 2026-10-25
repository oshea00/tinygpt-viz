"""Tokenizers behind one interface, so profiles differ only in config."""

import json
from pathlib import Path
from typing import Protocol

EOS = "\x04"          # end-of-story marker, replaces <|endoftext|>
EOS_DISPLAY = "␄"


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def token_text(self, id: int) -> str: ...


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text())["chars"])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False, indent=0))

    def encode(self, text: str) -> list[int]:
        # Unknown characters are dropped rather than mapped, so prompts can't smuggle
        # in a token the model never saw.
        return [self.stoi[c] for c in text.replace(EOS_DISPLAY, EOS) if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def token_text(self, id: int) -> str:
        c = self.chars[id]
        return {EOS: EOS_DISPLAY, "\n": "⏎", " ": "␣"}.get(c, c)
