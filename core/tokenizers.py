"""The byte-level BPE tokenizer, behind a small interface the rest of the core uses."""

from pathlib import Path
from typing import Protocol

EOS = "\x04"          # end-of-story marker, replaces <|endoftext|>
EOS_DISPLAY = "␄"


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def token_text(self, id: int) -> str: ...
    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...


class BPETokenizer:
    """Byte-level BPE (Hugging Face `tokenizers`), as in GPT-2: common words become single
    tokens, usually with their leading space ("␣dog"), and any text can be encoded."""

    EOS_TOKEN = "<|endoftext|>"

    def __init__(self, tok):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.eos_id = tok.token_to_id(self.EOS_TOKEN)
        # Display text per token, computed once: decoding single tokens is the slow part.
        self._display = [self._show(i) for i in range(self.vocab_size)]

    @classmethod
    def train(cls, texts, vocab_size: int) -> "BPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[cls.EOS_TOKEN],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path) -> None:
        self.tok.save(str(path))

    def encode(self, text: str) -> list[int]:
        # The end-of-story marker may arrive as the raw control character or its display form.
        parts = text.replace(EOS_DISPLAY, EOS).split(EOS)
        ids: list[int] = []
        for i, part in enumerate(parts):
            if i:
                ids.append(self.eos_id)
            if part:
                ids += self.tok.encode(part).ids
        return ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids and the [start, end) character span each came from (no end-of-story markers)."""
        e = self.tok.encode(text)
        return e.ids, [tuple(o) for o in e.offsets]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        out, run = [], []
        for i in ids:
            if i == self.eos_id:
                out += [self.tok.decode(run), EOS]
                run = []
            else:
                run.append(i)
        return "".join(out) + self.tok.decode(run)

    def _show(self, id: int) -> str:
        if id == self.eos_id:
            return EOS_DISPLAY
        return self.tok.decode([id]).replace(" ", "␣").replace("\n", "⏎")

    def token_text(self, id: int) -> str:
        return self._display[id]
