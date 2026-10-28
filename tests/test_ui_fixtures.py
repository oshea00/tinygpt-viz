"""Writes the binary frames the UI's Vitest suite decodes, so both sides test one contract."""

import json
from pathlib import Path

import torch

from core.model import GPT
from core.sampling import SampleSettings, shape_distribution
from core.trace_io import decode_trace, encode_trace, encode_weights
from core.tracer import trace

from .conftest import TINY

FIXTURES = Path(__file__).resolve().parent.parent / "ui" / "src" / "fixtures"


def test_write_ui_fixtures():
    torch.manual_seed(0)
    model = GPT(TINY.with_vocab(16))
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
    t = trace(model, ids, [chr(97 + i) for i in ids], step=42, next_token=9)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "trace.bin").write_bytes(encode_trace(t, {"prompt": "fixture", "reason": "run"}))
    w = {n: p.detach().numpy() for n, p in model.named_weights().items()}
    (FIXTURES / "weights.bin").write_bytes(encode_weights(w, 42, {"version": 1}))
    ablate = [{"kind": "head", "layer": 1, "head": 0, "mode": "zero"}]
    t2 = trace(model, ids, [chr(97 + i) for i in ids], step=42, next_token=9, interventions=ablate)
    (FIXTURES / "trace_ablated.bin").write_bytes(encode_trace(t2, {"prompt": "fixture", "reason": "run", "interventions": ablate}))
    assert decode_trace((FIXTURES / "trace.bin").read_bytes()).step == 42


def test_write_sampling_fixture():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(16, generator=g) * 2
    cases = []
    for t, k, p in [(1.0, None, None), (0.5, None, None), (2.0, None, None), (0, None, None),
                    (1.0, 3, None), (1.0, None, 0.5), (0.8, 5, 0.9), (1.0, None, 0.999)]:
        probs = shape_distribution(logits, SampleSettings(t, k, p))
        cases.append({"temperature": t, "topK": k, "topP": p, "probs": probs.tolist()})
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "sampling.json").write_text(json.dumps({"logits": logits.tolist(), "cases": cases}))


# Prose like the training stories, plus the cases byte-level BPE gets wrong most easily.
TOKENIZER_TEXTS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.",
    'One day, Tom said, "I don\'t want to go!" His mom smiled. "It\'s okay, we\'ll go tomorrow."',
    "The big, brown dog ran fast.\n\nThen it stopped.\nIt was tired, so it lay down under a tree.",
    "Ben had 3 red balls and 12 blue ones. He gave 1 to Sue; she said: thank you!!!",
    "  two leading spaces, a tab\there, trailing spaces   ",
    "café, naïve, Zoë, résumé — and “curly quotes” … ellipsis",
    "emoji 😀👍🏽 and 漢字 and العربية",
    "she'd, they've, I'm, you're, we'll, it's, 'quoted'",
    "end of one story\x04start of the next ␄ and a literal <|endoftext|> token",
    "a",
    " ",
    "\n",
]


def test_write_tokenizer_fixture():
    from core.data import BPE_DIR
    from core.tokenizers import BPETokenizer

    tok = BPETokenizer.load(BPE_DIR / "tokenizer.json")
    cases = []
    for text in TOKENIZER_TEXTS:
        ids, offsets = tok.encode_with_offsets(text.replace("\x04", "").replace("␄", ""))
        cases.append({"text": text, "ids": tok.encode(text), "decoded": tok.decode(tok.encode(text)),
                      "offsetText": text.replace("\x04", "").replace("␄", ""),
                      "offsetIds": ids, "offsets": offsets})
    display = [tok.token_text(i) for i in range(tok.vocab_size)]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "tokenizer.json").write_text(json.dumps({"cases": cases, "display": display}, ensure_ascii=False))
