"""Writes the binary frames the UI's Vitest suite decodes, so both sides test one contract."""

from pathlib import Path

import torch

from core.config import PROFILES
from core.model import GPT
from core.trace_io import decode_trace, encode_trace, encode_weights
from core.tracer import trace

FIXTURES = Path(__file__).resolve().parent.parent / "ui" / "src" / "fixtures"


def test_write_ui_fixtures():
    torch.manual_seed(0)
    model = GPT(PROFILES["tiny"].with_vocab(16))
    ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
    t = trace(model, ids, [chr(97 + i) for i in ids], step=42, next_token=9)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "trace.bin").write_bytes(encode_trace(t, {"prompt": "fixture", "reason": "run"}))
    w = {n: p.detach().numpy() for n, p in model.named_weights().items()}
    (FIXTURES / "weights.bin").write_bytes(encode_weights(w, 42, {"version": 1}))
    assert decode_trace((FIXTURES / "trace.bin").read_bytes()).step == 42
