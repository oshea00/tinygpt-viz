"""Writes the binary frames the UI's Vitest suite decodes, so both sides test one contract."""

import json
from pathlib import Path

import torch

from core.config import PROFILES
from core.model import GPT
from core.sampling import SampleSettings, shape_distribution
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
