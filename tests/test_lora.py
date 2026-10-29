import json

import torch

from core import lora
from core.config import TrainConfig
from core.model import GPT
from core.server import create_app
from core.stages import describe, stage_order
from core.trace_io import decode_trace
from core.tracer import trace
from core.train import Trainer
from fastapi.testclient import TestClient

from .conftest import STORIES, TINY
from .test_server import _until


def _trained(model: GPT, targets=("q", "v"), rank=2, seed=0) -> lora.Adapter:
    """An adapter with random (nonzero) B, as if trained."""
    ad = lora.new_adapter(model, "t", targets, rank, seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for B in ad.B.values():
            B.copy_(torch.randn(B.shape, generator=g) * 0.1)
    return ad


def test_new_adapter_is_a_no_op(tiny_model: GPT):
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    ad = lora.new_adapter(tiny_model, "t", ("q", "v", "mlp.down"), 4)
    base, _ = tiny_model(idx)
    with tiny_model.adapted(ad.pairs()):
        adapted, _ = tiny_model(idx)
    assert torch.equal(base, adapted)
    assert ad.num_params() == 2 * (4 * (32 + 32) * 2 + 4 * (128 + 32))


def test_adapter_equals_merged_weights(tiny_model: GPT):
    idx = torch.tensor([[1, 2, 3, 4, 5, 6]])
    ad = _trained(tiny_model, ("q", "v", "mlp.down"))
    with torch.no_grad(), tiny_model.adapted(ad.pairs(), 0.5):
        adapted, _ = tiny_model(idx)
    merged = GPT(tiny_model.cfg)
    merged.load_state_dict(tiny_model.state_dict())
    with torch.no_grad():
        for sid, (A, B) in ad.pairs().items():
            l, short = sid.split(".", 1)
            W = getattr(merged.blocks[int(l[1:])], {"q": "W_q", "v": "W_v", "mlp.down": "W_down"}[short])
            W += 0.5 * A @ B
        expect, _ = merged(idx)
    assert torch.allclose(adapted, expect, atol=1e-5)


def test_trace_has_lora_stages_only_with_an_adapter(tiny_model: GPT):
    tokens = [1, 2, 3, 4]
    plain = trace(tiny_model, tokens, ["x"] * 4)
    assert not any(s.meta.id.endswith(".lora") for s in plain.stages)
    ad = _trained(tiny_model)
    with tiny_model.adapted(ad.pairs()):
        t = trace(tiny_model, tokens, ["x"] * 4)
    ids = [s.meta.id for s in t.stages]
    assert ids == stage_order(tiny_model.cfg, set(ad.targets))
    lo = t.stage("L1.q.lora")
    assert lo.meta.kind == "lora" and lo.meta.inputs == ["L1.ln1"] and lo.meta.weights == ["L1.q.lora_A", "L1.q.lora_B"]
    x = t.stage("L1.ln1").output
    A, B = (p.detach().numpy() for p in ad.pairs()["L1.q"])
    assert abs((x @ A @ B) - lo.output).max() < 1e-5
    assert lo.grad is not None and abs(lo.grad).sum() > 0
    # The base weights get no gradient trace-to-trace leftovers from the adapter.
    assert all(p.grad is None for p in tiny_model.parameters())


def test_every_lora_stage_is_described(tiny_model: GPT):
    for sid in stage_order(tiny_model.cfg, lora=True):
        assert describe(sid, tiny_model.cfg).id == sid


def test_training_tilts_towards_the_stories_and_leaves_the_base_alone(data):
    tok = data.tokenizer
    model = GPT(TINY.with_vocab(tok.vocab_size))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ad = lora.new_adapter(model, "cat", ("q", "v", "mlp.down"), 4)
    texts = [STORIES[3]] * 4
    job = lora.LoraJob(model, tok, ad, texts, STORIES[:2], topic="cat", steps=40, lr=1e-2, batch=4, eval_every=20)
    start, last = job.start(), {}
    while not job.done:
        last = job.step()
    assert last["trainLoss"] < start["trainLoss"] - 0.5
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    assert [p["step"] for p in ad.info["curve"]] == [0, 20, 40]


def test_server_trains_an_adapter_and_runs_with_it(tmp_path, data):
    trainer = Trainer(TINY, TrainConfig(batch_size=8, eval_every=1000), data, "cpu", run_dir=tmp_path / "run")
    with TestClient(create_app(trainer, sample_every=10_000, training=False)) as c, c.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        ws.send_text(json.dumps({"type": "loraTrain", "name": "cat", "texts": [STORIES[3]], "topic": "cat",
                                 "steps": 12, "lr": 1e-2, "batch": 2}))
        _until(ws, lambda h: h["type"] == "loraProgress" and not h["running"])
        w, _ = _until(ws, lambda h: h["type"] == "adapter" and h["name"] == "cat")
        assert "L0.q.lora_A" in w["weights"] and w["version"] == 12 and len(w["info"]["curve"]) == 3
        ws.send_text(json.dumps({"type": "run", "prompt": "The cat sat", "adapter": {"name": "cat", "scale": 1}}))
        _, raw = _until(ws, lambda h: h["type"] == "trace" and h.get("prompt") == "The cat sat")
        assert raw is not None and any(s.meta.id == "L1.v.lora" for s in decode_trace(raw).stages)
