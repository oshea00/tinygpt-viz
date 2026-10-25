import json

import pytest
from fastapi.testclient import TestClient

from core.config import PROFILES, TrainConfig
from core.data import CharData
from core.server import create_app
from core.trace_io import decode_frame, decode_trace
from core.train import Trainer


@pytest.fixture
def client(tmp_path):
    trainer = Trainer(PROFILES["tiny"], TrainConfig(batch_size=8, trace_every=5, eval_every=1000),
                      CharData(), "cpu", run_dir=tmp_path)
    with TestClient(create_app(trainer, sample_every=10_000)) as c:
        yield c


def _recv(ws):
    m = ws.receive()
    if m.get("bytes") is not None:
        return decode_frame(m["bytes"])[0], m["bytes"]
    return json.loads(m["text"]), None


def _until(ws, pred, limit=200):
    for _ in range(limit):
        h, raw = _recv(ws)
        if pred(h):
            return h, raw
    raise AssertionError("message never arrived")


def test_connect_gets_hello_weights_and_trace(client):
    with client.websocket_connect("/ws") as ws:
        hello, _ = _until(ws, lambda h: h["type"] == "hello")
        assert hello["config"]["n_layer"] == 2 and len(hello["vocab"]) == hello["config"]["vocab_size"]
        w, _ = _until(ws, lambda h: h["type"] == "weights")
        assert "L0.W_q" in w["weights"]
        h, raw = _until(ws, lambda h: h["type"] == "trace")
        t = decode_trace(raw)
        assert t.stages[-1].meta.id == "loss" and t.loss is not None


def test_run_returns_prompt_trace(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        ws.send_text(json.dumps({"type": "run", "prompt": "The cat sat"}))
        h, raw = _until(ws, lambda h: h["type"] == "trace" and h.get("prompt") == "The cat sat")
        assert "".join(decode_trace(raw).text) == "The␣cat␣sat"


def test_training_is_serialised(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        ws.send_text(json.dumps({"type": "train", "action": "play"}))
        steps = []
        for _ in range(3):
            ws.send_text(json.dumps({"type": "run", "prompt": "Hi"}))
            h, _ = _until(ws, lambda h: h["type"] == "trace")
            steps.append(h["step"])
        ws.send_text(json.dumps({"type": "train", "action": "pause"}))
        _until(ws, lambda h: h["type"] == "status" and not h["playing"])
        assert steps == sorted(steps) and steps[-1] > 0


def test_single_step_and_reset(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        ws.send_text(json.dumps({"type": "train", "action": "step"}))
        m, _ = _until(ws, lambda h: h["type"] == "metrics")
        assert m["step"] == 1 and len(m["points"]) == 1
        ws.send_text(json.dumps({"type": "train", "action": "reset"}))
        _until(ws, lambda h: h["type"] == "reset")
        s, _ = _until(ws, lambda h: h["type"] == "status")
        assert s["step"] == 0


def test_sample_and_bad_command(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "sample", "prompt": "Once", "n": 5, "temperature": 0.8, "seed": 1}))
        s, _ = _until(ws, lambda h: h["type"] == "sample")
        assert len(s["tokens"]) == 5 and abs(sum(s["probs"]) - 1) < 1e-4
        ws.send_text(json.dumps({"type": "nope"}))
        e, _ = _until(ws, lambda h: h["type"] == "error")
        assert "nope" in e["message"]


def test_step_explain(client):
    from core.trace_io import decode_frame
    import numpy as np
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        prompt = "Once upon a time there was a cat."
        for _ in range(3):   # repeated steps on one prompt must lower its loss
            ws.send_text(json.dumps({"type": "stepExplain", "prompt": prompt}))
            h, raw = _until(ws, lambda h: h["type"] == "stepExplain")
            assert h["lossAfter"] < h["lossBefore"]
            assert sum(h["pAfter"]) > sum(h["pBefore"])
        _, data = decode_frame(raw)
        ref = h["delta"]["L0.W_up"]
        delta = np.frombuffer(data, "<f4", int(np.prod(ref["shape"])), ref["offset"])
        assert np.abs(delta).max() > 0
        assert set(h["grad"]) == set(h["before"])


def test_interventions_and_baseline_tag(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        ablate = [{"kind": "block", "layer": 0, "part": "attn"}]
        ws.send_text(json.dumps({"type": "run", "prompt": "The dog ran", "interventions": ablate}))
        ws.send_text(json.dumps({"type": "run", "prompt": "The dog ran", "tag": "baseline"}))
        got = {}
        while len(got) < 2:
            h, raw = _until(ws, lambda h: h["type"] == "trace" and h.get("prompt") == "The dog ran")
            got[h.get("tag", "main")] = (h, decode_trace(raw))
        assert got["main"][0]["interventions"] == ablate
        assert (got["main"][1].stage("L0.attn.out").output == 0).all()
        assert (got["baseline"][1].stage("L0.attn.out").output != 0).any()


def test_replay_keeps_unsaved_progress(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        for _ in range(5):   # steps 1-5; the schedule saves some of these automatically
            ws.send_text(json.dumps({"type": "train", "action": "step"}))
        _until(ws, lambda h: h["type"] == "status" and h["step"] == 5)
        ws.send_text(json.dumps({"type": "loadCheckpoint", "step": 1}))
        s, _ = _until(ws, lambda h: h["type"] == "status" and h["step"] == 1)
        assert 5 in s["checkpoints"] and 1 in s["checkpoints"]
        ws.send_text(json.dumps({"type": "loadCheckpoint", "step": 5}))
        _until(ws, lambda h: h["type"] == "status" and h["step"] == 5)


def test_manual_backward_command(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manualBackward", "prompt": "Once upon a time"}))
        m, _ = _until(ws, lambda h: h["type"] == "manualBackward")
        assert m["worstRelative"] < 1e-4 and "L0.attn.H0.weights" in m["stages"]
