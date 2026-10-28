import json

import pytest
from fastapi.testclient import TestClient

from core.config import TrainConfig
from core.server import UI_DIST, create_app
from core.trace_io import decode_frame, decode_trace
from core.train import Trainer

from .conftest import TINY


@pytest.fixture
def client(tmp_path, data):
    trainer = Trainer(TINY, TrainConfig(batch_size=8, trace_every=5, eval_every=1000),
                      data, "cpu", run_dir=tmp_path / "run")
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


def test_concept_commands(client):
    from core.trace_io import decode_frame
    with client.websocket_connect("/ws") as ws:
        prompts = [{"text": "The cat sat.", "targets": [{"start": 4, "end": 7, "word": "cat", "category": "animal"}]}]
        ws.send_text(json.dumps({"type": "capture", "prompts": prompts, "pool": "mean", "model": "current"}))
        h, raw = _until(ws, lambda h: h["type"] == "capture")
        assert h["items"][0]["word"] == "cat" and len(h["snapshots"]) == 5   # tiny: x0 + 2 layers × 2
        assert h["vectors"]["x"]["shape"] == [1, 5, 32]
        ws.send_text(json.dumps({"type": "capture", "prompts": prompts, "model": "untrained"}))
        e, _ = _until(ws, lambda h: h["type"] in ("capture", "error"))
        assert e["type"] == "error" and "step-0" in e["message"]     # this test run saved no checkpoint 0
        ws.send_text(json.dumps({"type": "generateWith", "prompt": "Once", "n": 5, "tag": "plain"}))
        g, _ = _until(ws, lambda h: h["type"] == "generated")
        assert g["tag"] == "plain" and g["text"]
        ws.send_text(json.dumps({"type": "occlude", "prompt": "The cat sat.", "position": 2}))   # The ␣cat ␣sat .
        o, _ = _until(ws, lambda h: h["type"] == "occlusion")
        assert len(o["drops"]) == 3
        ws.send_text(json.dumps({"type": "patch", "clean": "The cat sat.", "corrupt": "The dog sat.",
                                 "answer": " cat", "wrong": " dog", "what": "resid"}))
        p, raw = _until(ws, lambda h: h["type"] == "patch")
        assert p["tensors"]["grid"]["shape"] == [5, 4] and p["tensors"]["source"]["shape"] == [5, 4, 32]
        assert p["corruptText"] == ["The", "␣dog", "␣sat", "."] and p["answer"] == "␣cat"
        ws.send_text(json.dumps({"type": "patch", "clean": "The cat sat.", "corrupt": "The big dog sat.",
                                 "answer": " cat", "wrong": " dog", "what": "heads"}))
        e, _ = _until(ws, lambda h: h["type"] in ("patch", "error"))
        assert e["type"] == "error" and "same number of tokens" in e["message"]
        ws.send_text(json.dumps({"type": "lens", "prompt": "The cat"}))
        l, _ = _until(ws, lambda h: h["type"] == "lens")
        assert len(l["topIds"]) == 5 and len(l["topIds"][0]) == 2


def test_incremental_runs_reuse_the_cache(client):
    with client.websocket_connect("/ws") as ws:
        _until(ws, lambda h: h["type"] == "trace")
        for prompt in ("Once upon", "Once upon a", "Once upon a time"):     # one more token each time
            ws.send_text(json.dumps({"type": "run", "prompt": prompt, "withGrad": False, "incremental": True}))
            h, raw = _until(ws, lambda h: h["type"] == "trace" and h.get("prompt") == prompt)
        assert h["incremental"]["reused"] == 3 and h["incremental"]["computed"] == 1
        assert decode_trace(raw).stage("logits").output.shape[0] == 4


def test_guide_is_served_as_text(client):
    r = client.get("/docs/guide/01-getting-started.md")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "# Getting started" in r.text


def test_guide_files_are_revalidated(client):
    assert client.get("/docs/images/overview.png").headers["cache-control"] == "no-cache"


@pytest.mark.skipif(not UI_DIST.exists(), reason="UI not built")
def test_ui_page_is_revalidated(client):
    assert client.get("/").headers["cache-control"] == "no-cache"


def test_training_can_be_switched_off(tmp_path, data):
    trainer = Trainer(TINY, TrainConfig(batch_size=8), data, "cpu", run_dir=tmp_path / "run")
    with TestClient(create_app(trainer, training=False)) as c, c.websocket_connect("/ws") as ws:
        hello, _ = _until(ws, lambda h: h["type"] == "hello")
        assert hello["training"] is False
        for cmd in ({"type": "train", "action": "step"}, {"type": "stepExplain", "prompt": "Hi there"},
                    {"type": "loadCheckpoint", "step": 0}, {"type": "set", "lr": 0.1}):
            ws.send_text(json.dumps(cmd))
            e, _ = _until(ws, lambda h: h["type"] == "error")
            assert "--train" in e["message"]
        ws.send_text(json.dumps({"type": "run", "prompt": "Once"}))     # running the model still works
        h, raw = _until(ws, lambda h: h["type"] == "trace" and h.get("prompt") == "Once")
        assert h["step"] == 0


def test_belief_commands(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "beliefs", "n": 2, "cloze": " Everyone was very", "words": ["happy", "sad"]}))
        h, _ = _until(ws, lambda h: h["type"] in ("beliefs", "error"))
        assert h["type"] == "beliefs" and len(h["stories"]) == 2 and len(h["prefixes"]) == 4
        assert h["tensors"]["vectors"]["shape"] == [4, 5, 32] and h["tensors"]["readout"]["shape"] == [4, 2]
        steer = [{"kind": "steer", "stage": "L0.resid_mlp", "vector": [(-1.0) ** k for k in range(32)], "scale": 2.0}]
        ws.send_text(json.dumps({"type": "beliefSteer", "stories": h["stories"][:1], "cloze": h["cloze"],
                                 "words": h["words"], "interventions": steer, "tag": "sad"}))
        s, _ = _until(ws, lambda h: h["type"] == "beliefSteer")
        assert s["tag"] == "sad" and len(s["readout"]) == 2 and len(s["readout"][0]) == 2 and len(s["mass"]) == 2
