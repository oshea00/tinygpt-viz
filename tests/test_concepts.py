import numpy as np
import torch

from core import concepts
from core.data import CharData
from core.model import GPT
from core.tracer import trace

TOK = CharData().tokenizer


def _model() -> GPT:
    from core.config import PROFILES
    torch.manual_seed(0)
    return GPT(PROFILES["default"].with_vocab(TOK.vocab_size))


def test_capture_reads_the_target_word():
    m = _model()
    prompts = [{"text": "The cat sat.", "targets": [{"start": 4, "end": 7, "word": "cat", "category": "animal"}]},
               {"text": "A big dog ran.", "targets": [{"start": 6, "end": 9, "word": "dog", "category": "animal"}]}]
    vec, items = concepts.capture(m, TOK, prompts, pool="last")
    assert vec.shape == (2, 9, 64) and [i["word"] for i in items] == ["cat", "dog"]
    ids = TOK.encode("The cat sat.")
    t = trace(m, ids, [TOK.token_text(i) for i in ids], with_grad=False)
    np.testing.assert_allclose(vec[0, 4], t.stage("L1.resid_mlp").output[6], rtol=1e-5, atol=1e-6)   # 't' of cat
    mean, _ = concepts.capture(m, TOK, prompts[:1], pool="mean")
    np.testing.assert_allclose(mean[0, 0], t.stage("embed.sum").output[4:7].mean(0), rtol=1e-5, atol=1e-6)


def test_steering_changes_generation():
    m = _model()
    plain = concepts.generate(m, TOK, "Once upon", 20, None)
    # Not a constant vector: layer norm subtracts each row's mean, so adding the same amount to
    # every dimension would change nothing downstream.
    direction = np.random.default_rng(0).normal(size=64) * 3.0
    steer = [{"kind": "steer", "stage": "L1.resid_mlp", "vector": direction.tolist(), "scale": 1.0}]
    assert concepts.generate(m, TOK, "Once upon", 20, steer) != plain
    assert concepts.generate(m, TOK, "Once upon", 20, None) == plain   # greedy is deterministic


def test_occlusion_covers_earlier_tokens_only():
    m = _model()
    ids = TOK.encode("The cat sat on the mat.")
    drops = concepts.occlusion(m, ids, 10)
    assert len(drops) == 11 and any(abs(d) > 0 for d in drops)


def test_neuron_top_is_sorted():
    m = _model()
    ex = concepts.neuron_top(m, TOK, CharData(), 1, 5, windows=64, k=5)
    assert len(ex) == 5 and all(a["value"] >= b["value"] for a, b in zip(ex, ex[1:]))
