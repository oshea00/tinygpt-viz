import numpy as np
import torch

from core import concepts
from core.data import BPE_DIR
from core.model import GPT
from core.tokenizers import BPETokenizer
from core.tracer import trace

from .conftest import SMALL

TOK = BPETokenizer.load(BPE_DIR / "tokenizer.json")


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL.with_vocab(TOK.vocab_size))


def test_capture_reads_the_target_word():
    m = _model()
    prompts = [{"text": "The cat sat.", "targets": [{"start": 4, "end": 7, "word": "cat", "category": "animal"}]},
               {"text": "A big dog ran.", "targets": [{"start": 6, "end": 9, "word": "dog", "category": "animal"}]}]
    vec, items = concepts.capture(m, TOK, prompts, pool="last")
    assert vec.shape == (2, 9, 64) and [i["word"] for i in items] == ["cat", "dog"]
    assert items[0]["tokens"] == ["␣cat"]
    ids = TOK.encode("The cat sat.")
    t = trace(m, ids, [TOK.token_text(i) for i in ids], with_grad=False)
    np.testing.assert_allclose(vec[0, 4], t.stage("L1.resid_mlp").output[1], rtol=1e-5, atol=1e-6)   # ␣cat


def test_capture_pools_a_split_word():
    m = _model()
    text = "A big xylophone rang."
    prompts = [{"text": text, "targets": [{"start": 6, "end": 15, "word": "xylophone", "category": "thing"}]}]
    ids, offsets = TOK.encode_with_offsets(text)
    toks = [i for i, (s, e) in enumerate(offsets) if s < 15 and e > 6]
    assert len(toks) > 1   # a rare word: several tokens
    t = trace(m, ids, [TOK.token_text(i) for i in ids], with_grad=False)
    last, items = concepts.capture(m, TOK, prompts, pool="last")
    assert items[0]["position"] == toks[-1]
    np.testing.assert_allclose(last[0, 0], t.stage("embed.sum").output[toks[-1]], rtol=1e-5, atol=1e-6)
    mean, _ = concepts.capture(m, TOK, prompts, pool="mean")
    np.testing.assert_allclose(mean[0, 0], t.stage("embed.sum").output[toks].mean(0), rtol=1e-5, atol=1e-6)


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
    drops = concepts.occlusion(m, ids, len(ids) - 2)    # the last position has no next token to predict
    assert len(drops) == len(ids) - 1 and any(abs(d) > 0 for d in drops)


def test_neuron_top_is_sorted(data):
    m = _model()
    ex = concepts.neuron_top(m, TOK, data, 1, 5, windows=64, k=5)
    assert len(ex) == 5 and all(a["value"] >= b["value"] for a, b in zip(ex, ex[1:]))
