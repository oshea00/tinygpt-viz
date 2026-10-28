import numpy as np
import pytest
import torch

from core import beliefs
from core.data import BPE_DIR
from core.model import GPT
from core.tokenizers import BPETokenizer
from core.tracer import trace

from .conftest import SMALL, STORIES

TOK = BPETokenizer.load(BPE_DIR / "tokenizer.json")
CLOZE, WORDS = beliefs.DEFAULT_CLOZE, beliefs.DEFAULT_WORDS


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL.with_vocab(TOK.vocab_size))


def test_split_sentences_spans_end_at_punctuation():
    text = 'Tom ran. "Hi!" said Sam.  Then they played'
    spans = beliefs.split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["Tom ran.", '"Hi!" said Sam.', "Then they played"]


def test_vectors_are_the_bare_prefix_and_readout_is_a_distribution():
    m = _model()
    r = beliefs.run(m, TOK, STORIES[:2], CLOZE, WORDS)
    assert [p["sentence"] for p in r["prefixes"]] == [0, 1, 0, 1]
    assert r["vectors"].shape == (4, 9, 64) and r["readout"].shape == (4, 6)
    np.testing.assert_allclose(r["readout"].sum(1), 1, rtol=1e-5)
    assert ((r["mass"] > 0) & (r["mass"] <= 1)).all()
    # The cloze comes after the read position, so the causal mask hides it: same as the bare prefix.
    p = r["prefixes"][1]
    ids = TOK.encode(STORIES[0][:p["end"]])
    t = trace(m, ids, [TOK.token_text(i) for i in ids], with_grad=False)
    np.testing.assert_allclose(r["vectors"][1, 4], t.stage("L1.resid_mlp").output[-1], rtol=1e-4, atol=1e-5)


def test_long_prefixes_slide_to_fit_the_window():
    m = _model()
    long = " ".join(["The dog ran to the big tree in the park."] * 30)
    r = beliefs.run(m, TOK, [long], CLOZE, WORDS)
    assert len(r["prefixes"]) == 30 and np.isfinite(r["readout"]).all()


def test_multi_token_word_is_refused():
    with pytest.raises(ValueError, match="single tokens"):
        beliefs.word_ids(TOK, ["happy", "xylophonic"])


def test_steering_changes_the_readout():
    m = _model()
    plain = beliefs.run(m, TOK, STORIES[:1], CLOZE, WORDS, with_vectors=False)
    assert "vectors" not in plain
    steer = [{"kind": "steer", "stage": "L1.resid_mlp", "vector": [(-1.0) ** k for k in range(64)], "scale": 5.0}]   # not constant: layer norm removes the mean
    steered = beliefs.run(m, TOK, STORIES[:1], CLOZE, WORDS, steer, with_vectors=False)
    assert np.abs(plain["readout"] - steered["readout"]).max() > 1e-3


def test_story_set_reads_whole_stories_from_the_stream():
    ids = [i for s in STORIES for i in [*TOK.encode(s), TOK.eos_id]]
    got = beliefs.story_set(TOK, np.array(ids, dtype=np.uint16), 3)
    assert got == STORIES[:3]
