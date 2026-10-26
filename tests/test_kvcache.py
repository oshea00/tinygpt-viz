import numpy as np
import torch

from core.kvcache import KVCache, multiply_adds, trace_cached
from core.model import GPT
from core.tracer import trace

from .conftest import VOCAB


def _ids(n, seed=3):
    return torch.randint(VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_cached_forward_matches_a_full_pass(default_model: GPT):
    ids = _ids(20)
    full, _ = default_model(torch.tensor([ids]))
    default_model(torch.tensor([ids[:15]]), keep_kv=True)
    past = default_model.kv
    step, _ = default_model(torch.tensor([ids[15:]]), past=past)
    torch.testing.assert_close(step[0], full[0, 15:], rtol=1e-5, atol=1e-5)


def test_cached_trace_equals_full_trace(default_model: GPT):
    cache = KVCache()
    ids = _ids(30)
    text = [str(i) for i in ids]
    trace_cached(default_model, cache, ids[:29], text[:29], 0, weights_version=1)
    t, info = trace_cached(default_model, cache, ids, text, 0, weights_version=1)
    assert info["reused"] == 29 and info["computed"] == 1 and info["reason"] is None
    assert info["multiplyAdds"] < info["multiplyAddsFull"] / 10
    ref = trace(default_model, ids, text, with_grad=False)
    for s in ref.stages:
        mine = t.stage(s.meta.id).output
        if s.meta.kind == "attn_scores":
            # Old queries against the new key were never computed: NaN there, equal elsewhere.
            assert np.isnan(mine[:29, 29]).all()
            np.testing.assert_allclose(np.nan_to_num(mine), np.where(np.isnan(mine), 0, s.output), rtol=1e-4, atol=1e-5)
        else:
            np.testing.assert_allclose(mine, s.output, rtol=1e-4, atol=1e-5, err_msg=s.meta.id)


def test_cache_is_rebuilt_when_it_cannot_apply(default_model: GPT):
    cache = KVCache()
    ids = _ids(100)
    text = [str(i) for i in ids]
    _, info = trace_cached(default_model, cache, ids[:50], text[:50], 0, 1)
    assert "no cache" in info["reason"] and info["reused"] == 0
    _, info = trace_cached(default_model, cache, ids[:51], text[:51], 0, 2)
    assert "weights changed" in info["reason"]
    _, info = trace_cached(default_model, cache, ids[:40] + [5] * 12, text[:52], 0, 2)
    assert "prompt changed" in info["reason"]
    trace_cached(default_model, cache, ids, text, 0, 2)                       # fill the window
    longer = ids + [7]
    _, info = trace_cached(default_model, cache, longer, text + ["7"], 0, 2)   # one past the window
    assert "window slid" in info["reason"] and info["computed"] == 100


def test_multiply_adds_grow_with_position(default_model: GPT):
    assert multiply_adds(default_model, 50, 51) > multiply_adds(default_model, 0, 1)
    assert multiply_adds(default_model, 0, 10) == sum(multiply_adds(default_model, p, p + 1) for p in range(10))
