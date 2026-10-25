import numpy as np
import torch

from core.model import GPT
from core.stages import describe, stage_order
from core.trace_io import decode_trace, encode_trace
from core.tracer import trace

from .conftest import VOCAB


def _prompt(n=100, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(VOCAB, (n,), generator=g).tolist()
    return ids, [str(i) for i in ids]


def test_stage_order_matches_model(default_model: GPT):
    ids, text = _prompt()
    t = trace(default_model, ids, text)
    assert [s.meta.id for s in t.stages] == stage_order(default_model.cfg)
    # every input named by the registry is a real stage that ran earlier
    seen = set()
    for s in t.stages:
        assert set(s.meta.inputs) <= seen, s.meta.id
        seen.add(s.meta.id)


def test_traced_values_match_untraced(default_model: GPT):
    ids, text = _prompt()
    t = trace(default_model, ids, text, with_grad=False)
    with torch.no_grad():
        logits, _ = default_model(torch.tensor([ids]))
    np.testing.assert_array_equal(t.stage("logits").output, logits[0].numpy())


def test_weight_grads_match_plain_backward(default_model: GPT):
    ids, text = _prompt()
    t = trace(default_model, ids, text, next_token=3)
    default_model.zero_grad()
    x = torch.tensor([ids])
    _, loss = default_model(x, torch.tensor([ids[1:] + [3]]))
    loss.backward()
    assert abs(loss.item() - t.loss) < 1e-6
    for name, p in default_model.named_weights().items():
        np.testing.assert_allclose(t.weight_grads[name], p.grad.numpy(), rtol=1e-5, atol=1e-7)


def test_stage_grads_match_autograd(default_model: GPT):
    ids, text = _prompt()
    t = trace(default_model, ids, text)
    live = {}

    def keep(stage_id, x):
        live[stage_id] = x
        return x

    with default_model.tapped(keep):
        _, loss = default_model(torch.tensor([ids]), torch.tensor([ids[1:] + [-1]]))
    names = ["embed.sum", "L1.attn.H2.weights", "L2.mlp.gelu", "L3.resid_mlp", "logits"]
    grads = torch.autograd.grad(loss, [live[n] for n in names])
    for n, g in zip(names, grads):
        np.testing.assert_allclose(t.stage(n).grad, g[0].numpy(), rtol=1e-5, atol=1e-8)


def test_residual_gradient_highway(default_model: GPT):
    # At x' = x + f(x), dL/dx = dL/dx' + (path through f): the resid input to block 1
    # receives the downstream gradient unchanged plus what flows back through ln1.
    ids, text = _prompt()
    t = trace(default_model, ids, text)
    g_after = t.stage("L1.resid_attn").grad
    g_before = t.stage("L0.resid_mlp").grad
    assert np.linalg.norm(g_before - g_after) < np.linalg.norm(g_after) * 5   # same order of size
    assert not np.allclose(g_before, g_after)


def test_logit_gradient_is_p_minus_y(default_model: GPT):
    ids, text = _prompt(20)
    t = trace(default_model, ids, text, target_index=5)
    p = t.stage("probs").output[5]
    y = np.zeros_like(p)
    y[ids[6]] = 1
    np.testing.assert_allclose(t.stage("logits").grad[5], p - y, atol=1e-6)
    assert np.abs(t.stage("logits").grad[6]).max() == 0


def test_masked_cells(default_model: GPT):
    ids, text = _prompt(10)
    t = trace(default_model, ids, text)
    s = t.stage("L0.attn.H0.masked")
    assert np.isneginf(s.output[0, 1:]).all()
    assert not np.isnan(s.grad).any()
    assert (t.stage("L0.attn.H0.weights").output[0, 1:] == 0).all()


def test_no_grad_without_targets(default_model: GPT):
    t = trace(default_model, [1], ["a"])
    assert t.loss is None and all(s.grad is None for s in t.stages)


def test_trace_leaves_param_grads_alone(default_model: GPT):
    p = default_model.wte
    p.grad = torch.ones_like(p)
    ids, text = _prompt()
    trace(default_model, ids, text)
    assert (p.grad == 1).all()


def test_round_trip_and_size(default_model: GPT):
    ids, text = _prompt()
    t = trace(default_model, ids, text, step=7)
    buf = encode_trace(t)
    # ~4.5 MB of activations (PROJECT.md's 3 MB excludes per-head splits and pre-mask
    # scores), the same again in gradients, plus weight gradients
    assert 8e6 < len(buf) < 12e6
    t2 = decode_trace(buf)
    assert t2.step == 7 and t2.tokens == ids and t2.loss == t.loss and t2.targets == ids[1:] + [-1]
    for a, b in zip(t.stages, t2.stages):
        assert a.meta == b.meta
        np.testing.assert_array_equal(a.output, b.output)
        np.testing.assert_array_equal(a.grad, b.grad)
    np.testing.assert_array_equal(t2.weight_grads["L2.W_up"], t.weight_grads["L2.W_up"])


def test_describe_covers_tied():
    from core.config import PROFILES
    cfg = PROFILES["concepts"].with_vocab(4096)
    assert describe("logits", cfg).weights == ["wte"]
