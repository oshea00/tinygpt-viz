import numpy as np
import pytest
import torch

from core.model import GPT
from core.tracer import trace

from .conftest import VOCAB


def _run(model: GPT, interventions=None, n=30):
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(VOCAB, (n,), generator=g).tolist()
    return trace(model, ids, [str(i) for i in ids], interventions=interventions)


def test_no_interventions_is_unchanged(default_model: GPT):
    a, b = _run(default_model), _run(default_model, [])
    np.testing.assert_array_equal(a.stage("logits").output, b.stage("logits").output)


def test_head_zero_and_mean(default_model: GPT):
    t = _run(default_model, [{"kind": "head", "layer": 1, "head": 2, "mode": "zero"}])
    assert (t.stage("L1.attn.H2.mix").output == 0).all()
    assert (t.stage("L1.attn.concat").output[:, 32:48] == 0).all()   # head 2's band of 16 columns
    m = _run(default_model, [{"kind": "head", "layer": 1, "head": 2, "mode": "mean"}]).stage("L1.attn.H2.mix").output
    base = _run(default_model).stage("L1.attn.H2.mix").output
    np.testing.assert_allclose(m, np.broadcast_to(base.mean(0), m.shape), rtol=1e-5, atol=1e-6)


def test_neuron_ablation_touches_one_column(default_model: GPT):
    base = _run(default_model).stage("L0.mlp.gelu").output
    t = _run(default_model, [{"kind": "neuron", "layer": 0, "index": 17, "mode": "zero"}]).stage("L0.mlp.gelu").output
    assert (t[:, 17] == 0).all()
    np.testing.assert_array_equal(np.delete(t, 17, 1), np.delete(base, 17, 1))


def test_block_off_passes_residual_through(default_model: GPT):
    t = _run(default_model, [{"kind": "block", "layer": 2, "part": "mlp"}])
    np.testing.assert_array_equal(t.stage("L2.resid_mlp").output, t.stage("L2.resid_attn").output)


def test_forced_attention(default_model: GPT):
    w = _run(default_model, [{"kind": "attend", "layer": 0, "head": 1, "position": 4}]).stage("L0.attn.H1.weights").output
    assert (w[10] == np.eye(30)[4]).all()        # later queries look only at position 4
    assert (w[2] == np.eye(30)[2]).all()         # earlier ones cannot see it, so they attend to themselves


def test_set_value_propagates_causally(default_model: GPT):
    base = _run(default_model)
    t = _run(default_model, [{"kind": "set", "stage": "L0.q", "row": 12, "col": 3, "value": 25.0}])
    assert t.stage("L0.q").output[12, 3] == 25.0
    before, after = base.stage("logits").output, t.stage("logits").output
    np.testing.assert_array_equal(after[:12], before[:12])   # earlier positions cannot be affected
    assert not np.allclose(after[12:], before[12:])


def test_shuffled_positions(default_model: GPT):
    base = _run(default_model).stage("embed.pos").output
    s = _run(default_model, [{"kind": "shuffle_pos", "seed": 0}]).stage("embed.pos").output
    assert sorted(map(tuple, s.round(6))) == sorted(map(tuple, base.round(6)))
    assert not np.array_equal(s, base)


def test_gradients_still_flow(default_model: GPT):
    t = _run(default_model, [{"kind": "head", "layer": 3, "head": 0, "mode": "zero"}])
    assert np.abs(t.stage("embed.sum").grad).sum() > 0
    assert (t.stage("L3.attn.H0.weights").grad == 0).all()   # cut off from the loss by the ablation


def test_unknown_intervention(default_model: GPT):
    with pytest.raises(ValueError):
        _run(default_model, [{"kind": "nope"}])


def test_patch_overwrites_rows(default_model: GPT):
    base = _run(default_model).stage("L0.resid_mlp").output
    values = [[1.0] * base.shape[1], [2.0] * base.shape[1]]
    t = _run(default_model, [{"kind": "patch", "stage": "L0.resid_mlp", "positions": [3, 7], "values": values}])
    out = t.stage("L0.resid_mlp").output
    np.testing.assert_array_equal(out[[3, 7]], np.array(values, dtype=np.float32))
    np.testing.assert_array_equal(np.delete(out, [3, 7], axis=0), np.delete(base, [3, 7], axis=0))
