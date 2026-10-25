import math

import torch

from core.model import GPT, gelu, layer_norm, softmax
from core.sampling import SampleSettings, shape_distribution

from .conftest import VOCAB


def test_explicit_ops_match_torch():
    x = torch.randn(3, 7, 16)
    g, b = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(layer_norm(x, g, b), torch.nn.functional.layer_norm(x, (16,), g, b))
    torch.testing.assert_close(gelu(x), torch.nn.functional.gelu(x, approximate="tanh"))
    torch.testing.assert_close(softmax(x), torch.softmax(x, -1))


def test_param_count_default(default_model: GPT):
    # PROJECT.md: about 215k for the default config
    assert 200_000 < default_model.num_params() < 230_000
    groups = default_model.param_groups_count()
    assert groups["mlp"] / default_model.num_params() > 0.6   # "the MLPs hold two-thirds"


def test_stage_shapes(default_model: GPT):
    shapes = {}

    def rec(stage_id, x):
        shapes[stage_id] = tuple(x.shape)
        return x

    T = 100
    with default_model.tapped(rec):
        default_model(torch.randint(VOCAB, (1, T)), torch.randint(VOCAB, (1, T)))
    assert shapes["embed.sum"] == (1, T, 64)
    assert shapes["L0.q"] == (1, T, 64)
    assert shapes["L3.attn.H2.k"] == (1, T, 16)
    assert shapes["L1.attn.H0.weights"] == (1, T, T)
    assert shapes["L2.mlp.gelu"] == (1, T, 256)
    assert shapes["logits"] == (1, T, VOCAB)
    assert shapes["loss"] == ()
    # 3 input stages, 12 per block plus 7 per head, 4 output stages
    per_block = 12 + 7 * 4
    assert len(shapes) == 3 + 4 * per_block + 4


def test_causal(tiny_model: GPT):
    idx = torch.randint(VOCAB, (1, 50))
    a, _ = tiny_model(idx)
    idx2 = idx.clone()
    idx2[0, 30] = (idx2[0, 30] + 1) % VOCAB
    b, _ = tiny_model(idx2)
    torch.testing.assert_close(a[0, :30], b[0, :30])
    assert not torch.allclose(a[0, 30:], b[0, 30:])


def test_initial_loss_near_uniform(default_model: GPT):
    idx = torch.randint(VOCAB, (4, 100))
    _, loss = default_model(idx, torch.randint(VOCAB, (4, 100)))
    assert abs(loss.item() - math.log(VOCAB)) < 0.3


def test_ignored_targets(tiny_model: GPT):
    idx = torch.randint(VOCAB, (1, 20))
    y = torch.randint(VOCAB, (1, 20))
    one = torch.full_like(y, -1)
    one[0, 7] = y[0, 7]
    logits, loss = tiny_model(idx, one)
    expected = torch.nn.functional.cross_entropy(logits[0, 7:8], y[0, 7:8])
    torch.testing.assert_close(loss, expected)


def test_overfits_one_batch(tiny_model: GPT):
    idx = torch.randint(VOCAB, (2, 32))
    y = torch.randint(VOCAB, (2, 32))
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)
    loss = torch.tensor(float("inf"))
    for _ in range(300):
        _, loss = tiny_model(idx, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_sampling_filters():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    assert shape_distribution(logits, SampleSettings(temperature=0)).tolist() == [1, 0, 0, 0]
    p = shape_distribution(logits, SampleSettings(top_k=2))
    assert (p[2:] == 0).all() and abs(p.sum().item() - 1) < 1e-6
    p = shape_distribution(logits, SampleSettings(top_p=0.5))
    assert p[0] == 1.0
