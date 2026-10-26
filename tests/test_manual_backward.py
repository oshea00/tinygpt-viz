import numpy as np
import torch

from core.manual_backward import compare
from core.model import GPT
from core.tracer import trace

from .conftest import TINY, VOCAB


def _check(model: GPT, n: int, next_token: int | None = None, target_index: int | None = None):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(VOCAB, (n,), generator=g).tolist()
    t = trace(model, ids, [str(i) for i in ids], next_token=next_token, target_index=target_index)
    weights = {k: v.detach().numpy() for k, v in model.named_weights().items()}
    return compare(t, weights)


def _trained_like(model: GPT) -> GPT:
    # Larger, non-initial weights so every path (and GELU's curved region) carries real gradient.
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    return model


def test_every_stage_and_weight_matches_autograd(default_model: GPT):
    r = _check(_trained_like(default_model), 40, next_token=3)
    assert len(r["stages"]) > 150 and len(r["weights"]) == len(dict(default_model.named_weights()))
    bad = {k: v for k, v in {**r["stages"], **r["weights"]}.items() if v["relative"] > 1e-4}
    assert not bad, bad


def test_single_target_and_short_prompt(tiny_model: GPT):
    r = _check(_trained_like(tiny_model), 6, target_index=2)
    assert r["worstRelative"] < 1e-4


def test_tied_embeddings():
    torch.manual_seed(0)
    cfg = TINY.with_vocab(VOCAB)
    from dataclasses import replace
    model = _trained_like(GPT(replace(cfg, tied=True)))
    r = _check(model, 20)
    assert r["weights"]["wte"]["relative"] < 1e-4


def test_embedding_gradient_only_on_present_tokens(tiny_model: GPT):
    ids = [1, 2, 1, 3]
    t = trace(tiny_model, ids, list("abac"), next_token=2)
    rows = np.abs(t.weight_grads["wte"]).sum(1)
    assert set(np.nonzero(rows)[0]) == {1, 2, 3}


def test_key_bias_gradient_is_zero_in_theory(tiny_model: GPT):
    r = _check(_trained_like(tiny_model), 12)
    assert r["weights"]["L0.b_k"]["nearZero"] and not r["weights"]["L0.b_q"]["nearZero"]
