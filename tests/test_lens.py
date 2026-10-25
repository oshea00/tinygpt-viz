import numpy as np
import torch

from core.lens import logit_lens
from core.model import GPT
from core.tracer import trace

from .conftest import VOCAB


def test_last_snapshot_is_the_models_prediction(default_model: GPT):
    ids = torch.randint(VOCAB, (30,), generator=torch.Generator().manual_seed(4)).tolist()
    lens = logit_lens(default_model, ids, k=3)
    assert lens["snapshots"][0] == "embed.sum" and lens["snapshots"][-1] == "L3.resid_mlp" and len(lens["snapshots"]) == 9
    probs = trace(default_model, ids, [str(i) for i in ids], with_grad=False).stage("probs").output
    top = np.argsort(-probs, axis=-1)[:, :3]
    assert (np.array(lens["topIds"][-1]) == top).all()
    np.testing.assert_allclose(lens["pActual"][-1][:-1], probs[np.arange(29), ids[1:]], rtol=1e-5)
    assert lens["pActual"][-1][-1] == -1
