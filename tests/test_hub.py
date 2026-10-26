import torch

from core import hub
from core.config import TrainConfig
from core.data import Data
from core.train import Trainer

from .conftest import TINY, make_corpus


def _trainer(run_dir, data):
    return Trainer(TINY, TrainConfig(batch_size=4), data, "cpu", run_dir=run_dir)


def test_stripped_checkpoint_has_no_optimizer_and_loads(tmp_path, data):
    t = _trainer(tmp_path / "run", data)
    t.step()
    src = t.save_checkpoint()
    dst = tmp_path / "shared" / src.name
    hub.strip_checkpoint(src, dst)
    state = torch.load(dst, weights_only=True)
    assert "opt" not in state and state["step"] == 1
    assert dst.stat().st_size < src.stat().st_size

    other = _trainer(tmp_path / "shared", data)
    other.load_checkpoint()
    assert other.step_num == 1
    for a, b in zip(t.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
    other.step()   # trains on from fresh optimizer moments


def test_val_only_corpus_runs_without_building(tmp_path, monkeypatch):
    make_corpus(tmp_path, splits=("val",))
    monkeypatch.setattr("core.data.build_bpe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    data = Data(tmp_path, need_train=False)
    assert set(data.splits) == {"val"}
    x, y = data.get_batch("val", 2, 8, "cpu")
    assert x.shape == (2, 8)
