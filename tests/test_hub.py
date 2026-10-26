import shutil

import numpy as np
import torch

from core import hub
from core.config import PROFILES, TrainConfig
from core.data import DATA_DIR, CharData
from core.train import Trainer


def _trainer(tmp_path):
    return Trainer(PROFILES["char-tiny"], TrainConfig(batch_size=4), CharData(), "cpu", run_dir=tmp_path)


def test_stripped_checkpoint_has_no_optimizer_and_loads(tmp_path):
    t = _trainer(tmp_path / "run")
    t.step()
    src = t.save_checkpoint()
    dst = tmp_path / "shared" / src.name
    hub.strip_checkpoint(src, dst)
    state = torch.load(dst, weights_only=True)
    assert "opt" not in state and state["step"] == 1
    assert dst.stat().st_size < src.stat().st_size

    other = _trainer(tmp_path / "shared")
    other.load_checkpoint()
    assert other.step_num == 1
    for a, b in zip(t.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
    other.step()   # trains on from fresh optimizer moments


def test_val_only_corpus_runs_without_building(tmp_path, monkeypatch):
    shutil.copy(DATA_DIR / "char_vocab.json", tmp_path)
    np.memmap(DATA_DIR / "val.bin", dtype=np.uint8, mode="r")[:10_000].tofile(tmp_path / "val.bin")
    monkeypatch.setattr("core.data.build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    data = CharData(tmp_path, need_train=False)
    assert set(data.splits) == {"val"}
    x, y = data.get_batch("val", 2, 8, "cpu")
    assert x.shape == (2, 8)
