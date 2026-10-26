import pytest
import torch

from core.config import PROFILES, DeviceError, TrainConfig, pick_device
from core.data import load_data
from core.preflight import device_info, fmt_duration, measure
from core.train import Trainer


@pytest.fixture
def trainer() -> Trainer:
    cfg = PROFILES["char-tiny"]
    return Trainer(cfg, TrainConfig(batch_size=4), load_data(cfg.tokenizer), "cpu")


def test_requested_device_must_exist():
    """A typo or a CPU-only wheel must fail loudly, not train on the CPU for hours."""
    if not torch.cuda.is_available():
        with pytest.raises(DeviceError, match="cuda"):
            pick_device("cuda")
    with pytest.raises(RuntimeError):
        pick_device("gpu")              # not a device torch knows at all


def test_auto_device_is_real():
    assert pick_device() in ("cpu", "cuda", "mps")


def test_measure_leaves_the_trainer_untouched(trainer: Trainer):
    """The probe runs real steps, so it must restore weights, optimiser state and the counter;
    otherwise --resume would silently train from probe-advanced weights."""
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    schedule = trainer.schedule
    measure(trainer, steps=100)
    assert trainer.step_num == 0
    assert trainer.schedule == schedule
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_measure_extrapolates_from_the_current_step(trainer: Trainer):
    est = measure(trainer, steps=100)
    assert est.step_seconds > 0
    assert est.total_seconds == pytest.approx(est.train_seconds + est.eval_seconds + est.sample_seconds)
    assert est.train_seconds == pytest.approx(est.step_seconds * 100)
    assert measure(trainer, steps=0).train_seconds == 0


def test_device_info_names_the_device():
    info = device_info("cpu")
    assert info.device == "cpu" and "CPU" in info.name


@pytest.mark.parametrize("seconds,expected", [(12.4, "12s"), (200, "3m"), (7200, "2.0h")])
def test_fmt_duration(seconds, expected):
    assert fmt_duration(seconds) == expected
