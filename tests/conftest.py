import pytest
import torch

from core.config import PROFILES
from core.model import GPT

VOCAB = 64


@pytest.fixture
def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(PROFILES["tiny"].with_vocab(VOCAB))


@pytest.fixture
def default_model() -> GPT:
    torch.manual_seed(0)
    return GPT(PROFILES["default"].with_vocab(VOCAB))
