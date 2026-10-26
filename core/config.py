"""Model and training configurations.

The profile matches the table in PROJECT.md. vocab_size is filled in from the tokenizer at
load time.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    d_mlp: int
    block_size: int = 100
    vocab_size: int = 0          # set from the tokenizer
    tied: bool = False           # share embedding and unembedding

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def with_vocab(self, vocab_size: int) -> "ModelConfig":
        return replace(self, vocab_size=vocab_size)


@dataclass
class TrainConfig:
    lr: float = 6e-4
    batch_size: int = 128
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    trace_every: int = 50
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337


# The BPE model reads a 4,096-token vocabulary, so most words are one token.
PROFILES: dict[str, ModelConfig] = {
    "bpe-small": ModelConfig("bpe-small", n_layer=6, n_head=8, d_model=256, d_mlp=1024, tied=True),
}
DEFAULT_PROFILE = "bpe-small"


def profile(name: str) -> ModelConfig:
    return PROFILES[name]


class DeviceError(RuntimeError):
    """A device the user asked for that this machine cannot provide."""


def pick_device(requested: str | None = None) -> str:
    """Auto-detect, or validate a device named on the command line.

    A requested device must actually exist. Silently falling back to CPU is the worst
    outcome: the run still starts, and only the wall clock reveals the GPU was never used.
    """
    import torch

    if requested:
        kind = torch.device(requested).type
        if kind == "cuda" and not torch.cuda.is_available():
            raise DeviceError(
                f"--device {requested}, but torch.cuda.is_available() is False. This torch is "
                f"{torch.__version__} (CUDA build: {torch.version.cuda or 'none'}) — either a "
                "CPU-only wheel, or no driver visible to this container/WSL instance."
            )
        if kind == "mps" and not torch.backends.mps.is_available():
            raise DeviceError(f"--device {requested}, but no Metal device is available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
