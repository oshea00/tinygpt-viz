"""Model and training configurations.

The four profiles match the table in PROJECT.md. vocab_size is filled in from the
tokenizer at load time, so character profiles adapt to whatever charset the corpus has.
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
    tokenizer: str = "char"      # "char" | "bpe"
    tied: bool = False           # share embedding and unembedding

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def with_vocab(self, vocab_size: int) -> "ModelConfig":
        return replace(self, vocab_size=vocab_size)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    trace_every: int = 50
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 1337


PROFILES: dict[str, ModelConfig] = {
    "tiny": ModelConfig("tiny", n_layer=2, n_head=2, d_model=32, d_mlp=128),
    "default": ModelConfig("default", n_layer=4, n_head=4, d_model=64, d_mlp=256),
    "larger": ModelConfig("larger", n_layer=6, n_head=8, d_model=128, d_mlp=512),
    "concepts": ModelConfig("concepts", n_layer=6, n_head=8, d_model=256, d_mlp=1024,
                            tokenizer="bpe", tied=True),
}


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
