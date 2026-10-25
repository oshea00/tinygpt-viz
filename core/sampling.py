"""Turn a row of logits into a choice: greedy, temperature, top-k, top-p."""

from dataclasses import dataclass

import torch

from .model import softmax


@dataclass
class SampleSettings:
    temperature: float = 1.0      # 0 means greedy
    top_k: int | None = None
    top_p: float | None = None


def shape_distribution(logits: torch.Tensor, s: SampleSettings) -> torch.Tensor:
    """logits [V] → the probabilities actually sampled from, with excluded entries at 0."""
    if s.temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    probs = softmax(logits / s.temperature)
    if s.top_k is not None and s.top_k < probs.numel():
        cutoff = probs.topk(s.top_k).values[-1]
        probs = torch.where(probs >= cutoff, probs, 0.0)
    if s.top_p is not None and s.top_p < 1.0:
        sorted_p, order = probs.sort(descending=True)
        # Keep the smallest prefix whose mass reaches top_p (always at least one token).
        drop = sorted_p.cumsum(0) - sorted_p >= s.top_p
        sorted_p[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
    return probs / probs.sum()


def sample_next(logits: torch.Tensor, s: SampleSettings,
                generator: torch.Generator | None = None) -> tuple[int, torch.Tensor]:
    probs = shape_distribution(logits.float().cpu(), s)
    return int(torch.multinomial(probs, 1, generator=generator)), probs


@torch.no_grad()
def generate(model, ids: list[int], n: int, s: SampleSettings,
             generator: torch.Generator | None = None) -> list[int]:
    device = next(model.parameters()).device
    out = list(ids)
    for _ in range(n):
        window = out[-model.cfg.block_size:] or [0]
        logits, _ = model(torch.tensor([window], device=device))
        nxt, _ = sample_next(logits[0, -1], s, generator)
        out.append(nxt)
    return out[len(ids):]
