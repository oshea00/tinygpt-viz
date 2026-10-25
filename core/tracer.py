"""Record every stage's output, and its gradient, for one prompt.

The tracer installs a tap on the model that keeps each stage tensor, runs forward and
(optionally) backward, then copies values out. Parameter .grad values are saved and
restored around the trace, so tracing between training steps never disturbs training.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GPT
from .stages import StageMeta, describe


@dataclass
class StageRecord:
    meta: StageMeta
    output: np.ndarray
    grad: np.ndarray | None = None


@dataclass
class Trace:
    step: int
    tokens: list[int]
    text: list[str]
    stages: list[StageRecord]
    loss: float | None = None
    targets: list[int] = field(default_factory=list)   # next-token target per position, -1 for none
    weight_grads: dict[str, np.ndarray] = field(default_factory=dict)

    def stage(self, stage_id: str) -> StageRecord:
        return next(s for s in self.stages if s.meta.id == stage_id)


def _to_numpy(tensors: list[torch.Tensor]) -> list[np.ndarray]:
    """Copy many tensors to the CPU in one transfer: on a GPU, one copy per tensor would
    mean ~170 synchronisations per trace."""
    if not tensors:
        return []
    flat = torch.cat([t.detach().float().reshape(-1) for t in tensors]).cpu().numpy()
    out, i = [], 0
    for t in tensors:
        out.append(flat[i:i + t.numel()].reshape(tuple(t.shape)))
        i += t.numel()
    return out


def trace(model: GPT, tokens: list[int], token_text: list[str], step: int = 0,
          with_grad: bool = True, next_token: int | None = None,
          target_index: int | None = None) -> Trace:
    """Trace one forward (and backward) pass of `tokens` (at most block_size of them).

    Targets are the prompt's own next tokens; the last position has a target only if
    `next_token` is given. `target_index` restricts the loss to a single position.
    """
    tokens = tokens[-model.cfg.block_size:]
    token_text = token_text[-model.cfg.block_size:]
    device = next(model.parameters()).device
    idx = torch.tensor([tokens], device=device)
    targets = torch.tensor([tokens[1:] + [next_token if next_token is not None else -1]], device=device)
    if target_index is not None:
        keep = targets[0, target_index].item()
        targets.fill_(-1)
        targets[0, target_index] = keep
    has_target = bool((targets >= 0).any())
    with_grad = with_grad and has_target

    live: dict[str, torch.Tensor] = {}

    def record(stage_id: str, x: torch.Tensor) -> torch.Tensor:
        if with_grad and x.requires_grad:
            x.retain_grad()
        live[stage_id] = x
        return x

    saved_grads = {n: p.grad for n, p in model.named_parameters()}
    was_training = model.training
    model.eval()
    try:
        with torch.set_grad_enabled(with_grad), model.tapped(record):
            model.zero_grad(set_to_none=True)
            _, loss = model(idx, targets if has_target else None)
            if with_grad:
                loss.backward()
        named = [(n, p.grad) for n, p in model.named_weights().items() if with_grad and p.grad is not None]
        weight_grads = dict(zip([n for n, _ in named], _to_numpy([g for _, g in named])))
    finally:
        for n, p in model.named_parameters():
            p.grad = saved_grads[n]
        model.train(was_training)

    unbatch = lambda x: x[0] if x.dim() > 0 else x   # noqa: E731  (traces are batch size 1)
    outputs = _to_numpy([unbatch(x) for x in live.values()])
    with_grads = [x for x in live.values() if with_grad and x.grad is not None]
    grads = iter(_to_numpy([unbatch(x.grad) for x in with_grads]))
    stages = []
    for (stage_id, x), out in zip(live.items(), outputs):
        grad = None
        if with_grad and x.grad is not None:
            grad = next(grads)
        elif with_grad and x.requires_grad:
            grad = np.zeros(out.shape, dtype=np.float32)   # stage did not affect this loss
        stages.append(StageRecord(describe(stage_id, model.cfg), out, grad))
    return Trace(step=step, tokens=tokens, text=token_text, stages=stages,
                 loss=loss.item() if loss is not None else None, targets=targets[0].tolist(),
                 weight_grads=weight_grads)
