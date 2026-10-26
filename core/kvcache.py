"""Incremental generation with a KV cache, traced.

A causal model never lets a position see later ones, so appending a token changes nothing
at earlier positions: every earlier row of every stage is the same as before. Generation
can therefore keep each layer's keys and values (the KV cache) and run only the new
position. The trace is assembled from the cached rows plus the new one, and records how
much was reused.

The cache only helps while the window grows. Once the text is longer than the window, the
oldest token drops out and every position shifts by one, so (with learned absolute position
embeddings) every cached key and value is stale and everything must be recomputed: one
reason context length is a hard limit.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GPT, KV
from .stages import describe
from .tracer import StageRecord, Trace

ATTENTION_MATRICES = ("attn_scores", "mask", "softmax")   # [queries, keys] stages


@dataclass
class KVCache:
    tokens: list[int] = field(default_factory=list)
    weights_version: int = -1
    past: list[KV] = field(default_factory=list)
    outputs: dict[str, np.ndarray] = field(default_factory=dict)


def multiply_adds(model: GPT, start: int, end: int) -> int:
    """Multiply-adds to compute positions start..end-1 (a full pass: 0..T)."""
    c = model.cfg
    per_position = c.n_layer * (4 * c.d_model ** 2 + 2 * c.d_model * c.d_mlp) + c.d_model * c.vocab_size
    # Attention: position p scores against p + 1 keys and mixes p + 1 values, in every layer.
    attention = sum(2 * (p + 1) * c.d_model * c.n_layer for p in range(start, end))
    return (end - start) * per_position + attention


def why_not_reuse(cache: KVCache, tokens: list[int], weights_version: int, block: int) -> str | None:
    """None if the cache applies to `tokens`; otherwise the reason it cannot."""
    n = len(cache.tokens)
    if not n:
        return "no cache yet: the first pass computes every position"
    if cache.weights_version != weights_version:
        return "the weights changed, so every cached key and value is stale"
    if len(tokens) == block and n == block and tokens[:-1] == cache.tokens[1:]:
        return "the window slid: every position moved, so every cached key and value is stale"
    if len(tokens) <= n or tokens[:n] != cache.tokens:
        return "the prompt changed, so the cache no longer matches"
    return None


@torch.no_grad()
def trace_cached(model: GPT, cache: KVCache, tokens: list[int], token_text: list[str], step: int,
                 weights_version: int) -> tuple[Trace, dict]:
    """Trace `tokens`, reusing the cache where it applies, and update the cache."""
    block = model.cfg.block_size
    tokens, token_text = tokens[-block:], token_text[-block:]
    reason = why_not_reuse(cache, tokens, weights_version, block)
    t0 = 0 if reason else len(cache.tokens)
    past = None if reason else cache.past
    device = next(model.parameters()).device

    live: dict[str, torch.Tensor] = {}

    def record(stage_id: str, x: torch.Tensor) -> torch.Tensor:
        live[stage_id] = x
        return x

    was_training = model.training
    model.eval()
    with model.tapped(record):
        model(torch.tensor([tokens[t0:]], device=device), past=past, keep_kv=True)
    model.train(was_training)

    T = len(tokens)
    outputs: dict[str, np.ndarray] = {}
    for stage_id, x in live.items():
        new = x[0].float().cpu().numpy()
        meta = describe(stage_id, model.cfg)
        if t0 == 0:
            outputs[stage_id] = new
        elif meta.kind in ATTENTION_MATRICES:
            # Old query rows are unchanged; their entries for the new keys are masked (so −∞
            # before softmax, 0 after), and the raw scores there were never computed (NaN).
            full = np.full((T, T), {"attn_scores": np.nan, "mask": -np.inf, "softmax": 0.0}[meta.kind], np.float32)
            full[:t0, :t0] = cache.outputs[stage_id]
            full[t0:, :] = new
            outputs[stage_id] = full
        else:
            outputs[stage_id] = np.concatenate([cache.outputs[stage_id], new], axis=0)

    # The loss over the prompt's own next tokens, from the assembled logits.
    targets = tokens[1:] + [-1]
    logits = outputs["logits"].astype(np.float64)
    logp = logits - logits.max(-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
    kept = [i for i, y in enumerate(targets) if y >= 0]
    loss = float(-np.mean([logp[i, targets[i]] for i in kept])) if kept else None
    if loss is not None:
        outputs["loss"] = np.array(loss, dtype=np.float32)

    cache.tokens, cache.weights_version, cache.outputs = list(tokens), weights_version, outputs
    cache.past = model.kv or []
    model.kv = None

    stages = [StageRecord(describe(stage_id, model.cfg), out) for stage_id, out in outputs.items()]
    info = {"reused": t0, "computed": T - t0, "reason": reason,
            "multiplyAdds": multiply_adds(model, t0, T), "multiplyAddsFull": multiply_adds(model, 0, T)}
    return Trace(step=step, tokens=tokens, text=token_text, stages=stages, loss=loss, targets=targets), info
