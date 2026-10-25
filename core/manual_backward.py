"""The backward pass written by hand, one small function per operation, then compared with
autograd stage by stage. For learners building their own GPT: these are the ten or so
derivatives you need, and the comparison shows that the trace's gradients are what they
should be.

Everything here is numpy in float64, working from the activations stored in a trace. Each
stage's gradient is accumulated from the stages that read it, walking the network in reverse.
"""

import math

import numpy as np

from .tracer import Trace

EPS = 1e-5


# -- one function per operation: (upstream gradient, saved values) → input gradients ------------

def cross_entropy_backward(logits: np.ndarray, targets: list[int]) -> np.ndarray:
    """L = mean over kept positions of −log softmax(z)[y].  ∂L/∂z = (p − onehot(y)) / n."""
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    kept = [i for i, t in enumerate(targets) if t >= 0]
    g = np.zeros_like(p)
    for i in kept:
        g[i] = p[i]
        g[i, targets[i]] -= 1
    return g / len(kept)


def linear_backward(g: np.ndarray, x: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = x W + b.  ∂x = g Wᵀ,  ∂W = xᵀ g,  ∂b = Σ_rows g."""
    return g @ W.T, x.T @ g, g.sum(0)


def layer_norm_backward(g: np.ndarray, x: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = γ x̂ + β with x̂ = (x − μ)/σ per row.
    ∂γ = Σ g x̂,  ∂β = Σ g,  ∂x = (ĝ − mean(ĝ) − x̂ mean(ĝ x̂)) / σ  where ĝ = g γ."""
    mu = x.mean(-1, keepdims=True)
    sigma = np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + EPS)
    xhat = (x - mu) / sigma
    gh = g * gamma
    dx = (gh - gh.mean(-1, keepdims=True) - xhat * (gh * xhat).mean(-1, keepdims=True)) / sigma
    return dx, (g * xhat).sum(0), g.sum(0)


def gelu_backward(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    """GELU(u) = ½u(1 + tanh(a)) with a = √(2/π)(u + 0.044715u³).
    ∂u = g · [½(1 + tanh a) + ½u(1 − tanh² a) √(2/π)(1 + 3·0.044715u²)]."""
    c = math.sqrt(2 / math.pi)
    t = np.tanh(c * (u + 0.044715 * u ** 3))
    return g * (0.5 * (1 + t) + 0.5 * u * (1 - t ** 2) * c * (1 + 3 * 0.044715 * u ** 2))


def scores_backward(g: np.ndarray, q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S = q kᵀ / √d.  ∂q = g k / √d,  ∂k = gᵀ q / √d."""
    s = math.sqrt(q.shape[-1])
    return g @ k / s, g.T @ q / s


def mask_backward(g: np.ndarray) -> np.ndarray:
    """Masked entries were replaced by a constant (−∞), so no gradient flows to them."""
    return np.tril(g)


def softmax_backward(g: np.ndarray, a: np.ndarray) -> np.ndarray:
    """a = softmax(s) per row.  ∂s = a ⊙ (g − Σ_j g_j a_j)."""
    return a * (g - (g * a).sum(-1, keepdims=True))


def mix_backward(g: np.ndarray, a: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O = A V.  ∂A = g Vᵀ,  ∂V = Aᵀ g."""
    return g @ v.T, a.T @ g


# -- walking the network in reverse --------------------------------------------------------------

def manual_backward(trace: Trace, weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Gradients for every stage and every weight, computed by hand from the trace's activations.
    Assumes the trace was made without interventions."""
    out = {s.meta.id: s.output.astype(np.float64) for s in trace.stages}
    meta = {s.meta.id: s.meta for s in trace.stages}
    W = {n: w.astype(np.float64) for n, w in weights.items()}
    grad: dict[str, np.ndarray] = {}
    wgrad = {n: np.zeros_like(w) for n, w in W.items()}

    def give(stage_id: str, g: np.ndarray) -> None:
        grad[stage_id] = grad.get(stage_id, 0) + g

    give("logits", cross_entropy_backward(out["logits"], trace.targets))
    for s in reversed(trace.stages):
        m = s.meta
        if m.id not in grad or m.kind in ("loss", "probs"):
            continue
        g = grad[m.id]
        ins = m.inputs
        if m.kind in ("linear", "logits"):
            w = W[m.weights[0]]
            tied = m.kind == "logits" and m.weights[0] == "wte"
            dx, dw, db = linear_backward(g, out[ins[0]], w.T if tied else w)
            give(ins[0], dx)
            wgrad[m.weights[0]] += dw.T if tied else dw
            if len(m.weights) > 1:
                wgrad[m.weights[1]] += db
        elif m.kind == "norm":
            dx, dg, db = layer_norm_backward(g, out[ins[0]], W[m.weights[0]])
            give(ins[0], dx)
            wgrad[m.weights[0]] += dg
            wgrad[m.weights[1]] += db
        elif m.kind == "activation":
            give(ins[0], gelu_backward(g, out[ins[0]]))
        elif m.kind == "add":
            give(ins[0], g)          # the residual "highway": passed through unchanged
            give(ins[1], g)
        elif m.kind == "split":
            dh = g.shape[-1]
            full = np.zeros_like(out[ins[0]])
            full[:, m.head * dh:(m.head + 1) * dh] = g
            give(ins[0], full)
        elif m.kind == "concat":
            dh = g.shape[-1] // len(ins)
            for h, head_id in enumerate(ins):
                give(head_id, g[:, h * dh:(h + 1) * dh])
        elif m.kind == "attn_scores":
            dq, dk = scores_backward(g, out[ins[0]], out[ins[1]])
            give(ins[0], dq)
            give(ins[1], dk)
        elif m.kind == "mask":
            give(ins[0], mask_backward(g))
        elif m.kind == "softmax":
            give(ins[0], softmax_backward(g, out[m.id]))
        elif m.kind == "attn_mix":
            da, dv = mix_backward(g, out[ins[0]], out[ins[1]])
            give(ins[0], da)
            give(ins[1], dv)
        elif m.kind == "embed":
            if m.weights[0] == "wte":
                np.add.at(wgrad["wte"], trace.tokens, g)   # only rows of tokens present get gradient
            else:
                wgrad["wpe"][:g.shape[0]] += g
    for stage_id, m in meta.items():
        if stage_id not in grad and m.kind not in ("loss", "probs"):
            grad[stage_id] = np.zeros_like(out[stage_id])
    return grad, wgrad


def compare(trace: Trace, weights: dict[str, np.ndarray]) -> dict:
    """Hand-written vs autograd, per stage and per weight: max |difference|, relative to the
    gradient's largest entry. Some gradients are zero in exact arithmetic (the key bias: adding
    one vector to every key shifts each row of scores by a constant, which softmax ignores), so
    both sides hold only rounding noise; those are measured against the trace's largest gradient
    instead, and flagged as near zero."""
    grad, wgrad = manual_backward(trace, weights)
    refs = [s.grad for s in trace.stages if s.grad is not None] + list(trace.weight_grads.values())
    largest = max((float(np.abs(r).max()) for r in refs if r.size), default=0.0)
    floor = 1e-6 * largest

    def stats(mine: np.ndarray, ref: np.ndarray) -> dict:
        diff = float(np.abs(mine - ref).max()) if mine.size else 0.0
        scale = float(np.abs(ref).max()) if ref.size else 0.0
        near_zero = scale < floor
        denom = largest if near_zero else scale   # noise is judged against the trace's gradient scale
        return {"maxAbsDiff": diff, "relative": diff / denom if denom > 0 else diff, "nearZero": near_zero}

    stages = {s.meta.id: stats(grad[s.meta.id], s.grad) for s in trace.stages if s.grad is not None and s.meta.id in grad}
    weight_stats = {n: stats(wgrad[n], g) for n, g in trace.weight_grads.items()}
    worst = max([v["relative"] for v in [*stages.values(), *weight_stats.values()]] or [0.0])
    return {"stages": stages, "weights": weight_stats, "worstRelative": worst}
