"""A GPT written with explicit operations so the code reads like the maths.

No nn.MultiheadAttention, nn.LayerNorm or fused attention: every matmul, mask and
softmax is visible. Weights use the maths convention x @ W with W shaped [d_in, d_out].

Every stage output passes through `self.tap(stage_id, x)`. With nothing installed it
returns x unchanged; the tracer installs a recorder there, and interventions (ablation,
value edits) install functions that return a modified tensor.
"""

import math
from contextlib import contextmanager
from typing import Callable, cast

import torch
import torch.nn as nn

from .config import ModelConfig

TapFn = Callable[[str, torch.Tensor], torch.Tensor]
KV = tuple[torch.Tensor, torch.Tensor]   # one layer's keys and values, [B, positions, d]


def layer_norm(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * g + b


def gelu(x: torch.Tensor) -> torch.Tensor:
    # The tanh approximation used by GPT-2.
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: torch.Tensor) -> torch.Tensor:
    # Subtracting the row max changes nothing mathematically but keeps exp() finite.
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)


def _normal(*shape: int, std: float = 0.02) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * std)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, m = cfg.d_model, cfg.d_mlp
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)   # residual writers start small
        self.ln1_g, self.ln1_b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.W_q, self.b_q = _normal(d, d), nn.Parameter(torch.zeros(d))
        self.W_k, self.b_k = _normal(d, d), nn.Parameter(torch.zeros(d))
        self.W_v, self.b_v = _normal(d, d), nn.Parameter(torch.zeros(d))
        self.W_o, self.b_o = _normal(d, d, std=proj_std), nn.Parameter(torch.zeros(d))
        self.ln2_g, self.ln2_b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.W_up, self.b_up = _normal(d, m), nn.Parameter(torch.zeros(m))
        self.W_down, self.b_down = _normal(m, d, std=proj_std), nn.Parameter(torch.zeros(d))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "set vocab_size from the tokenizer"
        self.cfg = cfg
        V, T, d = cfg.vocab_size, cfg.block_size, cfg.d_model
        self.wte = _normal(V, d)
        self.wpe = _normal(T, d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f_g, self.ln_f_b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.W_unembed = None if cfg.tied else _normal(d, V)
        self.mask: torch.Tensor
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.bool)), persistent=False)
        self._tap: TapFn | None = None
        self.kv: list[KV] | None = None

    # -- taps ---------------------------------------------------------------

    def tap(self, stage_id: str, x: torch.Tensor) -> torch.Tensor:
        return self._tap(stage_id, x) if self._tap else x

    @contextmanager
    def tapped(self, fn: TapFn):
        prev, self._tap = self._tap, fn
        try:
            yield
        finally:
            self._tap = prev

    # -- forward ------------------------------------------------------------

    def attention(self, l: int, blk: Block, x: torch.Tensor, past: KV | None = None) -> torch.Tensor:
        """Attention for the positions in x. With `past` (the KV cache), x holds only the new
        positions: their keys and values are appended to the cached ones, and each new query
        attends to every key up to its own position."""
        T, H, dh = x.shape[1], self.cfg.n_head, self.cfg.d_head
        t0 = past[0].shape[1] if past is not None else 0
        q = self.tap(f"L{l}.q", x @ blk.W_q + blk.b_q)
        k_new = self.tap(f"L{l}.k", x @ blk.W_k + blk.b_k)
        v_new = self.tap(f"L{l}.v", x @ blk.W_v + blk.b_v)
        k = torch.cat([past[0], k_new], dim=1) if past is not None else k_new
        v = torch.cat([past[1], v_new], dim=1) if past is not None else v_new
        if self.kv is not None:
            self.kv.append((k.detach(), v.detach()))
        mask = self.mask[t0:t0 + T, :t0 + T]        # query rows t0.., key columns 0..
        heads = []
        for h in range(H):
            cols = slice(h * dh, (h + 1) * dh)       # head h owns this band of columns
            p = f"L{l}.attn.H{h}"
            qh = self.tap(f"{p}.q", q[..., cols])
            # The split stages hold the new rows; cached rows (if any) are put in front of them.
            kh = self.tap(f"{p}.k", k_new[..., cols])
            vh = self.tap(f"{p}.v", v_new[..., cols])
            if past is not None:
                kh = torch.cat([past[0][..., cols], kh], dim=1)
                vh = torch.cat([past[1][..., cols], vh], dim=1)
            scores = self.tap(f"{p}.scores", qh @ kh.transpose(-2, -1) / math.sqrt(dh))
            masked = self.tap(f"{p}.masked", scores.masked_fill(~mask, float("-inf")))
            weights = self.tap(f"{p}.weights", softmax(masked))
            heads.append(self.tap(f"{p}.mix", weights @ vh))
        concat = self.tap(f"L{l}.attn.concat", torch.cat(heads, dim=-1))
        return self.tap(f"L{l}.attn.out", concat @ blk.W_o + blk.b_o)

    def mlp(self, l: int, blk: Block, x: torch.Tensor) -> torch.Tensor:
        up = self.tap(f"L{l}.mlp.up", x @ blk.W_up + blk.b_up)
        act = self.tap(f"L{l}.mlp.gelu", gelu(up))
        return self.tap(f"L{l}.mlp.down", act @ blk.W_down + blk.b_down)

    def unembed_matrix(self) -> torch.Tensor:
        return self.wte.T if self.W_unembed is None else self.W_unembed

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                past: list[KV] | None = None, keep_kv: bool = False):
        """idx [B, T] token ids → logits [B, T, V]; loss is mean cross-entropy if targets given.

        targets may contain -1 to ignore a position (used to train on one character).
        past: the KV cache, cached keys and values per layer for the positions before idx; idx
        then continues from there. keep_kv: leave every layer's keys and values (cached plus
        new) in self.kv, to cache for the next step.
        """
        B, T = idx.shape
        t0 = past[0][0].shape[1] if past else 0
        assert t0 + T <= self.cfg.block_size
        self.kv = [] if keep_kv else None
        tok = self.tap("embed.tok", self.wte[idx])
        pos = self.tap("embed.pos", self.wpe[t0:t0 + T].expand(B, T, -1))
        x = self.tap("embed.sum", tok + pos)
        for l, blk in enumerate(cast(list[Block], self.blocks)):
            x = self.tap(f"L{l}.resid_attn", x + self.attention(l, blk, self.tap(f"L{l}.ln1", layer_norm(x, blk.ln1_g, blk.ln1_b)),
                                                                 past[l] if past else None))
            x = self.tap(f"L{l}.resid_mlp", x + self.mlp(l, blk, self.tap(f"L{l}.ln2", layer_norm(x, blk.ln2_g, blk.ln2_b))))
        x = self.tap("ln_f", layer_norm(x, self.ln_f_g, self.ln_f_b))
        logits = self.tap("logits", x @ self.unembed_matrix())
        # probs is shown, not differentiated through: the loss below reads logits directly,
        # so the logit gradient is the clean p - y the lessons teach.
        self.tap("probs", softmax(logits.detach()))
        if targets is None:
            return logits, None
        # Cross-entropy written out: -log softmax(logits)[target], averaged over kept positions.
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        keep = targets >= 0
        nll = -log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        loss = self.tap("loss", (nll * keep).sum() / keep.sum())
        return logits, loss

    # -- introspection ------------------------------------------------------

    def named_weights(self) -> dict[str, torch.Tensor]:
        """Parameters under display names: 'wte', 'L0.W_q', 'ln_f_g', ..."""
        return {n.replace("blocks.", "L"): p for n, p in self.named_parameters()}

    def param_groups_count(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, p in self.named_weights().items():
            short = name.split(".")[-1]
            g = ("embedding" if name in ("wte", "wpe") else
                 "unembedding" if name == "W_unembed" else
                 "attention" if short[:3] in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o") else
                 "mlp" if "up" in short or "down" in short else "layer norm")
            groups[g] = groups.get(g, 0) + p.numel()
        return groups

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
