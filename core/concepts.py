"""Server side of the concept explorer: capture word representations from prompt sets,
generate with and without steering, occlusion attribution, and neuron search. The
analyses themselves (similarity, projections, probes, directions) run in the browser on
the captured vectors."""

import numpy as np
import torch

from . import interventions as iv
from .model import GPT
from .sampling import SampleSettings, sample_next
from .tokenizers import Tokenizer


def snapshot_ids(model: GPT) -> list[str]:
    ids = ["embed.sum"]
    for l in range(model.cfg.n_layer):
        ids += [f"L{l}.resid_attn", f"L{l}.resid_mlp"]
    return ids


@torch.no_grad()
def capture(model: GPT, tok: Tokenizer, prompts: list[dict], pool: str = "last") -> tuple[np.ndarray, list[dict]]:
    """Residual vectors of every tagged target word at every snapshot.

    prompts: [{"text": str, "targets": [{"start", "end", "word", "category"}]}], with character
    spans. A word split into several tokens is read at its last token (pool="last") or
    averaged over them (pool="mean"). Returns vectors [items, snapshots, d] and item records."""
    device = next(model.parameters()).device
    snaps = snapshot_ids(model)
    T = model.cfg.block_size
    encoded = []
    for p in prompts:
        ids, offsets = tok.encode_with_offsets(p["text"])
        encoded.append((ids[:T], offsets[:T], p))
    vectors, items = [], []
    model.eval()
    for start in range(0, len(encoded), 64):
        chunk = encoded[start:start + 64]
        width = max(len(e[0]) for e in chunk) or 1
        # Right padding is harmless: with a causal mask, no real position can see it.
        batch = torch.tensor([ids + [0] * (width - len(ids)) for ids, _, _ in chunk], device=device)
        got: dict[str, torch.Tensor] = {}

        def keep(stage_id: str, x: torch.Tensor) -> torch.Tensor:
            if stage_id in snaps:
                got[stage_id] = x
            return x

        with model.tapped(keep):
            model(batch)
        stacked = torch.stack([got[s] for s in snaps], dim=2).float().cpu().numpy()   # [B, T, S, d]
        for b, (ids, offsets, p) in enumerate(chunk):
            for target in p["targets"]:
                toks = [i for i, (s, e) in enumerate(offsets) if s < target["end"] and e > target["start"]]
                if not toks:
                    continue
                v = stacked[b, toks[-1]] if pool == "last" else stacked[b, toks].mean(0)
                vectors.append(v)
                items.append({"word": target["word"], "category": target["category"], "prompt": p["text"],
                              "tokens": [tok.token_text(ids[i]) for i in toks], "position": toks[-1]})
    d = model.cfg.d_model
    arr = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, len(snaps), d), np.float32)
    return arr, items


@torch.no_grad()
def generate(model: GPT, tok: Tokenizer, prompt: str, n: int, interventions: list[dict] | None,
             temperature: float = 0.0, seed: int = 0) -> str:
    """Continue a prompt, with interventions (such as steering) applied at every step."""
    device = next(model.parameters()).device
    rewrites = iv.compile_all(interventions)
    ids = tok.encode(prompt) or [0]
    g = torch.Generator().manual_seed(seed)
    out: list[int] = []
    model.eval()
    with model.tapped(lambda stage_id, x: iv.apply(rewrites, stage_id, x)):
        for _ in range(n):
            window = (ids + out)[-model.cfg.block_size:]
            logits, _ = model(torch.tensor([window], device=device))
            nxt, _ = sample_next(logits[0, -1], SampleSettings(temperature=temperature), g)
            out.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def occlusion(model: GPT, ids: list[int], target_pos: int) -> list[float]:
    """For predicting the token after position `target_pos`: how much log p(actual) drops when
    each earlier token's embedding is zeroed (its position is kept). One batched forward."""
    ids = ids[-model.cfg.block_size:]
    device = next(model.parameters()).device
    t = target_pos
    n = t + 1
    batch = torch.tensor([ids[:n]] * (n + 1), device=device)   # row j zeroes token j; last row is untouched

    def zero_one(stage_id: str, x: torch.Tensor) -> torch.Tensor:
        if stage_id != "embed.tok":
            return x
        x = x.clone()
        idx = torch.arange(n, device=device)
        x[idx, idx] = 0
        return x

    model.eval()
    with model.tapped(zero_one):
        logits, _ = model(batch)
    logp = torch.log_softmax(logits[:, t], dim=-1)[:, ids[t + 1]]
    return (logp[n] - logp[:n]).tolist()   # positive: this token helped the prediction


@torch.no_grad()
def neuron_top(model: GPT, tok: Tokenizer, data, layer: int, neuron: int, windows: int = 512, k: int = 12) -> list[dict]:
    """The validation contexts where one MLP neuron fires most strongly."""
    device = next(model.parameters()).device
    T = model.cfg.block_size
    split = data.splits["val"]
    g = torch.Generator().manual_seed(0)          # the same windows every time
    starts = torch.randint(len(split) - T - 1, (windows,), generator=g).tolist()
    stage = f"L{layer}.mlp.gelu"
    best: list[tuple[float, int, int]] = []
    model.eval()
    for i in range(0, windows, 64):
        rows = starts[i:i + 64]
        batch = torch.tensor(np.stack([split[s:s + T].astype(np.int64) for s in rows]), device=device)
        acts: dict[str, torch.Tensor] = {}

        def keep(stage_id: str, x: torch.Tensor) -> torch.Tensor:
            if stage_id == stage:
                acts["a"] = x[..., neuron]
            return x

        with model.tapped(keep):
            model(batch)
        a = acts["a"].float().cpu()
        vals, pos = a.max(dim=1)
        best += [(float(v), rows[j], int(p)) for j, (v, p) in enumerate(zip(vals, pos))]
    best.sort(reverse=True)
    out = []
    for value, s, p in best[:k]:
        ctx = split[s + max(0, p - 12):s + p + 4].astype(np.int64).tolist()
        hi = min(p, 12)
        out.append({"value": value, "before": tok.decode(ctx[:hi]), "token": tok.token_text(ctx[hi]), "after": tok.decode(ctx[hi + 1:])})
    return out
