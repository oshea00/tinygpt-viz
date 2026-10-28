"""Interventions: change one thing inside the forward pass and see what follows.

Each intervention targets one stage and rewrites its output on the way through the
model's tap. They apply only to traces requested with them; training never sees them.
Specs are the JSON dicts the UI sends:

    {"kind": "head",    "layer": 1, "head": 2, "mode": "zero" | "mean"}
    {"kind": "neuron",  "layer": 0, "index": 17, "mode": "zero" | "mean"}
    {"kind": "block",   "layer": 2, "part": "attn" | "mlp"}
    {"kind": "attend",  "layer": 1, "head": 0, "position": 5}
    {"kind": "set",     "stage": "L0.q", "row": 3, "col": 7, "value": 1.5}
    {"kind": "shuffle_pos", "seed": 0}
    {"kind": "steer",   "stage": "L2.resid_mlp", "vector": [...], "scale": 4.0}
    {"kind": "patch",   "stage": "L2.resid_mlp", "positions": [4], "values": [[...]]}

"mean" replaces each value with its average over the prompt's positions, keeping the
unit's typical output but removing what is specific to each position. "patch" overwrites
rows (positions) of a stage with values taken from another run (activation patching).
"""

from typing import Callable

import torch

Rewrite = Callable[[torch.Tensor], torch.Tensor]


def _replace(x: torch.Tensor, mode: str, cols: slice | int = slice(None)) -> torch.Tensor:
    out = x.clone()
    if mode == "zero":
        out[..., cols] = 0
    elif mode == "mean":
        out[..., cols] = x[..., cols].mean(dim=1, keepdim=True).expand_as(x[..., cols])
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out


def _attend(position: int) -> Rewrite:
    def rewrite(w: torch.Tensor) -> torch.Tensor:
        # Each query attends only to `position`, or to itself if that is in its future.
        T = w.shape[-1]
        rows = torch.arange(T, device=w.device)
        target = torch.clamp(torch.full_like(rows, position), max=rows)
        return torch.nn.functional.one_hot(target, T).to(w.dtype).expand_as(w)
    return rewrite


def _set(row: int, col: int, value: float) -> Rewrite:
    def rewrite(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[0, row, col] = value
        return out
    return rewrite


def _patch(positions: list[int], values: list[list[float]]) -> Rewrite:
    def rewrite(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, positions] = torch.tensor(values, dtype=x.dtype, device=x.device)
        return out
    return rewrite


def _mulberry32(seed: int):
    """A small seeded generator that ui/src/engine/interventions.ts reproduces exactly."""
    a = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rand


def permutation(n: int, seed: int) -> list[int]:
    """A seeded Fisher–Yates permutation of 0..n-1, the same in the browser's engine."""
    perm, rand = list(range(n)), _mulberry32(seed)
    for i in range(n - 1, 0, -1):
        j = int(rand() * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _shuffle(seed: int) -> Rewrite:
    def rewrite(pos: torch.Tensor) -> torch.Tensor:
        return pos[:, torch.tensor(permutation(pos.shape[1], seed), device=pos.device)]
    return rewrite


def compile_spec(spec: dict) -> tuple[str, Rewrite]:
    """One spec → (stage id it rewrites, the rewrite)."""
    kind = spec["kind"]
    if kind == "head":
        return f"L{spec['layer']}.attn.H{spec['head']}.mix", lambda x: _replace(x, spec.get("mode", "zero"))
    if kind == "neuron":
        return f"L{spec['layer']}.mlp.gelu", lambda x: _replace(x, spec.get("mode", "zero"), int(spec["index"]))
    if kind == "block":
        stage = {"attn": "attn.out", "mlp": "mlp.down"}[spec["part"]]
        return f"L{spec['layer']}.{stage}", torch.zeros_like
    if kind == "attend":
        return f"L{spec['layer']}.attn.H{spec['head']}.weights", _attend(int(spec["position"]))
    if kind == "set":
        return spec["stage"], _set(int(spec["row"]), int(spec["col"]), float(spec["value"]))
    if kind == "steer":
        # Add a direction (e.g. a concept's mean difference) to the residual stream at every position.
        vec = torch.tensor(spec["vector"], dtype=torch.float32)
        scale = float(spec.get("scale", 1.0))
        return spec["stage"], lambda x: x + scale * vec.to(x.device)
    if kind == "patch":
        return spec["stage"], _patch([int(p) for p in spec["positions"]], spec["values"])
    if kind == "shuffle_pos":
        return "embed.pos", _shuffle(int(spec.get("seed", 0)))
    raise ValueError(f"unknown intervention {kind!r}")


def compile_all(specs: list[dict] | None) -> dict[str, list[Rewrite]]:
    by_stage: dict[str, list[Rewrite]] = {}
    for spec in specs or []:
        stage, rewrite = compile_spec(spec)
        by_stage.setdefault(stage, []).append(rewrite)
    return by_stage


def apply(by_stage: dict[str, list[Rewrite]], stage_id: str, x: torch.Tensor) -> torch.Tensor:
    for rewrite in by_stage.get(stage_id, ()):
        x = rewrite(x)
    return x
