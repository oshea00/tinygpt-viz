"""Binary frames for the WebSocket bridge.

    u32 little-endian header length (a multiple of 4)
    JSON header, space-padded to that length
    float32 data; header offsets are in bytes from the start of this section

Two frame types: "trace" (stage outputs and gradients) and "weights" (all parameters,
sent when the weights change, so traces need not repeat them).
"""

import json
import struct

import numpy as np

from .stages import StageMeta
from .tracer import StageRecord, Trace


class _Packer:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.size = 0

    def add(self, a: np.ndarray) -> dict:
        a = np.ascontiguousarray(a, dtype="<f4")
        ref = {"shape": list(a.shape), "offset": self.size}
        self.chunks.append(a.tobytes())
        self.size += a.nbytes
        return ref


def _frame(header: dict, packer: _Packer) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 4)
    return struct.pack("<I", len(raw)) + raw + b"".join(packer.chunks)


def encode_trace(t: Trace, extra: dict | None = None) -> bytes:
    pk = _Packer()
    stages = []
    for s in t.stages:
        out = pk.add(s.output)
        entry = {**s.meta.to_json(), "shape": out["shape"], "offset": out["offset"]}
        if s.grad is not None:
            entry["gradOffset"] = pk.add(s.grad)["offset"]
        stages.append(entry)
    header = {"type": "trace", "step": t.step, "loss": t.loss, "tokens": t.tokens, "text": t.text,
              "targets": t.targets, "stages": stages, "weightGrads": {n: pk.add(g) for n, g in t.weight_grads.items()},
              **(extra or {})}
    return _frame(header, pk)


def encode_tensors(frame_type: str, groups: dict[str, dict[str, np.ndarray]], extra: dict | None = None) -> bytes:
    """A frame of named tensor groups, e.g. {"weights": {"L0.W_q": ...}}."""
    pk = _Packer()
    header = {"type": frame_type, **{g: {n: pk.add(a) for n, a in ts.items()} for g, ts in groups.items()},
              **(extra or {})}
    return _frame(header, pk)


def encode_weights(weights: dict[str, np.ndarray], step: int, extra: dict | None = None) -> bytes:
    return encode_tensors("weights", {"weights": weights}, {"step": step, **(extra or {})})


def decode_frame(buf: bytes) -> tuple[dict, memoryview]:
    (n,) = struct.unpack_from("<I", buf, 0)
    return json.loads(buf[4:4 + n]), memoryview(buf)[4 + n:]


def _array(data: memoryview, ref: dict, shape: list[int] | None = None) -> np.ndarray:
    shape = ref["shape"] if shape is None else shape
    count = int(np.prod(shape)) if shape else 1
    return np.frombuffer(data, dtype="<f4", count=count, offset=ref["offset"]).reshape(shape)


def decode_tensors(buf: bytes, group: str) -> tuple[dict, dict[str, np.ndarray]]:
    """A tensor frame's header and one group's tensors (see encode_tensors)."""
    h, data = decode_frame(buf)
    return h, {n: _array(data, r) for n, r in h[group].items()}


def decode_trace(buf: bytes) -> Trace:
    h, data = decode_frame(buf)
    stages = []
    for s in h["stages"]:
        meta = StageMeta(**{k: s[k] for k in ("id", "kind", "layer", "head", "inputs", "weights",
                                              "formula", "explain", "backward") if k in s})
        grad = _array(data, {"offset": s["gradOffset"]}, s["shape"]) if "gradOffset" in s else None
        stages.append(StageRecord(meta, _array(data, s), grad))
    return Trace(step=h["step"], tokens=h["tokens"], text=h["text"], stages=stages, loss=h["loss"],
                 targets=h["targets"], weight_grads={n: _array(data, r) for n, r in h["weightGrads"].items()})
