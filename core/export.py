"""Export a checkpoint for the static browser demo (inference only, no Python needed).

    uv run python -m core.export --config default            # latest checkpoint
    cd ui && npm run build:demo                              # → ui/dist-demo, a static site

Writes, under ui/demo-public/demo/: weights.bin (the same binary frame the live app
receives), reference.bin (a trace of the probe prompt, which supplies each stage's
description and lets the demo check itself), and meta.json (config and vocabulary).
"""

import argparse
import json
from pathlib import Path

import torch

from .config import PROFILES, TrainConfig
from .data import CharData
from .scheduler import DEFAULT_PROBE
from .trace_io import encode_trace, encode_weights
from .tracer import trace
from .train import Trainer

OUT = Path(__file__).resolve().parent.parent / "ui" / "demo-public" / "demo"


def export(config: str, step: int | None, out: Path = OUT, probe: str = DEFAULT_PROBE) -> Path:
    trainer = Trainer(PROFILES[config], TrainConfig(), CharData(), "cpu")
    trainer.load_checkpoint(step)
    model, tok = trainer.model, trainer.data.tokenizer
    out.mkdir(parents=True, exist_ok=True)
    weights = {n: p.detach().float().numpy() for n, p in model.named_weights().items()}
    (out / "weights.bin").write_bytes(encode_weights(weights, trainer.step_num, {"version": 1}))
    ids = tok.encode(probe)
    with torch.no_grad():
        ref = trace(model, ids, [tok.token_text(i) for i in ids], step=trainer.step_num, with_grad=False)
    (out / "reference.bin").write_bytes(encode_trace(ref, {"prompt": probe, "reason": "reference"}))
    meta = {
        "config": trainer.model_cfg.__dict__,
        "vocab": [tok.token_text(i) for i in range(tok.vocab_size)],
        "chars": tok.chars,
        "probe": probe,
        "step": trainer.step_num,
        "params": model.num_params(),
        "paramGroups": model.param_groups_count(),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="default", choices=[k for k, v in PROFILES.items() if v.tokenizer == "char"])
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    args = p.parse_args()
    out = export(args.config, args.step)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"exported {args.config} to {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
