"""Write a profile's model in the form the in-browser backend reads (ui/src/engine/), so the UI
can run with no server at all.

    uv run python -m core.web_export --config bpe-small            # latest checkpoint and step 0
    uv run python -m core.web_export --config bpe-small --all      # every checkpoint, for replay
    uv run python -m core.web_export --config bpe-small --train    # with train.bin, to train in the browser

The output directory (default ui/public/model/, served at /model/) holds the following. The
GitHub Pages build reads the same files from Hugging Face instead (`core.hub push-web`).

    model.json       config, parameter counts, checkpoint steps, and every stage's metadata (an
                     adapter's stages too)
    tokenizer.json   the BPE tokenizer
    val.bin          the validation tokens (story sets, neuron search, validation loss), fetched when needed
    train.bin        with --train: the training tokens (119 MB), fetched when training starts
    ckpt_NNNNNN.bin  one checkpoint's weights as a "weights" frame (core/trace_io.py)
    adapters/*.bin   the low-rank adapters in runs/<profile>/adapters/, as "adapter" frames
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import DEFAULT_PROFILE, PROFILES, profile
from .data import data_dir
from .hub import local_adapters, local_checkpoints, pull
from .model import GPT
from .scheduler import DEFAULT_PROBE
from .stages import describe, stage_order
from .tokenizers import BPETokenizer
from .trace_io import encode_weights

OUT = Path(__file__).resolve().parent.parent / "ui" / "public" / "model"


def export(config: str, out: Path = OUT, every: bool = False, train: bool = False) -> None:
    tok = BPETokenizer.load(data_dir() / "tokenizer.json")
    cfg = profile(config).with_vocab(tok.vocab_size)
    ckpts = local_checkpoints(cfg)
    if not ckpts and pull(cfg):
        ckpts = local_checkpoints(cfg)
    if not ckpts:
        raise SystemExit(f"error: no checkpoints for {cfg.name}")
    steps = lambda paths: [int(p.stem.split("_")[1]) for p in paths]   # noqa: E731
    chosen = ckpts if every else [c for c in ckpts if steps([c])[0] == 0] + [ckpts[-1]]
    out.mkdir(parents=True, exist_ok=True)
    model = GPT(cfg)
    for path in chosen:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        w = {n: p.detach().float().numpy() for n, p in model.named_weights().items()}
        (out / f"{path.stem}.bin").write_bytes(encode_weights(w, state["step"], {"version": state["step"]}))
        print(f"wrote {out / path.stem}.bin")
    shutil.copy(data_dir() / "tokenizer.json", out / "tokenizer.json")
    adapters = local_adapters(cfg)
    for a in adapters:
        (out / "adapters").mkdir(exist_ok=True)
        shutil.copy(a, out / "adapters" / a.name)
    for split, wanted in (("val", True), ("train", train)):
        src = data_dir() / f"{split}.bin"
        if wanted and not src.exists():
            raise SystemExit(f"error: no {src}: run uv run python -m core.data to build it")
        if wanted:
            shutil.copy(src, out / f"{split}.bin")
    meta = {"config": cfg.__dict__, "params": model.num_params(), "paramGroups": model.param_groups_count(),
            "checkpoints": steps(chosen), "probe": DEFAULT_PROBE, "adapters": [a.stem for a in adapters],
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),   # the browser's cache key
            "stages": [describe(s, cfg).to_json() for s in stage_order(cfg, lora=True)]}
    (out / "model.json").write_text(json.dumps(meta))
    print(f"wrote {out / 'model.json'} ({len(chosen)} checkpoints)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PROFILE, choices=list(PROFILES))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--all", action="store_true", help="every checkpoint (for replay), not just step 0 and the latest")
    p.add_argument("--train", action="store_true", help="include train.bin (119 MB), so the browser can train")
    args = p.parse_args()
    export(args.config, args.out, args.all, args.train)


if __name__ == "__main__":
    main()
