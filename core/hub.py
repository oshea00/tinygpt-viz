"""Share trained checkpoints through a Hugging Face model repo, for machines that only run them.

The repo mirrors this checkout: runs/<profile>/ckpt_*.pt (weights only, no optimizer state) and
the profile's val.bin, so a download lands straight in place.

    uv run python -m core.hub push --config all        # on the machine that trained (hf auth login first)
    uv run python -m core.hub pull --config bpe-small  # elsewhere; the server also does this on start
"""

import argparse
import os
import tempfile
from pathlib import Path

import torch

from .config import PROFILES, ModelConfig, profile
from .data import data_dir
from .train import RUNS_DIR

ROOT = RUNS_DIR.parent
HF_REPO = os.environ.get("TINYGPT_HF_REPO", "oshea00/tinygpt-viz")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _val_path(cfg: ModelConfig) -> Path:
    return data_dir() / "val.bin"


def local_checkpoints(cfg: ModelConfig) -> list[Path]:
    return sorted((RUNS_DIR / cfg.name).glob("ckpt_*.pt"))


def has_checkpoints(cfg: ModelConfig) -> bool:
    return bool(local_checkpoints(cfg))


def strip_checkpoint(src: Path, dst: Path) -> None:
    """The checkpoint without its optimizer state, about a third of the size."""
    state = torch.load(src, map_location="cpu", weights_only=True)
    state.pop("opt", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)


def push(cfg: ModelConfig, repo: str = HF_REPO) -> None:
    from huggingface_hub import HfApi

    ckpts = local_checkpoints(cfg)
    if not ckpts:
        raise SystemExit(f"error: no checkpoints in {RUNS_DIR / cfg.name}")
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for c in ckpts:
            strip_checkpoint(c, staging / "runs" / cfg.name / c.name)
        val = _val_path(cfg)
        (staging / _rel(val)).parent.mkdir(parents=True, exist_ok=True)
        (staging / _rel(val)).write_bytes(val.read_bytes())
        api = HfApi()
        api.create_repo(repo, exist_ok=True)
        print(f"uploading {len(ckpts)} checkpoints of {cfg.name} to {repo}")
        api.upload_folder(repo_id=repo, folder_path=staging, commit_message=f"{cfg.name}: {len(ckpts)} checkpoints")


def pull(cfg: ModelConfig, repo: str = HF_REPO) -> bool:
    """Download a profile's checkpoints and val.bin; False (with a warning) if that isn't possible."""
    from huggingface_hub import snapshot_download

    print(f"fetching {cfg.name} checkpoints from huggingface.co/{repo}")
    try:
        snapshot_download(repo, allow_patterns=[f"runs/{cfg.name}/*", _rel(_val_path(cfg))], local_dir=ROOT)
    except Exception as e:
        print(f"warning: could not download checkpoints ({type(e).__name__}: {e})")
        return False
    if not has_checkpoints(cfg):
        print(f"warning: {repo} has no checkpoints for {cfg.name}")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--config", required=True, choices=[*PROFILES, "all"],
                   help="a profile, or all: every profile with local checkpoints (push) or every profile (pull)")
    p.add_argument("--repo", default=HF_REPO)
    args = p.parse_args()
    if args.config != "all":
        cfgs = [profile(args.config)]
    elif args.action == "push":
        cfgs = [c for c in PROFILES.values() if has_checkpoints(c)]
    else:
        cfgs = list(PROFILES.values())
    for cfg in cfgs:
        push(cfg, args.repo) if args.action == "push" else pull(cfg, args.repo)


if __name__ == "__main__":
    main()
