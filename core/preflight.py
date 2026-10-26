"""Check the machine before committing to a long training run, and estimate how long it will take.

The profiles in PROJECT.md were timed on a desktop GPU. On a laptop the same command can be
sixty times slower, so `train.py` times a handful of real steps first and prints an ETA rather
than letting the learner discover it after an hour of silence.

    uv run python -m core.preflight --config bpe-small --batch-size 128
"""

import argparse
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass

import torch

from .config import DEFAULT_PROFILE, PROFILES, DeviceError, ModelConfig, TrainConfig, pick_device, profile

WARMUP_STEPS = 3          # CUDA context, cuBLAS handles and allocator warm up on the first steps
TIMED_STEPS = 5
VRAM_HEADROOM = 0.85      # fraction of total VRAM above which a run is likely to OOM later


@dataclass
class DeviceInfo:
    device: str
    name: str
    total_vram: float = 0.0      # GB, 0 when not a GPU
    note: str = ""               # anything about this device the user should know before waiting


def device_info(device: str) -> DeviceInfo:
    kind = torch.device(device).type
    if kind == "cuda":
        p = torch.cuda.get_device_properties(torch.device(device))
        info = DeviceInfo(device, p.name, p.total_memory / 1e9)
        info.note = _cuda_throttle_note()
        return info
    if kind == "mps":
        return DeviceInfo(device, "Apple MPS")
    return DeviceInfo(device, f"CPU ({torch.get_num_threads()} threads)",
                      note="no GPU found — expect a long run; try fewer --steps first "
                           "(for an AMD GPU on Linux, install the ROCm build: see the README)")


def _cuda_throttle_note() -> str:
    """Laptop GPUs often run far below their rated clocks. Say so, with the numbers."""
    pairs = _nvidia_smi("clocks.current.memory", "clocks.max.memory", "power.draw")
    if not pairs:
        return ""
    try:
        mem, mem_max, power = float(pairs[0]), float(pairs[1]), float(pairs[2])
    except ValueError:
        return ""
    if mem_max and mem / mem_max < 0.5:
        return (f"memory clock {mem:.0f} of {mem_max:.0f} MHz at {power:.0f} W — this GPU is "
                "power-capped, so bandwidth-bound work runs several times slower than its rating")
    return ""


def _nvidia_smi(*fields: str) -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    import subprocess
    try:
        out = subprocess.run([exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]


@dataclass
class Estimate:
    step_seconds: float
    tokens_per_sec: float
    peak_vram: float             # GB, 0 when not CUDA
    train_seconds: float
    eval_seconds: float
    sample_seconds: float
    total_seconds: float
    checkpoint_gb: float


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def measure(trainer, steps: int, sample_every: int = 0) -> Estimate:
    """Run a few real steps to get a per-step time, then extrapolate the whole run.

    The probe must leave the trainer exactly as it found it, so weights, optimiser state and
    step counter are snapshotted and restored: a resumed run keeps its checkpoint. Checkpoint
    writes are suppressed while timing, so the figure is compute only; their disk cost is
    reported separately.
    """
    snapshot = ({k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
                deepcopy(trainer.opt.state_dict()), trainer.step_num)
    schedule, trainer.schedule = trainer.schedule, set()
    try:
        for _ in range(WARMUP_STEPS):
            trainer.step()
        _synchronize(trainer.device)
        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            m = trainer.step()
        _synchronize(trainer.device)
        step_seconds = (time.perf_counter() - t0) / TIMED_STEPS

        t0 = time.perf_counter()
        trainer.eval_val()
        _synchronize(trainer.device)
        eval_seconds = time.perf_counter() - t0

        sample_seconds = 0.0
        if sample_every:
            t0 = time.perf_counter()
            trainer.sample()
            _synchronize(trainer.device)
            sample_seconds = time.perf_counter() - t0
    finally:
        trainer.schedule = schedule
        model_state, opt_state, step_num = snapshot
        trainer.model.load_state_dict(model_state)
        trainer.opt.load_state_dict(opt_state)
        trainer.step_num = step_num

    remaining = max(0, steps - trainer.step_num)
    train_s = step_seconds * remaining
    eval_s = eval_seconds * (remaining / max(1, trainer.cfg.eval_every))
    sample_s = sample_seconds * (remaining / sample_every) if sample_every else 0.0
    peak = torch.cuda.max_memory_allocated(trainer.device) / 1e9 if _is_cuda(trainer.device) else 0.0
    n_ckpt = len([s for s in schedule if trainer.step_num <= s <= steps]) + 1
    bytes_per_ckpt = trainer.model.num_params() * 12      # fp32 weights + two AdamW moments
    return Estimate(
        step_seconds=step_seconds,
        tokens_per_sec=m["tokensPerSec"],
        peak_vram=peak,
        train_seconds=train_s,
        eval_seconds=eval_s,
        sample_seconds=sample_s,
        total_seconds=train_s + eval_s + sample_s,
        checkpoint_gb=n_ckpt * bytes_per_ckpt / 1e9,
    )


def _is_cuda(device: str) -> bool:
    return torch.device(device).type == "cuda"


def _synchronize(device: str) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(torch.device(device))


def report(cfg: ModelConfig, train_cfg: TrainConfig, info: DeviceInfo, est: Estimate,
           steps: int, done: int = 0) -> str:
    """The block printed before a run starts: what will run, where, and for how long."""
    lines = [
        f"{cfg.name}: {cfg.n_layer}L {cfg.n_head}H d{cfg.d_model} "
        f"vocab {cfg.vocab_size:,} ctx {cfg.block_size}, batch {train_cfg.batch_size}",
        f"device: {info.device} — {info.name}"
        + (f", {info.total_vram:.1f} GB VRAM" if info.total_vram else ""),
    ]
    if info.note:
        lines.append(f"  ! {info.note}")
    lines.append(
        f"measured: {est.step_seconds * 1000:,.0f} ms/step, {est.tokens_per_sec:,.0f} tok/s"
        + (f", peak {est.peak_vram:.1f} GB VRAM" if est.peak_vram else "")
    )
    if info.total_vram and est.peak_vram > info.total_vram * VRAM_HEADROOM:
        lines.append(f"  ! peak VRAM is within {(1 - VRAM_HEADROOM) * 100:.0f}% of the card — "
                     "a later step may OOM; lower --batch-size")
    parts = [f"{fmt_duration(est.train_seconds)} training", f"{fmt_duration(est.eval_seconds)} eval"]
    if est.sample_seconds:
        parts.append(f"{fmt_duration(est.sample_seconds)} sampling")
    lines.append(
        f"estimate: {fmt_duration(est.total_seconds)} for {steps - done:,} steps "
        f"({' + '.join(parts)}), ~{est.checkpoint_gb:.1f} GB of checkpoints"
    )
    return "\n".join(lines)


def main() -> None:
    from .data import load_data
    from .train import Trainer

    p = argparse.ArgumentParser(description="Report the device and estimate a run's duration, then exit.")
    p.add_argument("--config", default=DEFAULT_PROFILE, choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    try:
        device = pick_device(args.device)
    except DeviceError as e:
        raise SystemExit(f"error: {e}")
    cfg = profile(args.config)
    train_cfg = TrainConfig(batch_size=args.batch_size)
    trainer = Trainer(cfg, train_cfg, load_data(), device)
    est = measure(trainer, args.steps)
    print(report(trainer.model_cfg, train_cfg, device_info(device), est, args.steps))


if __name__ == "__main__":
    main()
