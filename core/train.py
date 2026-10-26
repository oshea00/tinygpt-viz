"""Trainer: one AdamW update at a time, so a UI can play, pause and single-step it.

The trainer owns no thread; the scheduler decides when step() runs.

    uv run python -m core.train --config bpe-small --steps 5000
"""

import argparse
import math
import time
from pathlib import Path

import torch

from .config import DEFAULT_PROFILE, PROFILES, DeviceError, ModelConfig, TrainConfig, pick_device, profile
from .data import Data, load_data
from .model import GPT
from .preflight import device_info, fmt_duration, measure, report
from .sampling import SampleSettings, generate

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
PROMPT = "Once upon a time"


def checkpoint_schedule(max_step: int = 20000, n: int = 30) -> set[int]:
    """Roughly log-spaced steps from 0, so the replay slider sees early learning in detail."""
    steps = {0}
    for i in range(1, n):
        steps.add(int(round(math.exp(math.log(max_step) * i / (n - 1)))))
    return steps


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, data: Data,
                 device: str, run_dir: Path | None = None):
        self.data = data
        self.cfg = train_cfg
        self.device = device
        self.model_cfg = model_cfg.with_vocab(data.tokenizer.vocab_size)
        self.run_dir = run_dir or self._default_run_dir()
        self.schedule = checkpoint_schedule()
        self.reset(train_cfg.seed)

    def _default_run_dir(self) -> Path:
        return RUNS_DIR / self.model_cfg.name

    def reset(self, seed: int | None = None) -> None:
        seed = self.cfg.seed if seed is None else seed
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.model = GPT(self.model_cfg).to(self.device)
        self.opt = self._make_optimizer()
        self.step_num = 0
        self.last_loss: float | None = None

    def _make_optimizer(self) -> torch.optim.AdamW:
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.lr, betas=(0.9, 0.99),
        )

    def current_lr(self) -> float:
        return self.cfg.lr * min(1.0, (self.step_num + 1) / max(1, self.cfg.warmup_steps))

    def step(self, batch: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
        """One forward, backward and AdamW update. Returns metrics for this step."""
        t0 = time.perf_counter()
        self.model.train()
        x, y = batch or self.data.get_batch("train", self.cfg.batch_size, self.model_cfg.block_size,
                                            self.device, self.gen)
        for g in self.opt.param_groups:
            g["lr"] = self.current_lr()
        _, loss = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.step_num += 1
        self.last_loss = loss.item()
        if self.step_num in self.schedule:
            self.save_checkpoint()
        dt = time.perf_counter() - t0
        return {"step": self.step_num, "loss": self.last_loss, "gradNorm": float(grad_norm),
                "lr": self.current_lr(), "tokensPerSec": x.numel() / dt}

    @torch.no_grad()
    def eval_val(self) -> float:
        self.model.eval()
        g = torch.Generator().manual_seed(0)      # same batches every time, so curves are comparable
        losses = [self.model(*self.data.get_batch("val", self.cfg.batch_size, self.model_cfg.block_size,
                                                  self.device, g))[1].item()
                  for _ in range(self.cfg.eval_batches)]
        return sum(losses) / len(losses)

    def sample(self, prompt: str = PROMPT, n: int = 200, settings: SampleSettings | None = None) -> str:
        self.model.eval()
        tok = self.data.tokenizer
        ids = generate(self.model, tok.encode(prompt), n, settings or SampleSettings(temperature=0.8),
                       torch.Generator().manual_seed(self.step_num))
        return prompt + tok.decode(ids)

    # -- checkpoints --------------------------------------------------------

    def checkpoint_path(self, step: int) -> Path:
        return self.run_dir / f"ckpt_{step:06d}.pt"

    def save_checkpoint(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path(self.step_num)
        torch.save({"step": self.step_num, "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(), "model_cfg": self.model_cfg.__dict__}, path)
        return path

    def checkpoints(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.run_dir.glob("ckpt_*.pt"))

    def load_checkpoint(self, step: int | None = None) -> None:
        steps = self.checkpoints()
        if not steps:
            raise FileNotFoundError(f"no checkpoints in {self.run_dir}")
        step = steps[-1] if step is None else step
        state = torch.load(self.checkpoint_path(step), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        if "opt" in state:      # shared checkpoints are weights only: training resumes with fresh moments
            self.opt.load_state_dict(state["opt"])
        self.step_num = state["step"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PROFILE, choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--device", default=None)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10,
                   help="steps between one-line progress updates (eval runs on its own, slower schedule)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the device report and time estimate, then exit without training")
    args = p.parse_args()

    try:
        device = pick_device(args.device)
    except DeviceError as e:
        raise SystemExit(f"error: {e}")
    cfg = profile(args.config)
    train_cfg = TrainConfig(lr=args.lr, batch_size=args.batch_size)
    print("loading the corpus (downloads and builds it on first use)", flush=True)
    trainer = Trainer(cfg, train_cfg, load_data(), device)
    if args.resume:
        trainer.load_checkpoint()
        print(f"resumed at step {trainer.step_num}", flush=True)

    groups = trainer.model.param_groups_count()
    print(f"{trainer.model.num_params():,} params: "
          + ", ".join(f"{k} {v:,}" for k, v in groups.items()), flush=True)
    print("timing a few steps to estimate the run...", flush=True)
    est = measure(trainer, args.steps, args.sample_every)
    print(report(trainer.model_cfg, train_cfg, device_info(device), est, args.steps, trainer.step_num),
          flush=True)
    if args.dry_run:
        return
    if not args.resume:
        trainer.save_checkpoint()   # step 0: the untrained baseline

    t0 = time.time()
    start_step = trainer.step_num
    while trainer.step_num < args.steps:
        m = trainer.step()
        s = m["step"]
        last = s == args.steps
        if args.log_every and (s % args.log_every == 0 or last):
            elapsed = time.time() - t0
            left = elapsed / max(1, s - start_step) * (args.steps - s)
            print(f"step {s:6d}/{args.steps}  loss {m['loss']:.3f}  {m['tokensPerSec']:,.0f} tok/s  "
                  f"{fmt_duration(elapsed)} elapsed  {fmt_duration(left)} left", flush=True)
        if s % trainer.cfg.eval_every == 0 or last:
            print(f"step {s:6d}  val {trainer.eval_val():.3f}", flush=True)
        if s % args.sample_every == 0 or last:
            print("---\n" + trainer.sample().replace("\x04", "␄\n") + "\n---", flush=True)
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
