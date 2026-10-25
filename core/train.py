"""Trainer: one AdamW update at a time, so a UI can play, pause and single-step it.

The trainer owns no thread; the scheduler decides when step() runs.

    uv run python -m core.train --config default --steps 5000
"""

import argparse
import math
import time
from pathlib import Path

import torch

from .config import PROFILES, ModelConfig, TrainConfig, pick_device
from .data import CharData, load_data
from .model import GPT
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
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, data: CharData,
                 device: str, run_dir: Path | None = None):
        self.data = data
        self.cfg = train_cfg
        self.device = device
        self.model_cfg = model_cfg.with_vocab(data.tokenizer.vocab_size)
        self.run_dir = run_dir or RUNS_DIR / self.model_cfg.name
        self.schedule = checkpoint_schedule()
        self.reset(train_cfg.seed)

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
        state = torch.load(self.checkpoint_path(step), map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.step_num = state["step"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="default", choices=list(PROFILES))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--device", default=None)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = pick_device(args.device)
    trainer = Trainer(PROFILES[args.config], TrainConfig(lr=args.lr, batch_size=args.batch_size),
                      load_data(PROFILES[args.config].tokenizer), device)
    if args.resume:
        trainer.load_checkpoint()
    else:
        trainer.save_checkpoint()   # step 0: the untrained baseline
    groups = trainer.model.param_groups_count()
    print(f"{args.config} on {device}: {trainer.model.num_params():,} params "
          + ", ".join(f"{k} {v:,}" for k, v in groups.items()))
    t0 = time.time()
    while trainer.step_num < args.steps:
        m = trainer.step()
        s = m["step"]
        if s % trainer.cfg.eval_every == 0 or s == args.steps:
            print(f"step {s:6d}  loss {m['loss']:.3f}  val {trainer.eval_val():.3f}  "
                  f"{m['tokensPerSec']:,.0f} tok/s  {time.time() - t0:.0f}s")
        if s % args.sample_every == 0 or s == args.steps:
            print("---\n" + trainer.sample().replace("\x04", "␄\n") + "\n---")
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
