"""Low-rank adapters (LoRA, Hu et al. 2021): fine-tuning a few small matrices beside frozen weights.

For a stage y = x W + b, an adapter adds (x A) B · s, with A [d_in, r] and B [r, d_out] for a
small rank r. W never changes; only A and B train. B starts at zero, so a new adapter is an exact
no-op, and the change it can make to W, ΔW = A B, has rank at most r.

An adapter lives outside the model's parameters (GPT.adapted switches it on), so checkpoints,
the optimizer and the replay slider never see it. Commands choose one by name, like interventions,
so on a shared server one learner's adapter never changes what another sees.

    uv run python -m core.lora --topic dragon            # the fine-tuning tab's story set (ui/src/finetune/)
    uv run python -m core.lora --topic dragon --train    # an adapter trained on it in advance (dragon-ready)
"""

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import fetch_head, split_stories
from .model import GPT
from .stages import LORA_TARGETS
from .tokenizers import BPETokenizer, Tokenizer

DEFAULT_TARGETS = ("q", "v")
DEFAULT_RANK = 4
EVAL_WINDOWS = 16       # windows of each set the losses are measured on (few: the browser's CPU repeats this)
# Openers that end where a noun goes: how often the model reaches for the topic word there is the
# lesson's measure of how far the adapter has tilted it.
OPENERS = [
    "Once upon a time, there was a",
    "One day, Tom went to the park and saw a",
    "Lily looked up at the sky and saw a big",
    "In the forest lived a",
    "Under the old bridge, there was a",
    "Max wanted to be friends with the",
    "The little girl had a toy",
    "Far away, on top of a hill, lived a",
]


@dataclass
class Adapter:
    name: str
    rank: int
    A: dict[str, torch.Tensor]             # per stage id ("L3.q"): [d_in, r]
    B: dict[str, torch.Tensor]             # [r, d_out]
    base_step: int                         # the checkpoint it was trained on
    info: dict = field(default_factory=dict)   # how it was trained, shown with it
    version: int = 0                       # bumps whenever A or B change

    @property
    def targets(self) -> list[str]:
        return list(self.A)

    def pairs(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {t: (self.A[t], self.B[t]) for t in self.A}

    def parameters(self) -> list[torch.Tensor]:
        return [*self.A.values(), *self.B.values()]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def weights(self) -> dict[str, np.ndarray]:
        """A and B under the names the stages list: "L3.q.lora_A", "L3.q.lora_B"."""
        out = {}
        for t in self.A:
            out[f"{t}.lora_A"] = self.A[t].detach().float().cpu().numpy().copy()
            out[f"{t}.lora_B"] = self.B[t].detach().float().cpu().numpy().copy()
        return out

    def header(self) -> dict:
        return {"name": self.name, "rank": self.rank, "targets": self.targets, "baseStep": self.base_step,
                "params": self.num_params(), "version": self.version, "info": self.info}


def stage_ids(model: GPT, targets: list[str] | tuple[str, ...]) -> list[str]:
    """Short target names ("q", "mlp.down") in every layer, or stage ids already ("L3.q")."""
    ids = []
    for t in targets:
        if t in LORA_TARGETS:
            ids += [f"L{l}.{t}" for l in range(model.cfg.n_layer)]
        elif t.split(".", 1)[-1] in LORA_TARGETS:
            ids.append(t)
        else:
            raise ValueError(f"an adapter can't attach to {t!r}")
    return ids


def new_adapter(model: GPT, name: str, targets=DEFAULT_TARGETS, rank: int = DEFAULT_RANK,
                seed: int = 0, base_step: int = 0) -> Adapter:
    """A ~ N(0, 1/d_in), so x·A has about unit size; B = 0, so the adapter starts as a no-op."""
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    A, B = {}, {}
    for sid in stage_ids(model, targets):
        l, short = sid.split(".", 1)
        W = getattr(model.blocks[int(l[1:])], LORA_TARGETS[short])
        d_in, d_out = W.shape
        A[sid] = (torch.randn(d_in, rank, generator=g) / d_in ** 0.5).to(device).requires_grad_()
        B[sid] = torch.zeros(rank, d_out, device=device, requires_grad=True)
    return Adapter(name, rank, A, B, base_step)


def from_weights(name: str, weights: dict[str, np.ndarray], header: dict, device: str = "cpu") -> Adapter:
    """An adapter from a weights frame (see Adapter.weights and Adapter.header)."""
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)   # noqa: E731
    targets = header["targets"]
    return Adapter(name, int(header["rank"]), {s: t(weights[f"{s}.lora_A"]) for s in targets},
                   {s: t(weights[f"{s}.lora_B"]) for s in targets}, int(header.get("baseStep", 0)),
                   header.get("info", {}))


def windows(tok: BPETokenizer, texts: list[str], block_size: int) -> list[list[int]]:
    """Each story as in training (between end-of-text tokens), cut into windows of block_size + 1."""
    out = []
    for text in texts:
        ids = [tok.eos_id, *tok.encode(text.strip()), tok.eos_id]
        for s in range(0, max(1, len(ids) - 1), block_size):
            w = ids[s:s + block_size + 1]
            if len(w) >= 2:
                out.append(w)
    return out


def batch_of(ws: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets, padded to the longest window; padded targets are -1 (ignored)."""
    T = max(len(w) for w in ws) - 1
    x = torch.zeros(len(ws), T, dtype=torch.long)
    y = torch.full((len(ws), T), -1, dtype=torch.long)
    for i, w in enumerate(ws):
        x[i, :len(w) - 1] = torch.tensor(w[:-1])
        y[i, :len(w) - 1] = torch.tensor(w[1:])
    return x.to(device), y.to(device)


@torch.no_grad()
def mean_loss(model: GPT, ws: list[list[int]], chunk: int = 32) -> float:
    """Mean loss per predicted token over the windows."""
    device = next(model.parameters()).device
    total, n = 0.0, 0
    for i in range(0, len(ws), chunk):
        x, y = batch_of(ws[i:i + chunk], device)
        k = int((y >= 0).sum())
        total += model(x, y)[1].item() * k
        n += k
    return total / max(n, 1)


@torch.no_grad()
def p_word(model: GPT, tok: Tokenizer, openers: list[str], word_id: int) -> float:
    """The mean probability of the word as the next token after each opener."""
    device = next(model.parameters()).device
    ps = []
    for text in openers:
        logits, _ = model(torch.tensor([tok.encode(text)[-model.cfg.block_size:]], device=device))
        ps.append(float(torch.softmax(logits[0, -1], -1)[word_id]))
    return sum(ps) / len(ps)


def topic_id(tok: Tokenizer, word: str) -> int:
    """The token of " word" (its first, if it splits)."""
    return tok.encode(" " + word.strip())[0]


def optimizer(ad: Adapter, lr: float) -> torch.optim.AdamW:
    """AdamW on A and B only, as the main trainer's but with no weight decay."""
    return torch.optim.AdamW(ad.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)


def train_step(model: GPT, ad: Adapter, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor,
               grad_clip: float = 1.0) -> float:
    """One step on A and B: the base weights take no gradients (they are frozen for the step)."""
    frozen = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.eval()
    try:
        for p in frozen:
            p.requires_grad_(False)
        with model.adapted(ad.pairs()):
            _, loss = model(x, y)
        for p in ad.parameters():
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ad.parameters(), grad_clip)
        opt.step()
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.train(was_training)
    return loss.item()


class LoraJob:
    """Trains an adapter one step at a time (so the scheduler can stream progress and stop it):
    AdamW on A and B only, on random batches of the stories' windows."""

    def __init__(self, model: GPT, tok: BPETokenizer, adapter: Adapter, texts: list[str], held_out: list[str],
                 topic: str = "dragon", steps: int = 150, lr: float = 3e-3, batch: int = 8,
                 eval_every: int = 10, seed: int = 0):
        self.model, self.tok, self.adapter = model, tok, adapter
        T = model.cfg.block_size
        self.train = windows(tok, texts, T)
        if not self.train:
            raise ValueError("no stories to train on")
        self.held_out = windows(tok, held_out, T)
        self.word = topic_id(tok, topic)
        self.steps, self.batch, self.eval_every = steps, batch, eval_every
        self.step_num = 0
        self.rand = random.Random(seed)
        self.opt = optimizer(adapter, lr)
        adapter.info = {"kind": "lora", "topic": topic, "stories": len(texts), "windows": len(self.train),
                        "steps": steps, "lr": lr, "batch": batch, "curve": []}

    @property
    def done(self) -> bool:
        return self.step_num >= self.steps

    def evaluate(self) -> dict:
        m, ad = self.model, self.adapter
        with m.adapted(ad.pairs()):
            return {"trainLoss": mean_loss(m, self.train[:EVAL_WINDOWS]),
                    "heldOutLoss": mean_loss(m, self.held_out[:EVAL_WINDOWS]),
                    "pTopic": p_word(m, self.tok, OPENERS, self.word)}

    def start(self) -> dict:
        """The evaluations before any training: the base model's numbers (B is still zero)."""
        point = {"step": 0, **self.evaluate()}
        self.adapter.info["curve"].append(point)
        return point

    def step(self) -> dict:
        """One AdamW step; every eval_every steps (and the last) the evaluations too."""
        m, ad = self.model, self.adapter
        x, y = batch_of(self.rand.sample(self.train, min(self.batch, len(self.train))), x_device(m))
        loss = train_step(m, ad, self.opt, x, y)
        self.step_num += 1
        ad.version += 1
        out = {"step": self.step_num, "loss": loss}
        if self.step_num % self.eval_every == 0 or self.done:
            out.update(self.evaluate())
            ad.info["curve"].append(out)
        return out


def x_device(model: GPT):
    return next(model.parameters()).device


STORY_DIR = Path(__file__).resolve().parent.parent / "ui" / "src" / "finetune"


def story_set(topic: str, n: int = 64, min_mentions: int = 3, max_chars: int = 1100, seed: int = 0) -> list[str]:
    """Training stories for a topic: TinyStories stories that mention it at least min_mentions times."""
    stories = split_stories(fetch_head("TinyStories-train.txt", 250_000_000))
    hits = [re.sub(r"\n\s*\n", "\n", s) for s in stories   # one paragraph each: the tab separates stories by blank lines
            if s.lower().count(topic.lower()) >= min_mentions and len(s) <= max_chars]
    random.Random(seed).shuffle(hits)
    return hits[:n]


RELATED = ["dragon", "monster", "cave", "fire", "knight", "castle", "princess"]


@torch.no_grad()
def mention_rates(model: GPT, tok: BPETokenizer, adapter: dict | None, scale: float, prompts: list[str], samples: int,
                  n_tokens: int = 60, temperature: float = 0.8, words: list[str] = RELATED, seed: int = 0) -> dict:
    """Sample `samples` continuations of each prompt (in one batch per prompt) and report, per word,
    the share of continuations that mention it: an offline measure of how far an adapter tilts the model."""
    device = next(model.parameters()).device
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    hits = {w: 0 for w in words}
    total = 0
    with model.adapted(adapter, scale):
        for prompt in prompts:
            idx = torch.tensor([tok.encode(prompt)] * samples, device=device)
            start = idx.shape[1]
            for _ in range(n_tokens):
                logits, _ = model(idx[:, -model.cfg.block_size:])
                probs = torch.softmax(logits[:, -1] / temperature, -1)
                idx = torch.cat([idx, torch.multinomial(probs, 1, generator=g)], 1)
            for row in idx[:, start:].tolist():
                text = tok.decode(row).split(tok.decode([tok.eos_id]))[0].lower()   # up to the end of the story
                for w in words:
                    hits[w] += w in text
                total += 1
    return {w: hits[w] / total for w in words}


def load_adapter(path: Path, device: str) -> Adapter:
    from .trace_io import decode_tensors

    h, w = decode_tensors(path.read_bytes(), "weights")
    return from_weights(h["name"], w, h, device)


def main() -> None:
    p = argparse.ArgumentParser(description="write a topic's story set for the fine-tuning tab, or train an adapter on it")
    p.add_argument("--topic", default="dragon")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--train", action="store_true",
                   help="train an adapter on the story set, as the lesson does, into runs/<profile>/adapters/<topic>-ready.bin "
                        "(for browsers too slow to train one)")
    p.add_argument("--name", default=None, help="with --train: the adapter's name (default <topic>-ready)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--eval", nargs="+", metavar="NAME",
                   help="sample continuations of the openers with these adapters (from runs/<profile>/adapters/) at each "
                        "strength, and report how often they mention the topic and related words")
    p.add_argument("--samples", type=int, default=50, help="with --eval: continuations per opener")
    p.add_argument("--prompt", action="append", help="with --eval: a prompt to continue instead of the openers (repeatable)")
    p.add_argument("--strengths", default="0,0.5,1,1.5,2")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = STORY_DIR / f"{args.topic}-stories.json"
    if args.eval:
        return evaluate_adapters(args)
    if not args.train:
        texts = story_set(args.topic, args.n)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"topic": args.topic, "stories": texts}, indent=1) + "\n")
        print(f"wrote {len(texts)} stories to {out}")
        return
    from .beliefs import story_set as validation_stories
    from .config import DEFAULT_PROFILE, TrainConfig, pick_device, profile
    from .data import load_data
    from .trace_io import encode_tensors
    from .train import Trainer

    data = load_data(need_train=False)
    t = Trainer(profile(DEFAULT_PROFILE), TrainConfig(), data, pick_device(args.device))
    t.load_checkpoint()
    texts = json.loads(out.read_text())["stories"]
    name = args.name or f"{args.topic}-ready"
    ad = new_adapter(t.model, name, args.targets.split(","), args.rank, base_step=t.step_num)
    job = LoraJob(t.model, data.tokenizer, ad, texts, validation_stories(data.tokenizer, data.splits["val"], 16), args.topic,
                  args.steps, args.lr)
    print({"step": 0, **job.start()})
    while not job.done:
        point = job.step()
        if "pTopic" in point:
            print(point)
    path = t.run_dir / "adapters" / f"{name}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_tensors("adapter", {"weights": ad.weights()}, ad.header()))
    print(f"wrote {path}")



def evaluate_adapters(args) -> None:
    from .config import DEFAULT_PROFILE, TrainConfig, pick_device, profile
    from .data import load_data
    from .train import Trainer

    data = load_data(need_train=False)
    t = Trainer(profile(DEFAULT_PROFILE), TrainConfig(), data, pick_device(args.device))
    t.load_checkpoint()
    tok, m = data.tokenizer, t.model
    words = [args.topic] + [w for w in RELATED if w != args.topic]
    strengths = [float(x) for x in args.strengths.split(",")]
    prompts = args.prompt or OPENERS
    print(f"{args.samples} continuations of each of {len(prompts)} prompts, 60 tokens at temperature 0.8: "
          f"the share that mention each word")
    print(f"{'adapter':>14} {'strength':>8}  " + " ".join(f"{w:>8}" for w in words) + f"  {'p(' + args.topic + ')':>10}")
    for name in args.eval:
        ad = load_adapter(t.run_dir / "adapters" / f"{name}.bin", t.device)
        for s in strengths:
            r = mention_rates(m, tok, ad.pairs(), s, prompts, args.samples, words=words)
            with m.adapted(ad.pairs(), s):
                pt = p_word(m, tok, OPENERS, topic_id(tok, args.topic))
            print(f"{name:>14} {s:>8.1f}  " + " ".join(f"{r[w]:>8.0%}" for w in words) + f"  {pt:>10.2%}")


if __name__ == "__main__":
    main()
