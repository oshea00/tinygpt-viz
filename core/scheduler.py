"""One thread owns the model. Every UI request and every training step runs on it, in
order, so a trace always reflects one consistent set of weights.

Commands arrive on a queue as dicts (the JSON protocol in PROJECT.md). Results leave
through `emit`, either JSON-ready dicts or binary frames (bytes).
"""

import math
import queue
import threading
import time
import traceback
from typing import Callable

import torch

from .manual_backward import compare as manual_compare
from .sampling import SampleSettings, sample_next
from .trace_io import encode_tensors, encode_trace, encode_weights
from .tracer import trace
from .train import Trainer

Message = dict | bytes
DEFAULT_PROBE = "Once upon a time, there was a little girl named Lily. She"
METRICS_INTERVAL = 0.1      # seconds between metrics messages while playing


class ModelWorker(threading.Thread):
    def __init__(self, trainer: Trainer, emit: Callable[[Message], None], probe: str = DEFAULT_PROBE,
                 sample_every: int = 500):
        super().__init__(daemon=True, name="model-worker")
        self.trainer = trainer
        self.emit = emit
        self.probe = probe
        self.sample_every = sample_every
        self.commands: queue.Queue[dict] = queue.Queue()
        self.playing = False
        self.stopping = False
        self.weights_version = 0         # bumps whenever the weights change
        self.sent_weights_version = -1
        self._points: list[list[float]] = []
        self._last_metrics = 0.0
        self._last_tps = 0.0

    # -- public, thread-safe ------------------------------------------------

    def submit(self, cmd: dict) -> None:
        self.commands.put(cmd)

    def stop(self) -> None:
        self.stopping = True
        self.commands.put({"type": "_wake"})

    def hello(self) -> dict:
        t = self.trainer
        tok = t.data.tokenizer
        return {"type": "hello", "config": t.model_cfg.__dict__, "vocab": [tok.token_text(i) for i in range(tok.vocab_size)],
                "params": t.model.num_params(), "paramGroups": t.model.param_groups_count(),
                "checkpoints": t.checkpoints(), "probe": self.probe, **self.status()}

    def status(self) -> dict:
        t = self.trainer
        return {"step": t.step_num, "playing": self.playing, "lr": t.cfg.lr, "batchSize": t.cfg.batch_size,
                "traceEvery": t.cfg.trace_every, "device": t.device, "checkpoints": t.checkpoints()}

    # -- the loop -----------------------------------------------------------

    def run(self) -> None:
        while not self.stopping:
            batch = self._drain(block=not self.playing)
            for cmd in batch:
                try:
                    self._dispatch(cmd)
                except Exception as e:     # one bad command must not kill the worker
                    traceback.print_exc()
                    self.emit({"type": "error", "message": f"{cmd.get('type')}: {e}"})
            if self.playing and not self.stopping:
                self._train_step()

    def _drain(self, block: bool) -> list[dict]:
        cmds = []
        try:
            cmds.append(self.commands.get(timeout=0.2) if block else self.commands.get_nowait())
            while True:
                cmds.append(self.commands.get_nowait())
        except queue.Empty:
            pass
        # Keystrokes arrive faster than they matter: of consecutive "run"s keep only the last
        # (per tag, so a baseline run is never swallowed by the run it is compared with).
        out: list[dict] = []
        for c in cmds:
            if out and c["type"] == "run" and out[-1]["type"] == "run" and c.get("tag") == out[-1].get("tag"):
                out[-1] = c
            else:
                out.append(c)
        return [c for c in out if c["type"] != "_wake"]

    def _dispatch(self, cmd: dict) -> None:
        kind = cmd["type"]
        t = self.trainer
        if kind == "run":
            self.send_trace(cmd["prompt"], with_grad=cmd.get("withGrad", True), reason="run",
                            target_index=cmd.get("targetIndex"), interventions=cmd.get("interventions"),
                            tag=cmd.get("tag"))
        elif kind == "train":
            action = cmd["action"]
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
                self._flush_metrics(force=True)
            elif action == "step":
                self.playing = False
                self._train_step()
                self._flush_metrics(force=True)
                self.send_trace(self.probe, reason="probe")
            elif action == "reset":
                self.playing = False
                t.reset(cmd.get("seed"))
                self._points.clear()
                self.weights_version += 1
                self.emit({"type": "reset"})
                self.send_trace(self.probe, reason="probe")
            self.emit({"type": "status", **self.status()})
        elif kind == "set":
            if "lr" in cmd:
                t.cfg.lr = float(cmd["lr"])
            if "batchSize" in cmd:
                t.cfg.batch_size = int(cmd["batchSize"])
            if "traceEvery" in cmd:
                t.cfg.trace_every = max(1, int(cmd["traceEvery"]))
            if "probe" in cmd:
                self.probe = cmd["probe"]
            self.emit({"type": "status", **self.status()})
        elif kind == "sample":
            self._sample(cmd)
        elif kind == "loadCheckpoint":
            self.playing = False
            # Replaying must not lose training: keep the current state as a checkpoint first.
            if t.step_num not in t.checkpoints():
                t.save_checkpoint()
            t.load_checkpoint(cmd.get("step"))
            self.weights_version += 1
            self.emit({"type": "status", **self.status()})
            self.send_trace(self.probe, reason="probe")
        elif kind == "stepExplain":
            self.playing = False
            self._step_explain(cmd.get("prompt", self.probe))
            self.emit({"type": "status", **self.status()})
        elif kind == "manualBackward":
            # Hand-written backward vs autograd, stage by stage, for the prompt's trace.
            prompt = cmd.get("prompt", self.probe)
            tr = self.trace_prompt(prompt)
            if not any(st.grad is not None for st in tr.stages):
                raise ValueError("the prompt needs at least two tokens to have a loss")
            weights = {n: p.detach().float().cpu().numpy() for n, p in t.model.named_weights().items()}
            self.emit({"type": "manualBackward", "prompt": prompt, "step": t.step_num, **manual_compare(tr, weights)})
        elif kind == "save":
            t.save_checkpoint()
            self.emit({"type": "status", **self.status()})
        elif kind == "hello":
            self.emit(self.hello())
        else:
            raise ValueError(f"unknown command {kind!r}")

    # -- work ---------------------------------------------------------------

    def _train_step(self) -> None:
        t = self.trainer
        m = t.step()
        self.weights_version += 1
        self._points.append([m["step"], m["loss"]])
        self._last_tps = m["tokensPerSec"]
        s = m["step"]
        val = t.eval_val() if s % t.cfg.eval_every == 0 else None
        self._flush_metrics(force=val is not None, val_loss=val)
        if s % t.cfg.trace_every == 0:
            self.send_trace(self.probe, reason="probe")
        if s % self.sample_every == 0:
            self.emit({"type": "sample", "step": s, "text": t.sample(self.probe, 200), "auto": True})

    def _step_explain(self, prompt: str) -> None:
        """One training step on the prompt itself, with everything needed to show it:
        loss and p(correct next token) before and after, and for every weight its value,
        its gradient, and the change Adam actually applied."""
        t = self.trainer
        tok = t.data.tokenizer
        ids = tok.encode(prompt)[-(t.model_cfg.block_size + 1):]
        if len(ids) < 2:
            raise ValueError("prompt needs at least two tokens")
        window = [tok.token_text(i) for i in ids[:-1]]

        def trace_window():
            return trace(t.model, ids[:-1], window, step=t.step_num, with_grad=False)

        before = trace_window()
        params = t.model.named_weights()
        w_before = {n: p.detach().float().cpu().numpy().copy() for n, p in params.items()}
        x = torch.tensor([ids[:-1]], device=t.device)
        y = torch.tensor([ids[1:]], device=t.device)
        m = t.step(batch=(x, y))
        self.weights_version += 1
        self._points.append([m["step"], m["loss"]])
        grads = {n: p.grad.float().cpu().numpy() for n, p in params.items() if p.grad is not None}
        deltas = {n: p.detach().float().cpu().numpy() - w_before[n] for n, p in params.items()}
        after = trace_window()

        def p_correct(tr) -> list[float]:
            probs = tr.stage("probs").output
            return [float(probs[i, ids[i + 1]]) for i in range(len(ids) - 1)]

        def loss(tr) -> float:
            return -sum(math.log(max(p, 1e-12)) for p in p_correct(tr)) / (len(ids) - 1)

        self.emit(encode_tensors("stepExplain", {"before": w_before, "grad": grads, "delta": deltas}, {
            "step": t.step_num, "prompt": prompt, "text": [tok.token_text(i) for i in ids],
            "lossBefore": loss(before), "lossAfter": loss(after),
            "pBefore": p_correct(before), "pAfter": p_correct(after), "lr": m["lr"],
        }))
        self._flush_metrics(force=True)
        self.send_trace(prompt, reason="stepExplain")

    def _flush_metrics(self, force: bool = False, val_loss: float | None = None) -> None:
        now = time.perf_counter()
        if not self._points and val_loss is None:
            return
        if force or now - self._last_metrics >= METRICS_INTERVAL:
            msg = {"type": "metrics", "step": self.trainer.step_num, "points": self._points,
                   "tokensPerSec": self._last_tps}
            if val_loss is not None:
                msg["valLoss"] = val_loss
            self.emit(msg)
            self._points = []
            self._last_metrics = now

    def send_weights_if_changed(self) -> None:
        if self.sent_weights_version == self.weights_version:
            return
        w = {n: p.detach().float().cpu().numpy() for n, p in self.trainer.model.named_weights().items()}
        self.emit(encode_weights(w, self.trainer.step_num, {"version": self.weights_version}))
        self.sent_weights_version = self.weights_version

    def trace_prompt(self, prompt: str, with_grad: bool = True, target_index: int | None = None,
                     interventions: list[dict] | None = None):
        tok = self.trainer.data.tokenizer
        ids = tok.encode(prompt) or [0]
        return trace(self.trainer.model, ids, [tok.token_text(i) for i in ids], step=self.trainer.step_num,
                     with_grad=with_grad, target_index=target_index, interventions=interventions)

    def send_trace(self, prompt: str, with_grad: bool = True, reason: str = "run",
                   target_index: int | None = None, interventions: list[dict] | None = None,
                   tag: str | None = None) -> None:
        self.send_weights_if_changed()
        tr = self.trace_prompt(prompt, with_grad, target_index, interventions)
        extra = {"prompt": prompt, "reason": reason, "weightsVersion": self.weights_version,
                 "interventions": interventions or []}
        if tag:
            extra["tag"] = tag   # lets the UI tell a baseline trace from the one it displays
        self.emit(encode_trace(tr, extra))

    @torch.no_grad()
    def _sample(self, cmd: dict) -> None:
        t = self.trainer
        tok = t.data.tokenizer
        settings = SampleSettings(cmd.get("temperature", 1.0), cmd.get("topK"), cmd.get("topP"))
        prompt = cmd.get("prompt", self.probe)
        ids = tok.encode(prompt) or [0]
        gen = torch.Generator().manual_seed(int(cmd.get("seed", time.time_ns() % 2**31)))
        model = t.model
        model.eval()
        out, chosen_probs, probs = [], [], None
        for _ in range(int(cmd.get("n", 1))):
            window = (ids + out)[-model.cfg.block_size:]
            logits, _ = model(torch.tensor([window], device=t.device))
            nxt, probs = sample_next(logits[0, -1], settings, gen)
            out.append(nxt)
            chosen_probs.append(float(probs[nxt]))
        self.emit({"type": "sample", "step": t.step_num, "prompt": prompt, "text": tok.decode(out),
                   "tokens": [tok.token_text(i) for i in out], "chosenProbs": chosen_probs,
                   "probs": probs.tolist() if probs is not None else None})
