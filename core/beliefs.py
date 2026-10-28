"""Belief trajectories (after Bigelow et al., "Stories in Space", arXiv 2605.12412): read a
story one sentence at a time and record, after each sentence, what the model believes about it
and the residual stream that holds that belief.

The paper asks an instruction-tuned model to rate each concept. A small story model cannot
answer questions, so belief is read with a cloze instead: each prefix is followed by a fixed
continuation (" Everyone was very") and the next-token probabilities of a word set (happy,
sad, …), renormalised over the set, are the belief.

Both readings come from one forward pass per prefix. With a causal mask, the positions before
the cloze cannot see it, so the residual at the prefix's last token is the bare prefix's. A
prefix longer than the window is read through its last block_size − len(cloze) tokens. The
analyses (probes, geometry, directions) run in the browser on what this returns."""

import re

import numpy as np
import torch

from . import interventions as iv
from .concepts import snapshot_ids
from .model import GPT
from .tokenizers import BPETokenizer, Tokenizer

DEFAULT_CLOZE = " Everyone was very"
DEFAULT_WORDS = ["happy", "sad", "angry", "scared", "surprised", "excited"]
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """[start, end) character spans of each sentence, ending after its punctuation."""
    spans, start = [], 0
    for m in SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def story_set(tok: BPETokenizer, val: np.ndarray, n: int) -> list[str]:
    """The first n stories of the validation stream: text the model never trained on."""
    ends = np.flatnonzero(np.asarray(val[:2_000_000]) == tok.eos_id)
    stories, start = [], 0
    for end in ends:
        text = tok.decode([int(i) for i in val[start:end]]).strip()
        start = end + 1
        if len(split_sentences(text)) >= 2:   # a trajectory needs two points
            stories.append(text)
        if len(stories) == n:
            break
    return stories


def word_ids(tok: Tokenizer, words: list[str]) -> list[int]:
    """The single token of each word, read with its leading space as it follows the cloze."""
    ids = []
    for w in words:
        enc = tok.encode(" " + w.strip())
        if len(enc) != 1:
            raise ValueError(f"belief words must be single tokens: {w!r} is {len(enc)} tokens")
        ids.append(enc[0])
    return ids


@torch.no_grad()
def run(model: GPT, tok: Tokenizer, stories: list[str], cloze: str, words: list[str],
        interventions: list[dict] | None = None, with_vectors: bool = True) -> dict:
    """Every sentence prefix of every story: readout [N, K] (renormalised over the words), mass
    [N] (their raw total probability), vectors [N, S, d] at the prefix's last token, and records."""
    device = next(model.parameters()).device
    targets = word_ids(tok, words)
    cloze_ids = tok.encode(cloze)
    keep_len = model.cfg.block_size - len(cloze_ids)
    if keep_len < 1:
        raise ValueError("the cloze does not fit in the context window")
    snaps = snapshot_ids(model)
    rewrites = iv.compile_all(interventions)
    prefixes = []
    for si, text in enumerate(stories):
        for k, (start, end) in enumerate(split_sentences(text)):
            ids = tok.encode(text[:end])[-keep_len:]
            prefixes.append(({"story": si, "sentence": k, "start": start, "end": end}, ids))
    readout, mass, vectors = [], [], []
    model.eval()
    for b0 in range(0, len(prefixes), 64):
        chunk = prefixes[b0:b0 + 64]
        rows = [ids + cloze_ids for _, ids in chunk]
        width = max(len(r) for r in rows)
        # Right padding is harmless: with a causal mask, no real position can see it.
        batch = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=device)
        got: dict[str, torch.Tensor] = {}

        def tap(stage_id: str, x: torch.Tensor) -> torch.Tensor:
            x = iv.apply(rewrites, stage_id, x)
            if with_vectors and stage_id in snaps:
                got[stage_id] = x
            return x

        with model.tapped(tap):
            logits, _ = model(batch)
        b = torch.arange(len(chunk), device=device)
        last = torch.tensor([len(r) - 1 for r in rows], device=device)
        p = torch.softmax(logits[b, last].float(), dim=-1)[:, targets]
        mass.append(p.sum(-1))
        readout.append(p / p.sum(-1, keepdim=True))
        if with_vectors:
            pos = torch.tensor([len(ids) - 1 for _, ids in chunk], device=device)
            vectors.append(torch.stack([got[s][b, pos] for s in snaps], dim=1).float())
    K, d = len(targets), model.cfg.d_model
    cat = lambda xs, shape: torch.cat(xs).cpu().numpy() if xs else np.zeros(shape, np.float32)   # noqa: E731
    out = {"readout": cat(readout, (0, K)), "mass": cat(mass, (0,)), "prefixes": [r for r, _ in prefixes],
           "snapshots": snaps}
    if with_vectors:
        out["vectors"] = cat(vectors, (0, len(snaps), d))
    return out
