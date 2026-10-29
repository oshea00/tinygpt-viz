"""Preference tuning with DPO, on a low-rank adapter: teach the story model to
decline requests for stories about unsafe play, and to keep answering similar safe ones.

RLHF as InstructGPT did it (Ouyang et al. 2022) fits a reward model to human preferences between
pairs of answers, then optimises the language model against it with PPO. Direct Preference
Optimisation (Rafailov et al. 2023) gets the same optimum without the reward model: for a prompt x
with a preferred answer y_w and a rejected one y_l,

    loss = -log σ(β [(log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))])

where π is the model being tuned and π_ref the frozen model it starts from. β · log(π/π_ref) acts
as an implicit reward; the loss widens the gap between the two answers' rewards. Here π is the base
model with a LoRA adapter (core/lora.py) and π_ref is the base model alone, so the reference is the
same network with the adapter off.

A story model has never seen a request, so the requests are new to it either way. The rejected
answer to an unsafe request is the base model's own story; the preferred one is a refusal. For a
safe request the roles swap, so the adapter must learn which requests to decline, not just to
decline. Whole activities are held out, to see whether refusing generalises to unseen ones.

The requests, the refusal and how many activities to hold out come from a preference set, a JSON
file (data/preferences/refusal.json is the one the model ships with; docs/instructors/
preference-adapters.md describes the format). The adapter is named after the set.

    uv run python -m core.dpo                        # trains runs/<profile>/adapters/refusal.bin (a GPU helps; a CPU takes minutes)
    uv run python -m core.dpo --eval                 # its refusal rates and held-out answers, without training
    uv run python -m core.dpo --set my-set.json --check   # a new set's requests and token counts, without training
    uv run python -m core.dpo --set my-set.json      # trains runs/<profile>/adapters/<its name>.bin
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from . import lora
from .config import DEFAULT_PROFILE, PROFILES, TrainConfig, pick_device, profile
from .data import load_data
from .model import GPT
from .sampling import SampleSettings, generate
from .tokenizers import BPETokenizer
from .trace_io import encode_tensors
from .train import Trainer

SETS_DIR = Path(__file__).resolve().parent.parent / "data" / "preferences"
DEFAULT_SET = SETS_DIR / "refusal.json"


def load_set(path: Path = DEFAULT_SET) -> dict:
    """A preference set: the refusal, the request template and its subjects, the activities to
    refuse ("unsafe") and to answer ("safe"), and how many of each list's last ones to hold out."""
    s = json.loads(Path(path).read_text())
    missing = {"name", "refusal", "template", "subjects", "unsafe", "safe", "heldOut"} - s.keys()
    if missing:
        raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", s["name"]):
        raise ValueError(f"{path}: name {s['name']!r} must be lower-case letters, digits and hyphens")
    if not s["refusal"].startswith(" "):
        s["refusal"] = " " + s["refusal"]       # an answer follows the request after a space, as a story would
    if "{subject}" not in s["template"] or "{activity}" not in s["template"]:
        raise ValueError(f"{path}: the template needs {{subject}} and {{activity}}")
    if not 0 < s["heldOut"] < min(len(s["unsafe"]), len(s["safe"])):
        raise ValueError(f"{path}: heldOut must leave activities of each kind to train on")
    s["subjects"] = [x if isinstance(x, dict) else {"subject": x} for x in s["subjects"]]
    return s


REFUSAL_SET = load_set()
REFUSAL = REFUSAL_SET["refusal"]


def refusal_start(refusal: str) -> str:
    """The refusal's first sentence, without its full stop: what counts as refusing (the UI's lib/lora.ts refuses())."""
    return refusal.split(".")[0]


def request(pset: dict, subject: dict, activity: str) -> str:
    """The template filled in for one subject, with the activity's {he} and {his} in the subject's words."""
    for k in ("he", "his"):
        activity = activity.replace("{" + k + "}", subject.get(k, k))
    return pset["template"].replace("{subject}", subject["subject"]).replace("{activity}", activity)


def prompt_sets(pset: dict = REFUSAL_SET) -> dict[str, list[dict]]:
    """Requests by split ("train", "heldOut") with their kind ("unsafe", "safe")."""
    out: dict[str, list[dict]] = {"train": [], "heldOut": []}
    first = pset["subjects"][0]
    for kind in ("unsafe", "safe"):
        acts = pset[kind]
        for i, act in enumerate(acts):
            split = "heldOut" if i >= len(acts) - pset["heldOut"] else "train"
            label = request({**pset, "template": "{activity}"}, first, act)
            out[split] += [{"prompt": request(pset, s, act), "kind": kind, "activity": label} for s in pset["subjects"]]
    return out


@torch.no_grad()
def story(model: GPT, tok: BPETokenizer, prompt: str, n: int, seed: int) -> str:
    """The base model's own continuation, cut at the end of a sentence."""
    model.eval()
    text = tok.decode(generate(model, tok.encode(prompt), n, SampleSettings(temperature=0.8), torch.Generator().manual_seed(seed)))
    text = " " + text.strip().split("\n")[0]
    end = max(text.rfind(c) for c in ".!?")
    return text[:end + 1] if end > 20 else text


def make_pairs(model: GPT, tok: BPETokenizer, prompts: list[dict], seed: int = 0, refusal: str = REFUSAL) -> list[dict]:
    pairs = []
    for i, p in enumerate(prompts):
        own = story(model, tok, p["prompt"], 40, seed + i)
        refuse = p["kind"] == "unsafe"
        pairs.append({**p, "chosen": refusal if refuse else own, "rejected": own if refuse else refusal})
    return pairs


def _sequences(tok: BPETokenizer, pairs: list[dict], key: str, block_size: int):
    """prompt + answer as inputs and targets, with only the answer's tokens as targets."""
    rows = []
    for p in pairs:
        a, b = tok.encode(p["prompt"]), tok.encode(p[key])
        ids = (a + b)[-(block_size + 1):]
        n_answer = min(len(b), len(ids) - 1)
        rows.append((ids, n_answer))
    T = max(len(ids) for ids, _ in rows) - 1
    x = torch.zeros(len(rows), T, dtype=torch.long)
    y = torch.full((len(rows), T), -1, dtype=torch.long)
    for i, (ids, n) in enumerate(rows):
        x[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        tgt[:len(ids) - 1 - n] = -1          # the prompt's own tokens are not scored
        y[i, :len(ids) - 1] = tgt
    return x, y


def log_probs(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log π(answer | prompt) per row: the sum of the answer tokens' log-probabilities."""
    logits, _ = model(x)
    lp = F.log_softmax(logits, -1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (y >= 0)).sum(-1)


def dpo_loss(pc: torch.Tensor, pr: torch.Tensor, rc: torch.Tensor, rr: torch.Tensor, beta: float):
    """The DPO loss and the implicit rewards: pc, pr are the policy's log-probabilities of the
    chosen and rejected answers, rc, rr the reference's."""
    reward_c, reward_r = beta * (pc - rc), beta * (pr - rr)
    return -F.logsigmoid(reward_c - reward_r).mean(), reward_c, reward_r


@torch.no_grad()
def score(model: GPT, tok: BPETokenizer, adapter: dict | None, scale: float, pairs: list[dict], beta: float) -> dict:
    """For each pair, log π(chosen), log π(rejected) with the adapter and without (the reference),
    and the implicit rewards. What the preferences panel shows, computed live."""
    device = next(model.parameters()).device
    model.eval()
    out = {}
    for key in ("chosen", "rejected"):
        x, y = (t.to(device) for t in _sequences(tok, pairs, key, model.cfg.block_size))
        out[f"ref_{key}"] = log_probs(model, x, y).tolist()
        with model.adapted(adapter, scale):
            out[f"policy_{key}"] = log_probs(model, x, y).tolist()
    margin = [beta * ((pc - rc) - (pr - rr)) for pc, rc, pr, rr in
              zip(out["policy_chosen"], out["ref_chosen"], out["policy_rejected"], out["ref_rejected"])]
    return {**out, "margin": margin, "beta": beta}


@torch.no_grad()
def refusal_rate(model: GPT, tok: BPETokenizer, adapter: dict | None, prompts: list[dict], refusal: str = REFUSAL) -> float:
    """The share of requests whose greedy answer starts with the refusal's first sentence."""
    model.eval()
    hits = 0
    with model.adapted(adapter):
        for p in prompts:
            ids = generate(model, tok.encode(p["prompt"]), 10, SampleSettings(temperature=0), torch.Generator())
            hits += tok.decode(ids).startswith(refusal_start(refusal))
    return hits / max(1, len(prompts))


def train(model: GPT, tok: BPETokenizer, pairs: list[dict], steps: int = 600, rank: int = 8, lr: float = 2e-3,
          beta: float = 0.5, nll: float = 0.5, batch: int = 8, seed: int = 0, base_step: int = 0, log=print,
          name: str = "refusal") -> lora.Adapter:
    device = next(model.parameters()).device
    ad = lora.new_adapter(model, name, ("q", "v", "mlp.down"), rank, seed, base_step)
    opt = lora.optimizer(ad, lr)
    rand = random.Random(seed)
    T = model.cfg.block_size
    seqs = {k: [t.to(device) for t in _sequences(tok, pairs, k, T)] for k in ("chosen", "rejected")}
    model.eval()
    with torch.no_grad():
        ref = {k: log_probs(model, *seqs[k]) for k in seqs}
    frozen = [p for p in model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    curve = []
    try:
        for step in range(1, steps + 1):
            ix = torch.tensor(rand.sample(range(len(pairs)), min(batch, len(pairs))), device=device)
            with model.adapted(ad.pairs()):
                pc = log_probs(model, *(t[ix] for t in seqs["chosen"]))
                pr = log_probs(model, *(t[ix] for t in seqs["rejected"]))
            loss, rc, rr = dpo_loss(pc, pr, ref["chosen"][ix], ref["rejected"][ix], beta)
            if nll:
                # The preferred answer's own likelihood, per token, so it rises rather than only
                # the rejected one falling.
                loss = loss - nll * (pc / (seqs["chosen"][1][ix] >= 0).sum(-1)).mean()
            for p in ad.parameters():
                p.grad = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
            opt.step()
            margin = (rc - rr).detach()
            acc = float((margin > 0).float().mean())
            if step % 10 == 0 or step == 1:
                curve.append({"step": step, "loss": loss.item(), "margin": float(margin.mean()), "accuracy": acc})
                log(f"step {step}: loss {loss.item():.3f}, reward margin {float(margin.mean()):.2f}, accuracy {acc:.2f}")
    finally:
        for p in frozen:
            p.requires_grad_(True)
    ad.info = {"kind": "dpo", "beta": beta, "nll": nll, "steps": steps, "lr": lr, "batch": batch, "curve": curve}
    ad.version = steps
    return ad


def check(pset: dict, tok: BPETokenizer, block_size: int, story_tokens: int = 40) -> bool:
    """Print a set's requests by split and their token counts; False if a request is too long to
    leave room for its answer in the context."""
    sets = prompt_sets(pset)
    n_refusal = len(tok.encode(pset["refusal"]))
    longest = max(len(tok.encode(q["prompt"])) for qs in sets.values() for q in qs)
    print(f"{pset['name']}: {pset.get('description', '')}")
    print(f"refusal {pset['refusal']!r}: {n_refusal} tokens; refused when an answer starts {refusal_start(pset['refusal'])!r}")
    for split, qs in sets.items():
        print(f"{split}: {sum(q['kind'] == 'unsafe' for q in qs)} to refuse, {sum(q['kind'] == 'safe' for q in qs)} to answer")
        for q in qs:
            print(f"  {q['kind']:6} {q['prompt']}  ({len(tok.encode(q['prompt']))} tokens)")
    room = longest + max(n_refusal, story_tokens)
    print(f"longest request {longest} tokens + answer {max(n_refusal, story_tokens)} = {room} of the {block_size + 1} the model reads")
    if room > block_size + 1:
        print("error: shorten the template, subjects or activities, or the requests lose their start")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_PROFILE, choices=list(PROFILES))
    p.add_argument("--device", default=None)
    p.add_argument("--set", type=Path, default=DEFAULT_SET, help="the preference set, a JSON file (default: the refusal set the model ships with)")
    p.add_argument("--name", default=None, help="the adapter's name (default: the set's name)")
    p.add_argument("--check", action="store_true", help="print the set's requests and token counts, without loading the model")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--eval", action="store_true", help="report the saved adapter's refusal rates and held-out answers, without training")
    p.add_argument("--nll", type=float, default=0.5,
                   help="weight of the preferred answer's likelihood, added to the DPO loss (0: DPO alone, which here "
                        "learns to push the rejected answers down without making the refusal likely)")
    args = p.parse_args()
    try:
        pset = load_set(args.set)
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}")
    name = args.name or pset["name"]
    data = load_data(need_train=False)
    if args.check:
        raise SystemExit(0 if check(pset, data.tokenizer, profile(args.config).block_size) else 1)
    t = Trainer(profile(args.config), TrainConfig(), data, pick_device(args.device))
    t.load_checkpoint()
    tok, model = data.tokenizer, t.model
    sets = prompt_sets(pset)
    out = t.run_dir / "adapters" / f"{name}.bin"
    if args.eval:
        from .lora import load_adapter

        report(model, tok, load_adapter(out, t.device), sets, pset["refusal"])
        return
    if not check(pset, tok, model.cfg.block_size):
        raise SystemExit(1)
    pairs = {split: make_pairs(model, tok, ps, seed=1000 * i, refusal=pset["refusal"]) for i, (split, ps) in enumerate(sets.items())}
    ad = train(model, tok, pairs["train"], args.steps, args.rank, args.lr, args.beta, args.nll, base_step=t.step_num, name=name)
    rates = report(model, tok, ad, sets, pset["refusal"])
    ad.info.update({"refusal": pset["refusal"], "set": pset["name"], "description": pset.get("description", ""),
                    "pairs": [{**q, "split": s} for s, qs in pairs.items() for q in qs], "rates": rates})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(encode_tensors("adapter", {"weights": ad.weights()}, ad.header()))
    print(f"wrote {out}: restart the server to load it")


def report(model: GPT, tok: BPETokenizer, ad: lora.Adapter, sets: dict[str, list[dict]], refusal: str = REFUSAL) -> dict:
    """Refusal rates by split and kind, with the adapter off and on, and every held-out answer."""
    rates = {}
    for split in ("train", "heldOut"):
        for kind in ("unsafe", "safe"):
            ps = [q for q in sets[split] if q["kind"] == kind]
            rates[f"{split}.{kind}"] = {"base": refusal_rate(model, tok, None, ps, refusal),
                                        "tuned": refusal_rate(model, tok, ad.pairs(), ps, refusal)}
    print(json.dumps(rates, indent=1))
    for q in sets["heldOut"]:
        with torch.no_grad(), model.adapted(ad.pairs()):
            ids = generate(model, tok.encode(q["prompt"]), 16, SampleSettings(temperature=0), torch.Generator())
        print(f"  {q['kind']:6} {q['prompt']} →{tok.decode(ids)!r}")
    return rates


if __name__ == "__main__":
    main()
