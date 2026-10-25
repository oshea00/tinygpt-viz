"""What each traced stage is: its kind, where its inputs come from, the weights it uses,
its formula and a lesson paragraph. The UI builds the pipeline map, stepper and
inspector from this, so nothing about the network's shape is hand-coded there.
"""

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .config import ModelConfig


@dataclass
class StageMeta:
    id: str
    kind: str
    layer: int | None = None
    head: int | None = None
    inputs: list[str] = field(default_factory=list)
    weights: list[str] = field(default_factory=list)
    formula: str = ""
    explain: str = ""

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _resid_before(l: int) -> str:
    return "embed.sum" if l == 0 else f"L{l - 1}.resid_mlp"


@lru_cache(maxsize=4096)
def describe(stage_id: str, cfg: ModelConfig) -> StageMeta:
    L = cfg.n_layer
    if stage_id == "embed.tok":
        return StageMeta(stage_id, "embed", weights=["wte"], formula=r"e_t = W_{te}[\text{token}_t]",
                         explain="Each token id picks one row of the token embedding table. "
                                 "This is a lookup, not a multiply: the row is the token's starting vector.")
    if stage_id == "embed.pos":
        return StageMeta(stage_id, "embed", weights=["wpe"], formula=r"p_t = W_{pe}[t]",
                         explain="Position t picks row t of the position table. Without it the model "
                                 "could not tell 'ab' from 'ba': attention by itself ignores order.")
    if stage_id == "embed.sum":
        return StageMeta(stage_id, "add", inputs=["embed.tok", "embed.pos"], formula=r"x_0 = e_t + p_t",
                         explain="Token and position vectors are added to make the residual stream x₀, "
                                 "the shared workspace every later layer reads from and writes to.")
    if stage_id == "ln_f":
        return StageMeta(stage_id, "norm", inputs=[f"L{L - 1}.resid_mlp"], weights=["ln_f_g", "ln_f_b"],
                         formula=r"\hat x = \gamma \odot \frac{x - \mu}{\sigma} + \beta",
                         explain="A final layer norm before reading out predictions.")
    if stage_id == "logits":
        w = "wte" if cfg.tied else "W_unembed"
        return StageMeta(stage_id, "logits", inputs=["ln_f"], weights=[w], formula=r"z = \hat x\, W_U",
                         explain="Each row becomes one score per vocabulary entry: a prediction of the "
                                 "next token at that position. Every position predicts, which is how "
                                 "training learns from all positions at once.")
    if stage_id == "probs":
        return StageMeta(stage_id, "probs", inputs=["logits"],
                         formula=r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
                         explain="Softmax turns each row of scores into a probability distribution "
                                 "that sums to 1. The last row is the prediction for the next token.")
    if stage_id == "loss":
        return StageMeta(stage_id, "loss", inputs=["logits"],
                         formula=r"L = -\frac{1}{T}\sum_t \log p_t[y_t]",
                         explain="Cross-entropy: the average surprise at the actual next token. "
                                 "Its gradient with respect to the logits is simply p − y.")

    m = re.fullmatch(r"L(\d+)\.(.+)", stage_id)
    if not m:
        raise KeyError(stage_id)
    l, rest = int(m.group(1)), m.group(2)
    P = f"L{l}"
    ln = r"\hat x = \gamma \odot \frac{x - \mu}{\sigma} + \beta"
    simple = {
        "ln1": StageMeta(stage_id, "norm", l, inputs=[_resid_before(l)], weights=[f"{P}.ln1_g", f"{P}.ln1_b"],
                         formula=ln, explain="Layer norm: each row is shifted to mean 0 and scaled to "
                                             "standard deviation 1, then scaled and shifted by learned γ and β."),
        "attn.concat": StageMeta(stage_id, "concat", l, inputs=[f"{P}.attn.H{h}.mix" for h in range(cfg.n_head)],
                                 formula=r"[\,\mathrm{head}_1 \mid \cdots \mid \mathrm{head}_H\,]",
                                 explain="The heads' outputs are placed side by side again."),
        "attn.out": StageMeta(stage_id, "linear", l, inputs=[f"{P}.attn.concat"], weights=[f"{P}.W_o", f"{P}.b_o"],
                              formula=r"a = [\ldots] W_O + b_O",
                              explain="The output projection mixes the heads together into one vector "
                                      "to write back to the residual stream."),
        "resid_attn": StageMeta(stage_id, "add", l, inputs=[_resid_before(l), f"{P}.attn.out"],
                                formula=r"x \leftarrow x + a",
                                explain="Residual add: attention's output is added to the stream rather "
                                        "than replacing it."),
        "ln2": StageMeta(stage_id, "norm", l, inputs=[f"{P}.resid_attn"], weights=[f"{P}.ln2_g", f"{P}.ln2_b"],
                         formula=ln, explain="Layer norm before the MLP."),
        "mlp.up": StageMeta(stage_id, "linear", l, inputs=[f"{P}.ln2"], weights=[f"{P}.W_up", f"{P}.b_up"],
                            formula=r"u = \hat x W_{up} + b_{up}",
                            explain=f"Each position's vector is expanded to {cfg.d_mlp} hidden neurons."),
        "mlp.gelu": StageMeta(stage_id, "activation", l, inputs=[f"{P}.mlp.up"],
                              formula=r"\mathrm{GELU}(u) = u\,\Phi(u)",
                              explain="GELU passes positive values almost unchanged and squashes negative "
                                      "ones towards zero. Without a non-linearity the MLP would be a single matrix."),
        "mlp.down": StageMeta(stage_id, "linear", l, inputs=[f"{P}.mlp.gelu"], weights=[f"{P}.W_down", f"{P}.b_down"],
                              formula=r"m = \mathrm{GELU}(u) W_{down} + b_{down}",
                              explain=f"The {cfg.d_mlp} neurons are projected back to {cfg.d_model} dimensions."),
        "resid_mlp": StageMeta(stage_id, "add", l, inputs=[f"{P}.resid_attn", f"{P}.mlp.down"],
                               formula=r"x \leftarrow x + m",
                               explain="Residual add: the MLP's output is added to the stream."),
    }
    if rest in simple:
        return simple[rest]
    if rest in ("q", "k", "v"):
        name = {"q": "query", "k": "key", "v": "value"}[rest]
        R = rest.upper()
        return StageMeta(stage_id, "linear", l, inputs=[f"{P}.ln1"], weights=[f"{P}.W_{rest}", f"{P}.b_{rest}"],
                         formula=rf"{R} = \hat x W_{R} + b_{R}",
                         explain=f"The {name} projection. Queries ask, keys advertise, values carry the content.")

    hm = re.fullmatch(r"attn\.H(\d+)\.(\w+)", rest)
    if not hm:
        raise KeyError(stage_id)
    h, part = int(hm.group(1)), hm.group(2)
    H = f"{P}.attn.H{h}"
    dh = cfg.d_head
    if part in ("q", "k", "v"):
        return StageMeta(stage_id, "split", l, h, inputs=[f"{P}.{part}"],
                         formula=rf"{part.upper()}_{h} = {part.upper()}[:, {h * dh}:{(h + 1) * dh}]",
                         explain=f"Head {h} takes its own band of {dh} columns.")
    heads = {
        "scores": StageMeta(stage_id, "attn_scores", l, h, inputs=[f"{H}.q", f"{H}.k"],
                            formula=rf"S = \frac{{Q_h K_h^\top}}{{\sqrt{{{dh}}}}}",
                            explain="Every query is compared with every key by a dot product: how well "
                                    "does what position i is looking for match what position j offers?"),
        "masked": StageMeta(stage_id, "mask", l, h, inputs=[f"{H}.scores"],
                            formula=r"S_{ij} = -\infty \text{ for } j > i",
                            explain="The causal mask: a position may not look at later positions, "
                                    "since at generation time they do not exist yet."),
        "weights": StageMeta(stage_id, "softmax", l, h, inputs=[f"{H}.masked"],
                             formula=r"A = \mathrm{softmax}(S)",
                             explain="Each row becomes weights that sum to 1: the attention pattern."),
        "mix": StageMeta(stage_id, "attn_mix", l, h, inputs=[f"{H}.weights", f"{H}.v"],
                         formula=r"\mathrm{head}_h = A\,V_h",
                         explain="Each position's output is a weighted average of the value vectors "
                                 "of the positions it attends to."),
    }
    return heads[part]


def stage_order(cfg: ModelConfig) -> list[str]:
    """Stage ids in execution order (the order the model taps them)."""
    ids = ["embed.tok", "embed.pos", "embed.sum"]
    for l in range(cfg.n_layer):
        P = f"L{l}"
        ids += [f"{P}.ln1", f"{P}.q", f"{P}.k", f"{P}.v"]
        for h in range(cfg.n_head):
            ids += [f"{P}.attn.H{h}.{x}" for x in ("q", "k", "v", "scores", "masked", "weights", "mix")]
        ids += [f"{P}.attn.concat", f"{P}.attn.out", f"{P}.resid_attn", f"{P}.ln2",
                f"{P}.mlp.up", f"{P}.mlp.gelu", f"{P}.mlp.down", f"{P}.resid_mlp"]
    return ids + ["ln_f", "logits", "probs", "loss"]
