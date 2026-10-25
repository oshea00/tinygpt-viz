"""The logit lens on the server, for large vocabularies: decode every residual snapshot with
the final layer norm and unembedding, and keep a compact summary (top guesses per cell, and
the probability of the actual next token). With a small vocabulary the UI computes it itself."""

import torch

from .model import GPT, layer_norm, softmax


@torch.no_grad()
def logit_lens(model: GPT, ids: list[int], k: int = 8) -> dict:
    ids = ids[-model.cfg.block_size:]
    device = next(model.parameters()).device
    snaps: dict[str, torch.Tensor] = {}

    def keep(stage_id: str, x: torch.Tensor) -> torch.Tensor:
        if stage_id == "embed.sum" or stage_id.endswith(("resid_attn", "resid_mlp")):
            snaps[stage_id] = x[0]
        return x

    was_training = model.training
    model.eval()
    with model.tapped(keep):
        model(torch.tensor([ids], device=device))
    model.train(was_training)
    names = list(snaps)                       # execution order
    x = torch.stack([snaps[n] for n in names])            # [S, T, d]
    probs = softmax(layer_norm(x, model.ln_f_g, model.ln_f_b) @ model.unembed_matrix())   # [S, T, V]
    top_p, top_i = probs.topk(k, dim=-1)
    targets = torch.tensor(ids[1:] + [ids[-1]], device=device)
    p_actual = probs.gather(-1, targets.view(1, -1, 1).expand(len(names), -1, 1)).squeeze(-1)
    p_actual[:, -1] = -1                                  # no next token at the last position
    return {"snapshots": names, "topIds": top_i.tolist(), "topP": top_p.tolist(), "pActual": p_actual.tolist()}
