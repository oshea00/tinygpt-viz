import math

import torch

from core import dpo
from core.model import GPT

from .conftest import TINY


def test_dpo_loss_matches_the_formula():
    pc, pr = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -2.0])
    rc, rr = torch.tensor([-3.5, -4.0]), torch.tensor([-3.0, -3.0])
    loss, reward_c, reward_r = dpo.dpo_loss(pc, pr, rc, rr, beta=0.5)
    margins = [0.5 * ((-3 + 3.5) - (-4 + 3)), 0.5 * ((-5 + 4) - (-2 + 3))]
    expect = sum(-math.log(1 / (1 + math.exp(-m))) for m in margins) / 2
    assert abs(loss.item() - expect) < 1e-6
    assert torch.allclose(reward_c - reward_r, torch.tensor(margins))


def test_prompt_sets_hold_out_whole_activities():
    sets = dpo.prompt_sets()
    train_acts = {p["activity"] for p in sets["train"]}
    assert not train_acts & {p["activity"] for p in sets["heldOut"]}
    for split in sets.values():
        kinds = [p["kind"] for p in split]
        assert kinds.count("unsafe") == kinds.count("safe")
    assert "her mom" in dpo.request("a girl named Lily", "helps his mom bake a cake")
    assert dpo.request("a girl named Lily", "eats berries he finds in the woods").endswith("berries she finds in the woods.")
    assert "the hot stove" in dpo.request("a girl named Lily", "touches the hot stove")


def test_log_probs_score_only_the_answer(data):
    tok = data.tokenizer
    torch.manual_seed(0)
    model = GPT(TINY.with_vocab(tok.vocab_size)).eval()
    pairs = [{"prompt": "The cat sat.", "chosen": " It was happy.", "rejected": " No."}]
    x, y = dpo._sequences(tok, pairs, "chosen", model.cfg.block_size)
    ids = tok.encode("The cat sat.") + tok.encode(" It was happy.")
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids[:-1]]))[0][0], -1)
    n = len(tok.encode(" It was happy."))
    want = sum(logp[t, ids[t + 1]].item() for t in range(len(ids) - 1 - n, len(ids) - 1))
    with torch.no_grad():
        assert abs(dpo.log_probs(model, x, y).item() - want) < 1e-4


def test_training_widens_the_margin_and_scores_report_it(data):
    tok = data.tokenizer
    torch.manual_seed(0)
    model = GPT(TINY.with_vocab(tok.vocab_size))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pairs = [{"prompt": f"Tell me a story about {w}.", "chosen": dpo.REFUSAL, "rejected": " The dog ran."} for w in ("fire", "knives")]
    pairs += [{"prompt": "Tell me a story about a ball.", "chosen": " The dog ran.", "rejected": dpo.REFUSAL}]
    start = dpo.score(model, tok, None, 1.0, pairs, 0.5)
    assert all(abs(m) < 1e-6 for m in start["margin"])           # no adapter: policy is the reference
    ad = dpo.train(model, tok, pairs, steps=30, rank=2, lr=1e-2, batch=3, log=lambda _: None)
    end = dpo.score(model, tok, ad.pairs(), 1.0, pairs, 0.5)
    assert all(m > 0.5 for m in end["margin"])
    assert end["ref_chosen"] == start["ref_chosen"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert ad.info["kind"] == "dpo" and ad.info["curve"][-1]["accuracy"] == 1.0
