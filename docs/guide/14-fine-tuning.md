# Fine-tuning: LoRA and preference tuning

The model learned from 270,000 stories. **Fine-tuning** trains it a little more, on far fewer examples, to change what it does. The **fine-tune** tab, after **beliefs** in the tab bar, does this without touching the model's weights. It uses low-rank adapters, the method most fine-tuning of large models uses today. The tab has two sections:

- **LoRA** trains an adapter here, live, on 64 stories about dragons, and shows how far the model tilts towards them.
- **Preference tuning (DPO)** loads an adapter trained offline on preference pairs, which declines requests for stories about unsafe play. Everything about it is measured live.

## Adapters

[LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) leaves a weight W frozen and adds two thin matrices beside it: A, d_in × r, and B, r × d_out, for a small **rank** r. The stage computes

```math
y = x W + b + (x A)\,B \cdot s
```

Only A and B train. B starts at zero, so a new adapter is an exact no-op, and training starts from the model as it was. Their product ΔW = A B is the change the adapter makes to W. It is a full matrix, but its rank is at most r: r (d_in + d_out) numbers describe it, instead of d_in × d_out. With r = 4 on the query and value projections of all six layers, that is 24,576 numbers, 0.42% of the model. The strength s scales B: 0 is the base model, 1 the trained adapter.

An adapter lives beside the model, not in it. Checkpoints, replay and the training panel never see it. Training one needs neither `--train` nor `?train=1`, because the model's own weights stay as they were. Every command chooses its adapter, the way interventions work, so on a shared class server one learner's adapter never changes what another sees. On a server, though, adapters are stored by name and shared by every page connected to it: a second learner, or a second tab of your own, that trains `dragon` replaces the first one's, and the other page's views use the new one from their next run. Only one adapter trains at a time. With the in-browser backend, each tab runs its own model and keeps its own adapters.

**switch on everywhere** runs every view with the adapter. The header shows **adapter … on**; click it to switch the adapter off. While it is on:

- The pipeline map gains an **adapter on …** row in each block, before the stages it adds to: `L3.q.lora`, `L3.v.lora` and so on.
- The inspector shows an adapter stage like any other. Its provenance squeezes the input row to r numbers through A, then takes the dot product with a column of B. The stage the term is added to (`L3.q`) lists it as an extra line.
- Attention, the logit lens, the residual stream, concepts and generation all run the adapted model.
- Manual-backward mode has no rule for an adapter stage. Switch the adapter off to use it.

An adapter records the checkpoint it was trained on. If replay loads another step, the tab warns you: an adapter only means something on the weights it was trained with.

## LoRA: tilting the model towards a topic

![The LoRA section after training: the loss and p(dragon) charts, generation with and without the adapter, and the adapter's matrices](../images/finetune-lora.png)

The training set is 64 TinyStories stories that mention “dragon” at least three times (`uv run python -m core.lora --topic dragon` writes it to `ui/src/finetune/`). **show / edit** shows them. Paste stories about another topic, one per paragraph, and change the topic word to train on something else.

- **adapt**: the stages to add an adapter to, in every layer: `q`, `k`, `v`, `attn.out`, `mlp.up`, `mlp.down`. The default, `q` and `v`, is the original paper's choice.
- **rank r**, **steps** and **learning rate**. The defaults (4, 150, 0.003) take about half a minute on a GPU. Training uses AdamW on A and B only, with batches of 8 windows of up to 100 tokens.
- **train the adapter** streams progress. The left chart is the loss: each step's batch (grey), the loss on the stories (blue), and the loss on 16 ordinary validation stories (orange). The right chart is the mean probability of the topic word after eight neutral openers that end where a noun goes, such as “Once upon a time, there was a”. For dragons it rises from about 0.2% to about 20%, while the ordinary stories' loss rises a little: the price of the tilt.

The browser's CPU, without WebGPU, takes about 12 seconds a step, so the default 150 steps take about half an hour; the tab says so before you start. `dragon-ready`, trained in advance with the same settings (`uv run python -m core.lora --topic dragon --train`), ships with the model. If you didn't train one here, the tab uses it.

**Generate with and without it** continues a prompt twice, with the same random draws, so where the texts differ, the adapter is the cause. Mentions of the topic word are highlighted and counted.

![The inspector on an adapter stage, with its provenance through A and B](../images/finetune-inspector.png)

**Where it plugs in** shows one adapted stage:

- A, B and ΔW = A·B as heatmaps.
- ΔW's singular values, which are at most r: how much the adapter uses each of its directions.
- ‖ΔW‖ / ‖W‖ for every adapted stage. Click a bar to show that stage.

## Preference tuning with DPO: teaching a refusal

![The preference-tuning section: training curves, held-out requests answered with and without the adapter, and the scored pairs](../images/finetune-preferences.png)

Preference tuning trains a model on which of two answers people prefer. The first recipe, [RLHF](https://arxiv.org/abs/2203.02155) (reinforcement learning from human feedback), fits a reward model to those choices, then optimises the model against it with reinforcement learning (PPO). [DPO](https://arxiv.org/abs/2305.18290), direct preference optimisation, is now the common choice: it reaches the same optimum with an ordinary loss, and no reward model or reinforcement learning. For a request x with a preferred answer y_w and a rejected one y_l:

```math
\mathcal L = -\log \sigma\Big(\beta\big[\log\tfrac{\pi(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log\tfrac{\pi(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\big]\Big)
```

π is the model being tuned and π_ref the frozen model it started from. Here π is this model with a LoRA adapter and π_ref is the same model with the adapter off.

`uv run python -m core.dpo` trains the `refusal` adapter from its preference set, `data/preferences/refusal.json`. It takes about two minutes on a GPU and writes `runs/<profile>/adapters/refusal.bin`. Other sets train other preference adapters, and when the model has more than one, the section has an **adapter** menu: see [Preference adapters](../instructors/preference-adapters.md).

- **Requests**: “Tell me a story about a boy named Tom who plays with matches.”, for 16 unsafe activities and 16 safe look-alikes (“…who plays with blocks”), each with Tom and with Lily. The last four activities of each list are held out.
- **Pairs**: for an unsafe request, the preferred answer is the refusal “I'm sorry, I can't tell that story. It is not safe.” and the rejected one is the base model's own story. For a safe request the roles swap, so the adapter has to learn which requests to decline, not just to decline.
- **Training**: rank 8 on `q`, `v` and `mlp.down`, β = 0.5, 600 steps. On this model DPO alone learned to make the rejected answers very unlikely without making the refusal likely, so it never refused. The script therefore adds the preferred answer's log-likelihood per token, weighted 0.5, a common fix ([Pang et al., 2024](https://arxiv.org/abs/2404.19733)). `--nll 0` trains with DPO alone.
- **Result**: the trained requests are all refused. 7 of 8 held-out unsafe requests are refused, and no held-out safe request is.

The section measures the adapter live:

- **answer them, with and without the adapter**: every held-out request, greedily. Look for **over-refusal**, a safe request refused, and for half-refusals: “I'm sorry, I can't wait to see you…”. A small model learns the refusal's first words most firmly.
- **score every pair**: for every pair, log p of both answers with the adapter and without. Δ log p is how much more likely the adapter makes an answer. β times it is DPO's **implicit reward**: the reward model RLHF would train is folded into the policy instead. The margin is what the loss pushed up.
- **find the direction**: [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) found that refusal in chat models is carried by one direction in the residual stream. The panel reads the stream at each training request's last token with the adapter on, and takes unsafe minus safe at every snapshot. Then it tests the direction at the selected snapshot, the middle of the network by default:
  - **subtract it** while the tuned model answers an unsafe request: on this model it then tells the story.
  - **add it** while it answers a safe one: it then refuses.
  - **add it to the base model**: the base model never trained on the refusal, so it doesn't refuse. But it stops telling a story and talks as "I" about being careful: "I will be careful and not let you get hurt." The direction carries the decision to answer that way; the adapter's later layers word it as the exact refusal. (A difference of means from the base model's own vectors only changes the story's details.)

## Training adapters yourself

```sh
uv run python -m core.lora --topic dragon            # the story set, ui/src/finetune/dragon-stories.json
uv run python -m core.lora --topic dragon --train    # runs/bpe-small/adapters/dragon-ready.bin
uv run python -m core.dpo                           # runs/bpe-small/adapters/refusal.bin
uv run python -m core.dpo --set my-set.json         # runs/bpe-small/adapters/<the set's name>.bin
```

To check that an adapter does what it should, without the browser, evaluate it on the GPU in batches:

```sh
uv run python -m core.lora --eval dragon-ready                                # topic and related words, by strength
uv run python -m core.lora --eval dragon-ready --prompt "Once upon a time," --samples 200
uv run python -m core.dpo --eval                                            # refusal rates and held-out answers
```

`core.lora --eval` samples 50 continuations (60 tokens, temperature 0.8) of each opener at strengths 0 to 2. It reports the share that mention the topic word and some related ones (monster, cave, knight…), and p(topic) after the openers. `--train` takes `--steps`, `--lr`, `--rank`, `--targets` and `--name`, so different settings can be trained and compared.

The server loads every adapter in `runs/<profile>/adapters/` when it starts. `core.hub push` uploads them with the checkpoints, and `core.hub push-web` (through `core.web_export`) puts them beside the browser's model, which loads the ones listed in `model.json`. A checkout that already has checkpoints but no adapters fetches them on the server's next start. `core.hub push-adapter --name <name>` uploads a single adapter, and `core.hub pull-adapters` fetches new ones ([Preference adapters](../instructors/preference-adapters.md#5-publish-it)).
