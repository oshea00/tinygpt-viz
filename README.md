# TinyGPT Visualizer

Step through one forward and backward pass of a real, trained GPT with a 4,096-token BPE
vocabulary (100-token context), seeing every number at every stage, and train it live. Design: [PROJECT.md](PROJECT.md).

**[The guide](docs/README.md)** explains every view and how to use it, for learners and for
instructors writing lessons. In the app, the **help** button explains the open tab, and each panel's **?** explains that panel.

Implemented: milestones 1–12 (instrumented core, tracer with gradients, WebSocket
bridge, pipeline map + stepper + tensor inspector, training controls and the one-step view,
attention views, generation, residual stream and logit lens, network atlas,
interventions, lessons, checkpoint replay, manual backward mode, and the concept explorer).
The character-level models and the static browser demo were removed on 2026-10-26; see PROJECT.md.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
uv sync                          # Python deps (PyTorch with CUDA 12.8 wheels on Linux/Windows)
uv run python -m core.data       # fetch a 250 MB TinyStories slice, train the BPE tokenizer, ~62M tokens
cd ui && npm install && npm run build && cd ..
```

## Train

```sh
uv run python -m core.train --steps 20000   # ~22 min on a 4090
```

There is one config, `bpe-small` (the default, so `--config` can be left out):

| Config | Reads | Params | Shape | Notes |
| --- | --- | --- | --- | --- |
| `bpe-small` | BPE tokens (4,096) | 5.8M | 6 layers × 8 heads, d=256, tied embeddings | most words are one token |

Checkpoints land in `runs/bpe-small/` on a log-spaced schedule for later replay.

Every run begins by timing a few real steps on the device it found and printing an estimate,
so a run that is going to take all night says so in the first minute. `--dry-run` stops after
that report; `--log-every N` sets how often the progress line appears (default every 10 steps).
The quoted timings below are for a desktop GPU: a power-capped laptop GPU can be far slower,
and the report flags one when it sees it.

```sh
uv run python -m core.train --steps 20000 --dry-run
```
```
bpe-small: 6L 8H d256 vocab 4,096 ctx 100, batch 128
device: cuda - NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.4 GB VRAM
  ! memory clock 810 of 5501 MHz at 10 W - this GPU is power-capped, so bandwidth-bound work runs several times slower than its rating
measured: 4,011 ms/step, 3,192 tok/s, peak 4.2 GB VRAM
estimate: 22.9h for 20,000 steps (22.3h training + 34m eval + 2m sampling), ~2.2 GB of checkpoints
```

## Run the app

```sh
uv run python -m core.server           # loads the latest checkpoint; --fresh for random weights
uv run python -m core.server --train   # also enables training, replay and the training step tab
```

Open http://localhost:8765 (from WSL2, in the Windows browser). For UI development, run
`npm run dev` in `ui/` alongside the server; Vite proxies `/ws` to port 8765.

- **Prompt box**: every keystroke re-runs the model; click a token to follow its row through every stage.
- **Pipeline map** (left): the residual spine with attention and MLP branches, one live thumbnail per stage.
- **Stepper**: ← / → walk forward through every stage, past the loss, and back up through the gradients.
- **Tensor inspector**: any stage's values or gradients as a heatmap with a visible colour scale;
  click a cell to see how it was computed (dot products, layer-norm statistics, softmax terms, gradient sums).
- **Attention**: arcs over the text from the selected position, the masked T×T matrix per head,
  a Q·K explorer (one query against every earlier key: products, dot, scaled score, weight), and a
  head gallery with pattern badges and names you can give each head.
- **Residual stream**: one position's vector at every snapshot (or what each sublayer writes),
  write sizes per sublayer and position, and the gradient highway: at each residual add, how much
  gradient passes straight through and how much comes back through the sublayer.
- **Logit lens**: the final layer norm and unembedding applied to every residual snapshot, as a
  snapshots × positions grid of top guesses, with p(actual next token) through the layers.
- **Network atlas**: every weight and this prompt's activations as one zoomable bitmap (WebGL2),
  one pixel per number; labels appear as you zoom and then the numbers themselves. Modes: values,
  gradients, weight × input for the selected position, diff against a pinned snapshot, and the
  most recent weight change during training. Click any pixel to open it in the inspector.
- **Interventions**: ablate a head or MLP neuron (zero or mean), switch off a block's attention or
  MLP, force a head to attend to one position, set any activation cell, or shuffle the position
  embeddings. Every view then shows the changed model; the interventions tab compares it with the
  original run (loss, next-token distribution, p(actual) per position, and where the change spreads
  stage by stage), and the inspector can show any stage as its change. Training is never affected.
- **Concept explorer**: capture each tagged word of a prompt set
  (editable templates such as "Lily saw a {animal} in the garden.") at every residual snapshot, from
  the trained model and the untrained step-0 model as a baseline. Then: similarity matrices,
  PCA maps (click a point to open its trace), nearest neighbours in the embedding table and in
  context, context drift, linear probes tested on held-out words (with shuffled-label and untrained
  baselines), concept directions with steering during generation, attribution (attention
  rollout, gradient × input, occlusion), and the top-activating contexts of any MLP neuron.
- **One training step**: an AdamW update on the prompt itself, with p(correct) before and after,
  and weight, gradient and actual update side by side.
- **Next token and generation** (right): the full next-token distribution, reshaped live by
  temperature, top-k and top-p (excluded tokens greyed). Sample one token or run auto; generated
  text is coloured by how likely each choice was, the 100-token window is boxed as it slides,
  and clicking a generated token reloads the trace that produced it. **KV cache** mode reuses
  every earlier position's keys and values and computes only the new one: reused rows are greyed in
  every view, and the panel counts the multiply-adds saved, until the window slides and every
  cached key and value goes stale.
- **Training panel** (with `--train`): play, pause, single-step, reset, learning rate, batch size, and a replay
  slider over saved checkpoints (with "play through") to watch the model learn.
- **Lessons** (header menu): ten guided lessons that drive the views, with questions checked
  against what you click. Lessons are plain JSON in `ui/src/lessons/`; new ones need no code.
- **Manual backward**: in any stage's gradient view, the inspector shows the backward rule and
  checks autograd against a hand-written backward pass (`core/manual_backward.py`).

## Tests

```sh
uv run pytest          # model, tracer parity with autograd, trace format, server protocol; refreshes UI fixtures
cd ui && npm test      # frame decoding and cell provenance against those fixtures
cd ui && npm run check # svelte-check type checking
```

## Layout

```
core/    model.py (explicit-ops GPT with tap points), stages.py (stage registry), tracer.py,
         trace_io.py (binary frames), train.py, scheduler.py (single model thread), server.py
ui/src/  bridge.ts, state/, lib/ (colour maps, scales, heatmaps, provenance, pipeline layout), views/
tests/   pytest suites; ui/src/fixtures/ holds frames they generate for Vitest
```
