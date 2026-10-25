# TinyGPT Visualizer

Step through one forward and backward pass of a real, trained, character-level GPT
(100-token context), seeing every number at every stage, and train it live. Design: [PROJECT.md](PROJECT.md).

Implemented: milestones 1–12 (instrumented core, tracer with gradients, WebSocket
bridge, pipeline map + stepper + tensor inspector, training controls and the one-step view,
attention views, generation, residual stream and logit lens, network atlas,
interventions, lessons, checkpoint replay, manual backward mode, static demo, and the
BPE model with its concept explorer).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
uv sync                          # Python deps (PyTorch with CUDA 12.8 wheels on Linux/Windows)
uv run python -m core.data       # fetch a 10 MB TinyStories slice, build the character vocab
cd ui && npm install && npm run build && cd ..
```

## Train

```sh
uv run python -m core.train --config char-small --steps 20000
```

Configs are named by how the model reads text, then by size:

| Config | Reads | Params | Shape | Notes |
| --- | --- | --- | --- | --- |
| `char-tiny` | characters (64) | 32k | 2 layers × 2 heads, d=32 | every number readable |
| `char-small` | characters (64) | 215k | 4 × 4, d=64 | the default; val loss ≈ 0.95 in ~7 min on a GPU (~30 on CPU) |
| `char-large` | characters (64) | 1.2M | 6 × 8, d=128 | better text; the atlas needs zooming |
| `bpe-small` | BPE tokens (4,096) | 5.8M | 6 × 8, d=256, tied embeddings | most words are one token; for the concept explorer |

The earlier names `tiny`, `default`, `larger` and `concepts` still work; their checkpoint
directories are renamed to the new names the first time they are used. Checkpoints land in `runs/<config>/` on a log-spaced schedule for later replay.

```sh
uv run python -m core.data --bpe     # Concepts corpus: 250 MB slice, BPE tokenizer, ~62M tokens
uv run python -m core.train --config bpe-small --steps 20000 --lr 6e-4 --batch-size 128   # ~22 min on a 4090
uv run python -m core.server --config bpe-small
```

## Run the app

```sh
uv run python -m core.server --config char-small # loads the latest checkpoint; --fresh for random weights
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
- **Concept explorer** (best with `--config bpe-small`): capture each tagged word of a prompt set
  (editable templates such as "Lily saw a {animal} in the garden.") at every residual snapshot, from
  the trained model and the untrained step-0 model as a baseline. Then: similarity matrices,
  PCA maps (click a point to open its trace), nearest neighbours in the embedding table and in
  context, context drift, linear probes tested on held-out words (with shuffled-label and untrained
  baselines), concept directions with steering during generation, attribution (attention
  rollout, gradient × input, occlusion), and the top-activating contexts of any MLP neuron.
- **One training step**: an AdamW update on the prompt itself, with p(correct) before and after,
  and weight, gradient and actual update side by side.
- **Next token and generation** (right): the full next-token distribution, reshaped live by
  temperature, top-k and top-p (excluded tokens greyed). Sample one character or run auto; generated
  text is coloured by how likely each choice was, the 100-character window is boxed as it slides,
  and clicking a generated character reloads the trace that produced it.
- **Training panel**: play, pause, single-step, reset, learning rate, batch size, and a replay
  slider over saved checkpoints (with "play through") to watch the model learn.
- **Lessons** (header menu): nine guided lessons that drive the views, with questions checked
  against what you click. Lessons are plain JSON in `ui/src/lessons/`; new ones need no code.
- **Manual backward**: in any stage's gradient view, the inspector shows the backward rule and
  checks autograd against a hand-written backward pass (`core/manual_backward.py`).

## Static demo

An inference-only build that runs the model in the browser (TypeScript, in a Web Worker) with no
Python, for sharing:

```sh
uv run python -m core.export --config char-small # latest checkpoint → ui/demo-public/demo/
cd ui && npm run build:demo                      # → ui/dist-demo/, a static site (about 4.5 MB)
```

Serve `ui/dist-demo/` from any static host. Training, interventions and gradients need the server;
the demo supports the character profiles only.

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
