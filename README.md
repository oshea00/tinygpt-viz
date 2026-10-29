# TinyGPT Visualizer

Step through one forward and backward pass of a real, trained GPT with a 4,096-token BPE
vocabulary (100-token context), seeing every number at every stage, and train it live. Design: [PROJECT.md](PROJECT.md).

**[The guide](docs/README.md)** explains every view and how to use it, for learners and for
instructors writing lessons. In the app, the **help** button explains the open tab, and each panel's **?** explains that panel.

## Walkthrough
[Walkthrough Video - Inside the Network](https://github.com/user-attachments/assets/7179244e-de67-4ddc-81fe-318dc4194406)

Features: an instrumented core with a tracer that records every stage and its gradient, a
pipeline map with a stepper and tensor inspector, attention views, generation with a KV cache,
the residual stream and logit lens, a network atlas, interventions, guided lessons, live
training with checkpoint replay, a manual backward mode, a concept explorer, and fine-tuning with LoRA and preference tuning (DPO).

## Run it

**In the browser, with nothing to install:** open **<https://oshea00.github.io/tinygpt-viz/>**.
The model runs in the page, on the GPU through WebGPU (Chrome, Edge and Safari; Firefox on
Windows), or on the CPU in browsers without it. Every view works; add `?train=1` to the address to
train it as well (the first step downloads the 124 MB training set). The model is downloaded from
Hugging Face once and kept by the browser. See
[Getting started](docs/guide/01-getting-started.md#in-the-browser) for what differs from the server.

**Locally, with the Python server:** for development, and for long training runs at full speed.
Requires [uv](https://docs.astral.sh/uv/) and Node 20+. A GPU is optional: the model runs on any CPU.

```sh
uv sync                                          # Python and PyTorch
cd ui && npm install && npm run build && cd ..   # the UI
uv run python -m core.server                     # add --train for live training and replay
```

Then open http://localhost:8765. No training is needed first: the first run downloads the trained
checkpoints from Hugging Face ([oshea00/tinygpt-viz](https://huggingface.co/oshea00/tinygpt-viz)).
[Getting started](docs/guide/01-getting-started.md) covers the PyTorch builds (CUDA, CPU-only, ROCm),
the model, and the server's options.

## What's inside

- **Prompt box**: every keystroke re-runs the model; click a token to follow its row through every stage.
- **Pipeline map** (left): the residual spine with attention and MLP branches, one live thumbnail per stage.
- **Stepper**: ← / → walk forward through every stage, past the loss, and back up through the gradients.
- **Tensor inspector**: any stage's values or gradients as a heatmap with a visible colour scale;
  click a cell to see how it was computed (dot products, layer-norm statistics, softmax terms, gradient sums).
- **Attention**: arcs over the text from the selected position, the masked T×T matrix per head,
  a Q·K explorer (one query against every earlier key: products, dot, scaled score, weight), and a
  head gallery with pattern badges (and an induction score when a token repeats) and names you can give each head.
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
  rollout, gradient × input, occlusion), activation patching between a clean and a corrupted prompt
  (per snapshot and position, or per head), and the top-activating contexts of any MLP neuron.
- **Belief trajectories**: read stories one sentence at a time and track the model's belief about
  their mood (a cloze readout over emotion words), probe it from the residual stream, compare its
  geometry in both spaces, and steer it to see what else moves (after [Stories in Space](https://arxiv.org/abs/2605.12412)).
- **Fine-tuning**: train a low-rank (LoRA) adapter on stories about dragons and watch the model tilt
  towards them, with its own weights frozen; and explore an adapter preference-tuned with DPO, the
  reward-model-free alternative to RLHF, to refuse requests for stories about unsafe play: held-out requests, live pair
  scores and implicit rewards, and the refusal direction in the residual stream. Either adapter can
  be switched on for every view.
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
- **Lessons** (header menu): thirteen guided lessons that drive the views, with questions checked
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

## License

[MIT](LICENSE)
