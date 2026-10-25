# TinyGPT Visualizer

Step through one forward and backward pass of a real, trained, character-level GPT
(100-token context), seeing every number at every stage, and train it live. Design: [PROJECT.md](PROJECT.md).

Implemented so far: milestones 1–6 (instrumented core, tracer with gradients, WebSocket
bridge, pipeline map + stepper + tensor inspector, training controls and the one-step view,
attention views).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
uv sync                          # Python deps (PyTorch with CUDA 12.8 wheels on Linux/Windows)
uv run python -m core.data       # fetch a 10 MB TinyStories slice, build the character vocab
cd ui && npm install && npm run build && cd ..
```

## Train

```sh
uv run python -m core.train --config default --steps 20000
```

Profiles: `tiny` (32k params), `default` (215k), `larger` (1.2M). Default reaches
validation loss ≈ 0.95 in about 7 minutes on a GPU (about 30 on CPU). Checkpoints land in
`runs/<config>/` on a log-spaced schedule for later replay.

## Run the app

```sh
uv run python -m core.server --config default    # loads the latest checkpoint; --fresh for random weights
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
- **One training step**: an AdamW update on the prompt itself, with p(correct) before and after,
  and weight, gradient and actual update side by side.
- **Training panel**: play, pause, single-step, reset, learning rate, batch size, checkpoints.

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
