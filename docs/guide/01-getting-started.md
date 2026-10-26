# Getting started

## Install

You need [uv](https://docs.astral.sh/uv/), Node 20 or later, and Chrome or another modern browser. A GPU is optional if you [download trained checkpoints](#or-download-trained-checkpoints); training from scratch wants one.

```sh
uv sync                                   # Python and PyTorch (CUDA wheels on Linux and Windows)
cd ui && npm install && npm run build && cd ..
```

## The model

There is one model, `bpe-small`: 6 layers × 8 heads, 256-wide vectors, 5.8M parameters. It reads text as BPE tokens from a 4,096-token vocabulary, so most words are a single token (usually with the space before them, as in `␣dog`). That is what you want when you compare how it represents *dog* and *cat*.

## Train it

```sh
uv run python -m core.train --steps 20000   # ~22 minutes on a GPU
```

The first run downloads a slice of [TinyStories](https://arxiv.org/abs/2305.07759), short stories in simple English, and trains the 4,096-token vocabulary on it. Every 1,000 steps it prints a sample, and you can watch the text go from noise to *"Once upon a time, there was a girl named…"*. Before it starts, it times a few steps and prints an estimate of how long the run will take.

Checkpoints are saved under `runs/bpe-small/` at roughly logarithmic intervals, so that early learning can be [replayed](08-training.md#replaying-training) in detail.

You don't have to train first: with `--train`, the app can train from random weights while you watch. See [Training](08-training.md).

### Or download trained checkpoints

If `runs/bpe-small/` is empty, the server first downloads the model's checkpoints from Hugging Face ([oshea00/tinygpt-viz](https://huggingface.co/oshea00/tinygpt-viz)), along with the small validation set the neuron examples use. A model that only runs needs nothing else, so a machine without a GPU never downloads or tokenizes TinyStories. The shared checkpoints contain only weights. With `--train`, training continues from them using a fresh optimizer, and the full corpus is built the first time training needs it.

To publish your own runs, log in once with `hf auth login`, then:

```sh
uv run python -m core.hub push --config all     # every profile with checkpoints; --repo <user>/<name> for another repo
```

Set `TINYGPT_HF_REPO` (or pass `--repo` to `core.hub pull`) to download from a repo other than the default.

## Run the app

```sh
uv run python -m core.server
```

Then open <http://localhost:8765>. The server loads the latest checkpoint; add `--fresh` to start from random weights.

By default the model only runs: its weights stay as loaded. Add **`--train`** to turn on [training](08-training.md): the training panel, the training step tab and checkpoint replay.

```sh
uv run python -m core.server --train
```

## The screen

![The whole screen](../images/overview.png)

- **Prompt box** (top middle): every keystroke runs the model again. The chips under it are the positions the model sees, one per token. Click a chip to follow that position through every view.
- **Pipeline map** (left): the network from input to output. See [Pipeline map and stepper](02-pipeline-and-stepper.md).
- **Stepper** (under the prompt): walk the stages one by one.
- **Tabs** (middle): different ways of looking at the same forward pass: the [inspector](03-inspector.md), [attention](04-attention.md), [residual stream and logit lens](05-residual-and-lens.md), the [network atlas](06-atlas.md), [interventions](09-interventions.md), the [concept explorer](10-concepts.md) and, with `--train`, [one training step](08-training.md#one-training-step).
- **Next token** (top right): what the model predicts next, and [generation](07-generation.md).
- **Training** (bottom right, with `--train`): live training and [checkpoint replay](08-training.md).
- **lessons** and **help** (header): [guided lessons](11-lessons.md), and a short explanation of the open tab. The **?** beside the pipeline map, **Next token** and **Training** opens help on that panel; the buttons at the top of the help switch between topics.

![Help on the training panel](../images/help-training.png)

### One selection, everywhere

Everything on screen shares one **selection**: a stage, a position, and a dimension or key. Clicking in any view changes it, and every other view follows. Select a token chip and its row is outlined in the inspector, the atlas and the attention matrix. Select a stage in the atlas and the pipeline map scrolls to it. This is how you follow one token through the whole computation.

### Reading the colours

- **Blue, white, red**: signed values: negative, zero, positive. Each heatmap prints its scale above it; by default each tensor gets its own scale.
- **White to purple**: probabilities and attention weights, from 0 up.
- **Grey hatching**: masked cells (a position may not look at later ones), or nothing to show.
- **Outlines** mark the selection; they never change a cell's colour.
