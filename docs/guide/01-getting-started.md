# Getting started

## Install

You need [uv](https://docs.astral.sh/uv/), Node 20 or later, and Chrome or another modern browser. A GPU is optional: the published model runs on any CPU.

```sh
uv sync                                   # Python and PyTorch (CUDA wheels on Linux and Windows)
cd ui && npm install && npm run build && cd ..
```

## Run the app

```sh
uv run python -m core.server
```

Then open <http://localhost:8765>. You don't need to train anything first. The first time the server starts, it downloads the trained checkpoints from Hugging Face ([oshea00/tinygpt-viz](https://huggingface.co/oshea00/tinygpt-viz)), along with the small validation set the neuron examples use. After that it loads the latest checkpoint from `runs/bpe-small/`. A model that only runs needs nothing else, so a machine without a GPU never downloads or tokenizes TinyStories.

## The model

There is one model, `bpe-small`: 6 layers × 8 heads, 256-wide vectors, 5.8M parameters. It reads text as BPE tokens from a 4,096-token vocabulary, so most words are a single token (usually with the space before them, as in `␣dog`). That is what you want when you compare how it represents *dog* and *cat*.

## Watch it train

By default the model only runs: its weights stay as loaded. Add **`--train`** to turn on [training](08-training.md): the training panel, the training step tab and checkpoint replay.

```sh
uv run python -m core.server --train
```

Training continues from the downloaded checkpoints with a fresh optimizer; the full corpus is downloaded and built the first time training needs it. Add `--fresh` to start from random weights instead and watch the model learn from nothing.

To train the model from the command line, change it, or publish your own checkpoints, see [Training and publishing your own models](../training-your-own-models.md).

## The screen

![The whole screen](../images/overview.png)

- **Prompt box** (top middle): every keystroke runs the model again. The chips under it are the positions the model sees, one per token. Click a chip to follow that position through every view.
- **Pipeline map** (left): the network from input to output. See [Pipeline map and stepper](02-pipeline-and-stepper.md).
- **Stepper** (under the prompt): walk the stages one by one.
- **Tabs** (middle): different ways of looking at the same forward pass: the [inspector](03-inspector.md), [attention](04-attention.md), [residual stream and logit lens](05-residual-and-lens.md), the [network atlas](06-atlas.md), [interventions](09-interventions.md), the [concept explorer](10-concepts.md) and, with `--train`, [one training step](08-training.md#one-training-step).
- **Next token** (top right): what the model predicts next, and [generation](07-generation.md).
- **Training** (bottom right, with `--train`): live training and [checkpoint replay](08-training.md).
- **lessons** and **help** (header): [guided lessons](11-lessons.md), and a short explanation of the open tab. The **?** beside the pipeline map, **Next token** and **Training** opens help on that panel; the buttons at the top of the help switch between topics.
- **☾ / ☀** (header): switches between the dark and light themes. The app follows your system's setting until you choose, then remembers your choice.

![Help on the training panel](../images/help-training.png)

### One selection, everywhere

Everything on screen shares one **selection**: a stage, a position, and a dimension or key. Clicking in any view changes it, and every other view follows. Select a token chip and its row is outlined in the inspector, the atlas and the attention matrix. Select a stage in the atlas and the pipeline map scrolls to it. This is how you follow one token through the whole computation.

### Reading the colours

- **Blue, white, red**: signed values: negative, zero, positive. Each heatmap prints its scale above it; by default each tensor gets its own scale.
- **White to purple**: probabilities and attention weights, from 0 up.
- **Grey hatching**: masked cells (a position may not look at later ones), or nothing to show.
- **Outlines** mark the selection; they never change a cell's colour.
