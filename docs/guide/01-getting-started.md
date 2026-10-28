# Getting started

## In the browser

Open <https://oshea00.github.io/tinygpt-viz/>. Nothing is installed: the model runs in the page,
in a background worker, on the GPU through WebGPU, or on the CPU where the browser has no WebGPU
(the header shows which). It downloads the trained model from Hugging Face on the first visit,
about 25 MB, and the browser keeps it for the next.

Everything the server does works in the page, with a few differences:

- **Training** is off unless the address ends in `?train=1`, as the server needs `--train`. The
  first step downloads the training set (124 MB). Training starts with fresh optimizer moments,
  as the server does from the published checkpoints. A few steps a second on a desktop GPU
  is enough to watch the model learn, but slower than the server.
- **Checkpoints** saved during a session (by the schedule or **save now**) are kept in memory, the
  most recent eight; the published ones can always be replayed.
- **Random choices** (sampling, and the windows the neuron search reads) use the page's own seeded
  generator, so they differ from the server's draws.

To run the same page from your own machine, with your own checkpoints:

```sh
uv run python -m core.web_export --all          # writes ui/public/model/ (--train adds train.bin)
cd ui && npm install && npm run build           # ui/dist is the whole site: serve it with any web server
```

and open it with `?backend=local` (the build talks to the Python server unless told otherwise), or
build with `VITE_BACKEND=local` to make that the default. Add `?gpu=0` to force the CPU. The site
needs a web server: browsers do not run it from a file. For the guide's pages, copy `docs/` into
`ui/dist/docs/`.

## Install

You need [uv](https://docs.astral.sh/uv/), Node 20 or later, and Chrome or another modern browser. A GPU is optional: the published model runs on any CPU.

```sh
uv sync                                   # Python and PyTorch
cd ui && npm install && npm run build && cd ..
```

### PyTorch builds

PyTorch comes in one of three builds, chosen by uv dependency group:

- **`cuda`** (the default): CUDA 12.8 on x86 Linux and Windows, CUDA 13.0 on ARM Linux (such as DGX Spark), and the standard build with MPS on macOS.
- **`cpu`**: a smaller CPU-only install.
- **`rocm`**: PyTorch for ROCm, for AMD GPUs on x86 Linux. **Not yet tested on AMD hardware.**

To switch, turn the default off for this shell (add the line to your shell profile to keep it), then sync the build you want. `uv run` keeps that build from then on.

```sh
export UV_NO_GROUP=cuda
uv sync --group cpu                       # or --group rocm
```

## Run the app

```sh
uv run python -m core.server
```

Then open <http://localhost:8765>. You don't need to train anything first. The first time the server starts, it downloads the trained checkpoints from Hugging Face ([oshea00/tinygpt-viz](https://huggingface.co/oshea00/tinygpt-viz)), along with the small validation set the neuron examples use. After that it loads the latest checkpoint from `runs/bpe-small/`. A model that only runs needs nothing else, so a machine without a GPU never downloads or tokenizes TinyStories.

### Server options

- **`--port`** (default 8765) and **`--host`**: the server binds 127.0.0.1, so only this machine can reach it. Under WSL2, open the address in the Windows browser. Pass `--host 0.0.0.0` to reach it from other machines; it has no authentication.
- **`--fresh`**: start from random weights instead of the latest checkpoint.
- **`--train`**: enable training (see [Watch it train](#watch-it-train)).
- **`--device`**: choose the device (`cpu`, `cuda`, `mps`…) instead of the best one available.

For UI development, run `npm run dev` in `ui/` alongside the server and open <http://localhost:5173>: Vite reloads on every save and proxies `/ws` to port 8765.

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
- **Tabs** (middle): different ways of looking at the same forward pass: the [inspector](03-inspector.md), the [embedding maps](12-embeddings.md), [attention](04-attention.md), [residual stream and logit lens](05-residual-and-lens.md), the [network atlas](06-atlas.md), [interventions](09-interventions.md), the [concept explorer](10-concepts.md), [belief trajectories](13-beliefs.md) and, with `--train`, [one training step](08-training.md#one-training-step).
- **Next token** (top right): what the model predicts next, and [generation](07-generation.md).
- **Training** (bottom right, with `--train`): live training and [checkpoint replay](08-training.md).
- **lessons** and **help** (header): [guided lessons](11-lessons.md), and a short explanation of the open tab. The **?** beside the pipeline map, **Next token** and **Training** opens help on that panel; the buttons at the top of the help switch between topics. **Glossary** lists every weight, every kind of stage and the main ideas, each with the formula the inspector uses; a dotted-underlined name in a lesson or in the help opens its entry. **Bibliography**, the last of them, lists the papers behind the views ([also here](../bibliography.md)).
- **☾ / ☀** (header): switches between the dark and light themes. The app follows your system's setting until you choose, then remembers your choice.

![Help on the training panel](../images/help-training.png)

### One selection, everywhere

Everything on screen shares one **selection**: a stage, a position, and a dimension or key. Clicking in any view changes it, and every other view follows. Select a token chip and its row is outlined in the inspector, the atlas and the attention matrix. Select a stage in the atlas and the pipeline map scrolls to it. This is how you follow one token through the whole computation.

### Reading the colours

- **Blue, white, red**: signed values: negative, zero, positive. Each heatmap prints its scale above it; by default each tensor gets its own scale.
- **White to purple**: probabilities and attention weights, from 0 up.
- **Grey hatching**: masked cells (a position may not look at later ones), or nothing to show.
- **Outlines** mark the selection; they never change a cell's colour.
