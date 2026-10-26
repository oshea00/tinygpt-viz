# Training and publishing your own models

**You probably don't need this page.** The app downloads trained checkpoints for every model the first time it runs, and with `--train` it lets you watch training live in the browser (see [Getting started](guide/01-getting-started.md) and [Training](guide/08-training.md)). This page is for the brave souls who want to go further: retrain the models from scratch, change them for their own purposes, and share the results.

## Train from the command line

```sh
uv run python -m core.train --config char-small --steps 20000
```

The first run downloads a slice of [TinyStories](https://arxiv.org/abs/2305.07759), short stories in simple English, and builds the character vocabulary. Training takes about 7 minutes on a desktop GPU and about 30 on a CPU. Every 1,000 steps it prints a sample, and you can watch the text go from noise to *"Once upon a time, there was a girl named…"*.

For the BPE model:

```sh
uv run python -m core.data --bpe          # a larger slice, and a 4,096-token vocabulary
uv run python -m core.train --config bpe-small --steps 20000 --lr 6e-4 --batch-size 128   # ~22 minutes on a desktop GPU
```

Every run starts by timing a few real steps on the device it found and printing an estimate, so a run that is going to take all night says so in the first minute. The timings above are for a desktop GPU: a power-capped laptop GPU can be far slower, and the report flags one when it sees it. `--dry-run` stops after the report; `--log-every N` sets how often the progress line appears (default every 10 steps).

```sh
uv run python -m core.train --config bpe-small --steps 20000 --batch-size 128 --dry-run
```
```
bpe-small: 6L 8H d256 vocab 4,096 ctx 100, batch 128
device: cuda - NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.4 GB VRAM
  ! memory clock 810 of 5501 MHz at 10 W - this GPU is power-capped, so bandwidth-bound work runs several times slower than its rating
measured: 4,011 ms/step, 3,192 tok/s, peak 4.2 GB VRAM
estimate: 22.9h for 20,000 steps (22.3h training + 34m eval + 2m sampling), ~2.2 GB of checkpoints
```

Checkpoints are saved under `runs/<config>/` at roughly logarithmic intervals, so that early learning can be [replayed](guide/08-training.md#replaying-training) in detail. Your own checkpoints take the place of the downloaded ones: the server loads the latest checkpoint it finds there.

## Publish your checkpoints

To share your runs on Hugging Face, log in once with `hf auth login`, then:

```sh
uv run python -m core.hub push --config all     # every profile with checkpoints; --repo <user>/<name> for another repo
```

The pushed checkpoints contain only weights, not optimizer state.

To make the app download from your repo instead of the default, set `TINYGPT_HF_REPO=<user>/<name>`, or pull explicitly:

```sh
uv run python -m core.hub pull --config char-small --repo <user>/<name>
```
