# TinyGPT Visualizer guide

TinyGPT Visualizer shows one real, trained GPT doing its work, every number of it. You type a prompt and see what happens at each stage of the forward pass. You can generate text one token at a time, train the model one step at a time, and change parts of it to see what breaks.

![The app: pipeline map, prompt, inspector, next-token panel and training controls](images/overview.png)

## For learners

Start with [Getting started](guide/01-getting-started.md), then open **lessons → 1. Text to numbers** in the app. In the app, the **help** button explains whichever tab is open, and the **?** beside the pipeline map, the next-token panel and the training panel explains that panel.

| Page | What it covers |
| --- | --- |
| [Getting started](guide/01-getting-started.md) | Installing, running the app with the published models, the screen layout |
| [Pipeline map and stepper](guide/02-pipeline-and-stepper.md) | Finding your way through the network, stage by stage |
| [Inspector](guide/03-inspector.md) | Any stage's numbers, how each was computed, and its gradient |
| [Attention](guide/04-attention.md) | Which tokens each position looks at, and why |
| [Residual stream and logit lens](guide/05-residual-and-lens.md) | How one position's vector is built up, and when the prediction forms |
| [Network atlas](guide/06-atlas.md) | The whole model as one zoomable image |
| [Generation](guide/07-generation.md) | Sampling, temperature, top-k and top-p, time travel, the KV cache |
| [Training](guide/08-training.md) | Live training, checkpoint replay, and one training step up close |
| [Interventions](guide/09-interventions.md) | Ablating heads and neurons, editing values, before and after |
| [Concept explorer](guide/10-concepts.md) | How the model represents words: similarity, probes, steering |
| [Lessons](guide/11-lessons.md) | The thirteen guided lessons |
| [Embeddings](guide/12-embeddings.md) | The token and position tables as maps |
| [Belief trajectories](guide/13-beliefs.md) | A story's emotional belief, sentence by sentence: probes, geometry, steering and entanglement |

For the papers behind the views, see the [bibliography](bibliography.md), also under **Bibliography** in the app's **help** panel. For a term, open the glossary from the same panel.

## For instructors and lesson authors

| Page | What it covers |
| --- | --- |
| [Writing lessons](instructors/writing-lessons.md) | The lesson file format, answer checks, and finding token positions |
| [Prompt sets for the concept explorer](instructors/prompt-sets.md) | Defining categories and templates, and avoiding misleading results |

## For the brave: training and publishing your own models

If you want to retrain the models from scratch, modify them, or share your own checkpoints, see [Training and publishing your own models](training-your-own-models.md). Nobody needs this just to use the app.

## Regenerating the screenshots

The images in `docs/images/` come from `scripts/screenshots.mjs`, which drives headless Chrome against a running server. See the top of that file for how to run it. Use servers started just for this, because some of the shots train the model.
