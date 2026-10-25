# TinyGPT Visualizer — Design

Oct 25, 2026 · @Mike O'Shea

## Goals and audience

The tool lets a learner type a prompt, then step through one forward pass of a real, trained GPT with a 100-token context, seeing every number at every stage, watch it generate text one token at a time, and watch it learn one gradient step at a time.

Audience: students and self-learners who know basic linear algebra (vectors, matrix multiply) but have never seen inside a transformer. A secondary audience is someone who wants to build their own small GPT and needs a reference that shows what correct intermediate values look like. A third is someone exploring how concepts are represented: where similar words land, and which prompt words drive a response.

Design principles:

- **Real model, real numbers.** Everything shown comes from an actual trained network, not an animation of an idealised one. The text it writes will be imperfect, which is itself instructive.
- **Nothing hidden.** The model is sized so every weight and activation fits on screen at some zoom level. No stage is summarised away.
- **Overview first, then zoom.** The learner always knows where they are in the pipeline; any tensor can be expanded down to a single number and the formula that produced it.
- **One token, one path.** Selecting a token position highlights its row through every stage, so the 100×d blocks stop being abstract.
- **Poke it.** Learners can change inputs, ablate heads, and change sampling settings and see the consequences immediately, because inference is fast enough to re-run on every change.

## The model

Recommended default: a character-level GPT with 100-token context, 4 layers, 4 heads, width 64, about 215k parameters. Every weight fits in a \~465×465 pixel image, and one full forward pass produces about 800k activation values (≈3 MB as float32), so the whole network really can be shown as a bitmap.

| Config | Vocab | Context | Layers | Heads × head dim | Width d | MLP hidden | Params (approx) | Use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny | 65 chars | 100 | 2 | 2 × 16 | 32 | 128 | 32k | First lesson; every number readable |
| Default | 65 chars | 100 | 4 | 4 × 16 | 64 | 256 | 215k | Main teaching model |
| Larger | 65 chars | 100 | 6 | 8 × 16 | 128 | 512 | 1.2M | Better text; atlas view needs zoom |
| Concepts | 4,096 BPE | 100 | 6 | 8 × 32 | 256 | 1,024 | 5.8M | Conceptual associations; trained on TinyStories |

Where the 215k comes from: token embedding 65×64 (4k), learned position embedding 100×64 (6.4k), each block ≈50k (Q, K, V and output projections 4×64×64 = 16k; MLP 64×256 + 256×64 = 33k; biases and layer norms ≈1k), and the output unembedding 64×65 (4k). Teaching the parameter count is a lesson in itself: the MLPs hold two-thirds of the weights.

Per layer the trace records Q, K, V (3 × 100×64), attention scores and weights (2 × 4 heads × 100×100), MLP hidden pre- and post-activation (2 × 100×256), and several residual snapshots (100×64 each). A forward pass is roughly 50 MFLOP, a few milliseconds in PyTorch or in plain JavaScript, so the tool can re-run the model on every keystroke or slider move, even while training.

**Why characters for the mechanics profiles.** With 65 tokens the output distribution is a readable bar chart, the embedding table is a small image whose rows are recognisable letters, and tokenisation needs no explanation. The cost is that 100 characters is only about 15–20 words of context, and words are spread across several positions, which makes comparing words awkward. That is why the Concepts profile, below, uses BPE. The tokenizer sits behind an interface, so profiles differ only in config.

**Training data.** Tiny Shakespeare (≈1 MB) is the classic choice and trains in minutes on a laptop. Alternatives that make generated text easier to judge: simple children's stories, names, or a narrow domain such as recipes. Expect output that looks right in shape (line breaks, speaker names, word-like strings) but is often nonsense: that gap is a teaching point about scale.

**Architecture choices, fixed for clarity:** pre-layer-norm blocks (as in GPT-2), learned absolute position embeddings (a 100×64 image the learner can inspect), GELU in the MLP, causal mask, no dropout at inference, untied embedding and unembedding so each can be shown separately.

### Concepts profile

A second model built for finding conceptual associations rather than for stepping through mechanics: BPE with about 4,096 tokens, trained on TinyStories, about 5.8M parameters. Same code, same 100-token context, same trace format; only the tokenizer, config and corpus change.

- **Why BPE.** Common words become single tokens, so one word is one position and one vector per layer to compare. The embedding table becomes a word table whose nearest neighbours can carry meaning. And 100 tokens is roughly 75 words, enough for a short prompt and response.
- **Why about 4k tokens, not 256–512.** A very small BPE vocabulary mostly produces word fragments, which brings back the character-level problem in a milder form. Around 4k covers most of TinyStories' common words as whole tokens. Rarer words still split; the tools read the last sub-token by default and can pool across all of them.
- **Why TinyStories.** It is a synthetic set of short stories that use only words a typical 3- to 4-year-old understands, generated by GPT-3.5 and GPT-4. Its authors show that models under 10M parameters trained on it write fluent, coherent stories, and that such small models are notably interpretable ([Eldan and Li, 2023](https://arxiv.org/abs/2305.07759)). Its vocabulary is full of easy-to-probe concepts: animals, colours, emotions, family roles, actions.
- **Tied embeddings.** Unlike the mechanics profiles, input and output share one 4,096×256 table (about 1M parameters), so the vocabulary does not double the model's size.
- **Costs.** Training wants a GPU for reasonable turnaround. The atlas becomes a map to zoom around (about 2,400×2,400 pixels) rather than something taken in at a glance. Pretrained TinyStories models from 1M to 33M parameters are published on Hugging Face and make a useful comparison, though they use a different architecture and GPT-Neo's 50k-token vocabulary.

## Architecture

One desktop app on one machine: a Python process owns the model on the CPU or GPU and does training, inference, autograd and trace capture, while a web front end in a native window shows it all live. A browser-only static demo, running a TypeScript engine on exported weights, stays as a secondary mode for sharing.

&#91;embedded content: system architecture · Python side and browser side\]

Training and visualisation become one loop: the learner can pause training, take one update at a time, and see activations, gradients and weight changes as they happen. Every UI request goes through one queue, so traces always reflect a single consistent set of weights.

1. **Core (Python, PyTorch).** Owns weights, optimiser and data. The model is written with explicit matmuls, masking, softmax and layer norm rather than `nn.MultiheadAttention`, so the code reads like the maths and doubles as a teaching artifact. Runs on CPU, CUDA or Apple MPS.
2. **Trainer.** A worker thread with play, pause, single-step and reset; learning rate, batch size and trace frequency adjustable live. Saves checkpoints for later replay.
3. **Tracer.** Forward hooks and gradient hooks record every stage's output and gradient for a probe prompt. It runs on demand: every N training steps, after a single step, or on any UI request. Not on every batch, since a full trace is about 3 MB and capturing it each step would slow training many times over.
4. **Scheduler.** One command queue serialises access to the model. UI requests (run a prompt, ablate a head, sample) run between training steps, so there is no GPU contention and no half-updated weights in a trace.
5. **Bridge.** A local WebSocket. Commands go to the core as JSON; traces come back as binary frames: a JSON header listing stages, shapes and byte offsets, then the float32 data.
6. **UI (TypeScript).** The views in this doc, in a pywebview or Tauri window. Each view is a pure function of (trace, selection, settings), and the selection is shared: stage, layer, head, position, dimension.
7. **Static demo mode.** A hand-written TypeScript forward pass that loads exported weights in any browser with no Python; inference only. A parity test against the core keeps it honest.

The trace is a flat, ordered list of named stages, now with a gradient beside each output, so the pipeline map, the stepper and the atlas are generated from it rather than hand-coded:

```typescript
interface Tensor { shape: number[]; data: Float32Array; }

interface Stage {
  id: string;            // "L2.attn.H3.weights"
  kind: "embed" | "norm" | "linear" | "attn_scores" | "mask" | "softmax"
      | "attn_mix" | "concat" | "add" | "activation" | "logits" | "probs" | "loss";
  layer?: number; head?: number;
  output: Tensor;        // e.g. [100, 64] or [100, 100]
  grad?: Tensor;         // dLoss/dOutput, same shape; present when traced with a backward pass
  inputs: string[];      // ids of stages it reads from
  weights?: string[];    // names of weight tensors used
  formula: string;       // TeX shown in the inspector
  explain: string;       // one-paragraph lesson text
}

interface Trace {
  step: number;          // training step of the weights traced
  loss?: number;
  tokens: number[];      // up to 100 ids
  text: string[];
  stages: Stage[];       // in execution order
  weightGrads?: Record<string, Tensor>;   // per weight matrix
  sampled?: { probs: Float32Array; chosen: number; temperature: number; topK: number };
}

// UI → core
type Command =
  | { type: "run"; prompt: string; withGrad?: boolean }
  | { type: "train"; action: "play" | "pause" | "step" | "reset" }
  | { type: "set"; lr?: number; batchSize?: number; traceEvery?: number }
  | { type: "ablate"; target: string; mode: "zero" | "mean" | "off" }
  | { type: "sample"; temperature: number; topK?: number; topP?: number; n: number }
  | { type: "loadCheckpoint"; step: number };
// core → UI: "trace" (binary), "metrics" {step, loss, valLoss, tokensPerSec}, "sample" {text, probs}
```

Because each stage names its inputs and weights, the inspector can answer "where did this number come from?" for any cell: the input row, the weight column, and the dot product that produced it, and for gradients, the chain-rule terms that produced them.

## The stages to visualize

One forward pass is 21 stages: 3 at the input, 14 repeated in each block, 4 at the output. Shapes below use the default model (T = up to 100 positions, d = 64, 4 heads of 16, vocab 65). Each is a step in the stepper and a stage in the trace.

**Input**

1. **Tokenise.** Text → up to 100 ids. If the text is longer, keep the last 100; the window visibly slides during generation.
2. **Token embedding.** Look up one row of the 65×64 table per token → \[T, 64\].
3. **Position embedding + add.** Add row t of the 100×64 position table to position t → residual stream x₀ \[T, 64\].

**Each block (×4)**

4. **LayerNorm 1.** Each row: subtract its mean, divide by its standard deviation, then scale and shift by learned γ, β.
5. **Q, K, V projections.** Three \[T, 64\] × \[64, 64\] multiplies.
6. **Split into heads.** Reshape each into 4 × \[T, 16\]. Show this as cutting the 64 columns into four coloured bands.
7. **Scores.** Q·Kᵀ / √16 per head → 4 × \[T, T\].
8. **Causal mask.** Set the upper triangle to −∞, so a position cannot see later positions.
9. **Softmax.** Each row becomes weights that sum to 1 → the attention pattern.
10. **Mix values.** Weights × V → 4 × \[T, 16\]: each position's output is a weighted average of earlier positions' value vectors.
11. **Concat + output projection.** Rejoin to \[T, 64\], multiply by W\_O \[64, 64\].
12. **Residual add.** x ← x + attention output. Show the before, the delta, and the after.
13. **LayerNorm 2.**
14. **MLP up.** \[T, 64\] × \[64, 256\] + bias.
15. **GELU.** Element-wise; show the curve with each value as a dot on it.
16. **MLP down.** \[T, 256\] × \[256, 64\] + bias.
17. **Residual add.** x ← x + MLP output.

**Output**

18. **Final LayerNorm.**
19. **Unembedding.** \[T, 64\] × \[64, 65\] → logits \[T, 65\]. Every row is a prediction of the next character at that position, which is how the model trains on 100 targets at once.
20. **Softmax with temperature** on the last row → a distribution over 65 characters.
21. **Sample** (greedy, temperature, top-k or top-p) → append the token and start again.

The core formula, shown live in the inspector at stages 7–10 with the selected head's numbers:

```latex
\mathrm{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_{\text{head}}}} + M\right) V_h
```

An optional **incremental mode** shows generation as a real chat model does it, with a KV cache: after the first pass only the new last row is computed, and the view greys out every cell that was reused. This makes it obvious why generation is cheap per token and why context length is a hard limit.

## Backward pass and live training

Every forward stage has a matching gradient: PyTorch autograd runs the 21 stages in reverse, each producing a tensor the same shape as the stage's output, so every view can show gradients as well as activations. The stepper simply keeps going past the loss and walks back up the network.

**The backward stages, in order:**

1. **Loss.** Cross-entropy of each position's prediction against the actual next character, averaged over the 100 positions. Lessons can instead pick one target character so a single gradient is easy to follow.
2. **Logit gradient.** For softmax followed by cross-entropy this is simply the predicted probability minus 1 at the correct character, and the probability itself everywhere else. It is the cleanest place to show where learning starts.
3. **Unembedding and final layer norm.** Gradients for the 64×65 weight matrix and for the residual stream.
4. **Each block in reverse.** MLP down, GELU derivative, MLP up, layer norm 2; then the output projection, value mix, softmax backward, scores, and Q, K, V. At each residual add the gradient passes straight through and also branches into the sublayer.
5. **Embeddings.** Gradients land only on the rows of the token table for characters that appear in the batch, which the atlas shows as a few lit rows.

```latex
\frac{\partial L}{\partial z_i} = p_i - y_i
```

**What to show:**

- **The residual "gradient highway":** the gradient's size at each residual snapshot, layer by layer. Because each add passes the gradient through unchanged, it reaches early layers intact, which is why deep residual networks train at all.
- **Weight, gradient and update side by side** for any weight matrix: the current weights, their gradient, and the actual change the optimiser (Adam) applied, which often looks quite different from the raw gradient.
- **One training step, start to finish:** forward, loss, backward, update, then re-run the same prompt and show that the probability of the correct character rose slightly. This loop is the single most important idea in training.
- **Live training dashboard:** train and validation loss, tokens per second, a fresh sample every N steps, and the head gallery and embedding projection updating as the model learns.

**Optional manual-backward mode.** For learners building their own GPT, the core can include hand-written backward functions for each operation (about ten, from matmul to cross-entropy) and compare them with autograd's gradients stage by stage. This doubles as a test that the trace's gradient values are what the learner thinks they are.

Speed: one training step costs roughly three forward passes. The Tiny and Default configs train comfortably on a laptop CPU; a GPU matters for the Larger config or for training fast while watching.

## Views

Eight linked views share one selection (stage, layer, head, position, dimension); clicking in any view updates all the others. The pipeline map and the output panel are always on screen; the rest open in the main area.

| View | Answers the question | Core visual |
| --- | --- | --- |
| Pipeline map | Where am I in the computation? | Residual "spine" with attention and MLP branches; thumbnail per stage |
| Tensor inspector | What exactly is this number and where did it come from? | Heatmap of any tensor; cell drill-down to the dot product |
| Attention views | Which earlier characters does each position look at, and why? | Masked T×T matrix, arcs over text, head gallery, Q·K explorer |
| Residual stream | How does a token's vector change through the network? | Stacked \[T×64\] snapshots; per-sublayer "write size" bars |
| Logit lens | When does the model figure out the next character? | Layers × positions grid of top predictions |
| Output and generation | What comes next, and how does sampling choose? | 65-bar distribution, sampling sliders, probability-coloured text |
| Network atlas | What does the whole network look like at once? | Every weight and activation as one zoomable bitmap |
| Embedding space | What has the model learned about characters and positions? | 2D projections of embeddings and residual trajectories |

### Pipeline map and stepper

The residual stream is drawn as a vertical spine; each block's attention and MLP branch off it and rejoin at an add node. Each stage node carries a tiny live thumbnail of its output tensor. Controls: step forward/back one stage, play at adjustable speed, jump to a layer. Stepping animates the data moving: rows flowing out of the spine, through the branch, and back.

### Tensor inspector

Any tensor renders as a heatmap: rows are positions (labelled with the actual character), columns are dimensions. Hovering shows the value; clicking a cell opens its provenance. For a linear stage that is the input row and weight column as two bar strips, their element-wise products, and the sum, with the bias added. For layer norm it is the row's mean and standard deviation; for softmax, the exponentials and their total. This makes every matrix multiply concrete without asking the learner to trust an animation.

### Attention views

- **Matrix:** the 100×100 weights per head, upper triangle greyed as masked. Diagonal and column stripes are visible at a glance.
- **Arcs over text:** the prompt laid out as characters; selecting one draws arcs (or tints backgrounds) to the earlier characters it attends to, thickness by weight.
- **Head gallery:** a 4 layers × 4 heads grid of thumbnails for the current prompt, so learners can spot recognisable patterns, such as a head that attends to the previous character or to the start of the line, and name them.
- **Q·K explorer:** for one query position, its query vector next to every key vector, the raw dot products, scaled scores, and final softmax weights. This is the "search" metaphor of attention made literal.

### Residual stream

Nine snapshots per pass in the default model (after embedding, then after each attention and each MLP). For the selected position, show its 64-dim vector at each snapshot as a strip, plus a bar per sublayer showing how large a change it wrote (the norm of its delta). Learners see that the stream is a shared workspace that each sublayer reads from and adds to.

### Logit lens

Apply the final layer norm and unembedding to every residual snapshot, not just the last. The result is a grid: rows are snapshots, columns are positions, each cell shows the top predicted next character, shaded by confidence. Reading down a column shows the guess forming layer by layer, often wrong early and right late.

### Output and generation

A bar per vocabulary character for the next-character distribution, sortable by probability. Temperature, top-k and top-p sliders reshape the bars live, with the excluded tail greyed. A Sample button appends one character; Auto runs continuously. Generated text is coloured by the probability of each chosen character, so surprising choices stand out. Clicking any generated character reloads the trace that produced it (time travel through the generation).

### Network atlas (the bitmap)

The whole model as one image, laid out left to right in execution order: one column per block, each weight matrix as a tile at one pixel per weight (W\_Q, W\_K, W\_V, W\_O as 64×64 tiles, MLP matrices as 64×256 and 256×64), with the current trace's activations as \[T × d\] tiles beneath. About 215k weight pixels plus 800k activation pixels, rendered as a single WebGL texture so pan and zoom are instant.

- **Modes:** weights; activations for the current prompt; weight × input (what is actually flowing through each connection for the selected position); difference between two traces (before and after an edit); and, live during training, gradients and the most recent update (or change since an earlier checkpoint).
- **Level of detail:** zoomed out it is a texture showing structure (dead MLP neurons as dark columns, attention stripes); mid-zoom adds tile labels; fully zoomed shows numbers in cells.
- **Linked:** hovering any pixel names its tensor and index; clicking jumps the inspector to that stage and cell.

The atlas is striking and good for "it's really just these numbers", but pixel position has no inherent meaning, so it works as a map paired with the other views rather than on its own.

### Embedding space and other ideas

- **Embedding projections:** a 2D PCA of the 65 token embeddings (vowels, digits and punctuation tend to cluster; upper and lower case often pair) and of the 100 position embeddings (often a smooth curve).
- **Residual trajectories:** each position's vector projected to 2D at each snapshot, drawn as a path through the network.
- **Neuron pages:** pick one of the 1,024 MLP neurons; see where it fires in the current text and the training snippets that activate it most (precomputed).
- **Dimension tracking:** pick one of the 64 residual dimensions and highlight its column through every stage.
- **3D stacked view:** layers as stacked planes in 3D. Visually impressive but harder to read than 2D; a later nice-to-have.

## Interactivity

The tool earns its keep when learners change things and watch the effect; because a forward pass takes milliseconds, every change re-runs the model and every view updates.

**Interventions** (each shown as a before/after diff across all views):

- **Edit the prompt** character by character and watch attention patterns and predictions shift.
- **Ablate a head:** zero its output (or replace it with its average) and see which predictions break. This is how learners discover what a head does.
- **Ablate an MLP neuron or a whole block.**
- **Override attention:** force a head to attend only to one position.
- **Edit a value:** set one cell of any activation and propagate the change forward.
- **Swap position embeddings:** shuffle them and watch the model lose track of order.

**Live training and replay.** Training runs inside the app: play, pause, single-step, reset, with learning rate and batch size adjustable while it runs. The probe prompt is re-traced every N steps, so learners watch attention heads organise, embedding clusters form, the loss fall, and the generated text go from noise to almost-English. Saved checkpoints (say 30, from random initialisation to final) give a replay slider for jumping back to any point without retraining.

**Guided lessons.** A lesson is a script over the same views: a sequence of steps, each setting the selection, display mode and prompt, with a paragraph of narration and optionally a question ("Which head attends to the previous character? Click it."). Lessons are plain JSON, so new ones need no code. A starter set:

1. Text to numbers: tokens and embeddings.
2. Where am I? Position embeddings.
3. Attention as search: one head, one query, Q·K by hand.
4. Many heads: the head gallery and ablation.
5. The MLP: neurons that fire on specific patterns.
6. The residual stream and the logit lens.
7. From logits to text: temperature and sampling.
8. Why only 100 characters: the context window and the KV cache.
9. How it learns: one training step by hand, then live training.

## Concept explorer

A toolset for finding conceptual associations: run sets of prompts, capture the representation of chosen words at every layer, compare them, then test findings by intervening. It works with any profile but is designed for the Concepts profile, where one word is usually one token.

1. **Build a prompt set.** Type or import prompts, or fill templates such as "The {animal} ran to the {place}.", with target words tagged by category (animal, colour, emotion).
2. **Capture.** The core runs the set in batches and stores, for each occurrence of a target word, its residual vector at every snapshot, optionally with MLP activations. A thousand occurrences at 13 snapshots of 256 dimensions is about 13 MB.
3. **Compare.** Use the tools below; every point in every plot links back to its prompt and opens the full trace in the mechanics views.
4. **Test.** Turn a finding into an intervention, such as steering or ablation, and see whether generation changes as predicted.

| Tool | Question it answers | How |
| --- | --- | --- |
| Similarity matrix | Which words does the model treat as alike, and at which layer? | Cosine similarity between captured vectors, with a layer slider and sorting by category |
| Projection map | Do categories form clusters, and from which layer? | PCA (fast, linear) or UMAP of captured vectors, coloured by tag |
| Nearest neighbours | What sits near this word? | Nearest neighbours in the embedding table and in each layer's residual space |
| Context drift | Does a word's meaning shift with context? | One word across many prompts: spread and pairwise similarity per layer (river bank vs money bank) |
| Linear probes | Where does a concept become readable? | A logistic regression per layer on tagged vectors; accuracy by layer, tested on held-out words |
| Concept directions | Is there an axis for emotion, size, or tense? | Mean difference between two groups of contexts; project any word onto it |
| Steering | Does that direction actually change behaviour? | Add a scaled direction to the residual stream during generation; compare outputs side by side |
| Attribution | Which prompt words drove this response word? | Attention rollout, per-token ablation, and gradient × input, shown together because they often disagree |
| Neuron and feature search | What do individual units respond to? | Top-activating contexts per MLP neuron; stretch goal: a small sparse autoencoder on the residual stream to find features without choosing concepts in advance |

**Guarding against false findings:**

- **Centre before comparing.** Raw residual vectors share a large common component and a few dominant dimensions, which makes everything look similar. Subtract each layer's mean (and optionally normalise per dimension) before computing similarity.
- **Always show a baseline:** random words, shuffled category labels, and the same analysis on the untrained model at step 0. Structure that also appears in the untrained model is not something it learned.
- **Probes show correlation; steering and ablation test cause.** The explorer pairs them, so a concept found by a probe can be checked by an intervention in two clicks.

The explorer adds three commands to the protocol:

```typescript
  | { type: "capture"; promptSet: string; targets: "tagged" | "all"; pool: "last" | "mean" }
  | { type: "probe"; captureId: string; label: string; heldOut: string[] }
  | { type: "steer"; direction: string; layer: number; scale: number; prompt: string }
```

A tenth starter lesson fits here: where concepts live, covering clusters, a probe, and a steering demo on the Concepts profile.

## Visual encoding rules

One consistent colour language across every view: signed values use a diverging blue–white–red map centred on zero; probabilities and attention weights use a single-hue sequential map from white to dark; masked cells are a neutral hatch, never a colour.

- **Scale choice is a visible toggle:** per-tensor (best contrast, the default), per-stage-type across layers (so layer 1 and layer 4 are comparable), or global. The current scale and its range are always printed beside the heatmap. Silent rescaling is the commonest way such tools mislead.
- **Clip outliers** at a percentile (for example the 99th) with the clipped cells marked, since a few large values otherwise wash out everything else. Offer a symmetric log mode for residual streams, whose magnitudes grow with depth.
- **Rows are always positions, labelled with the character**, in every \[T × something\] view, so the eye can run along a token's row from stage to stage. Columns are dimensions.
- **Head identity is a hue**, used consistently for the four column bands of Q, K, V, the head gallery borders, and the arcs over text.
- **Selection is an outline, not a fill**, so it never competes with the data colours.
- **Accessible:** colour maps chosen to survive colour-blindness (blue–red and viridis-like sequential), and every value available as a number on hover.

## Tech stack and repo layout

A desktop app: a Python core with PyTorch, a TypeScript front end in a native window, and a local WebSocket between them; it runs on macOS, Windows or Linux, with a GPU optional.

| Layer | Choice | Why |
| --- | --- | --- |
| Core | Python 3 + PyTorch (CPU, CUDA or Apple MPS), model code adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) | Autograd and GPU support for free; nanoGPT's CPU recipe (4 layers, 4 heads, width 128, context 64) trains in about 3 minutes |
| Concurrency | Trainer in a worker thread; one command queue serialises model access | PyTorch releases the GIL inside operations; one queue keeps traces consistent and the code simple |
| Bridge | FastAPI (or plain `websockets`) on localhost; binary frames | A 3 MB trace crosses in milliseconds locally |
| App shell | pywebview (all-Python, simplest) or Tauri (smaller and more polished, launches the Python core as a sidecar) | A dedicated window rather than a browser tab |
| UI framework | TypeScript + Vite; Svelte or React | Views are pure functions of trace + selection |
| Heatmaps | Canvas 2D `ImageData` | A 100×256 tensor is 25,600 pixels: no GPU needed |
| Atlas | WebGL (raw or regl), tensors as textures | Instant pan and zoom over \~1M pixels |
| Diagrams, arcs, charts | SVG with d3-scale and d3-scale-chromatic | Crisp, easy hit-testing, standard colour maps |
| Formulas | KaTeX | Fast TeX rendering in the inspector |
| Static demo engine | Hand-written TypeScript forward pass in a Web Worker | Share a model with no install |
| Tests | pytest (core, tracer, manual-vs-autograd gradients); Vitest (demo parity) | The trace is the contract every view relies on |
| Packaging | uv-managed environment for development; PyInstaller or a Tauri sidecar for installers | PyTorch makes installers large: hundreds of MB, more with CUDA |
| Tokenizers | A 65-entry character map for mechanics; Hugging Face tokenizers (BPE) for Concepts | Trains a 4k vocabulary on TinyStories quickly; one interface for both |
| Analysis | scikit-learn (PCA, logistic-regression probes), umap-learn | Standard, and fast at this scale |

```text
tinygpt-viz/
  core/             # model.py, train.py, tracer.py, scheduler.py, server.py, export.py, manual_backward.py, tokenizers.py, analysis/ (capture, probes, steering, attribution, sae)
  data/             # corpus + tokenizer vocab
  runs/             # checkpoints and exported weights per config (tiny/, default/, larger/)
  ui/src/bridge.ts  # WebSocket client, binary trace decoding
  ui/src/views/     # pipeline/, inspector/, attention/, residual/, lens/, output/, training/, atlas/, embed/, concepts/
  ui/src/state/     # shared selection, trace store, settings
  ui/src/lessons/   # lesson JSON files + player
  ui/src/demo/      # static mode: forward.ts, worker.ts
  app/              # pywebview or Tauri launcher
  tests/            # pytest + Vitest, reference traces
```

The app is desktop-first: a 100×100 attention matrix and the atlas need a wide screen. The static demo can offer a reduced phone layout.

## Milestones

Twelve milestones, each ending in something usable; milestones 1–5 already give a working live tool, and later ones can be built in any order (12 needs only 1–3).

1. **Instrumented core.** Explicit-ops model and trainer with play, pause, single-step and reset; Tiny and Default configs train from the command line. Done when samples look recognisably Shakespeare-shaped.
2. **Tracer with gradients.** Forward and gradient hooks emit a full trace for a probe prompt. Tests: traced activations match an un-hooked run, and traced gradients match autograd's `.grad` values.
3. **Bridge and app shell.** WebSocket protocol, binary trace frames, scheduler; a pywebview or Tauri window showing a live loss curve and samples.
4. **Skeleton UI.** Prompt box, pipeline map, stepper running forward and backward, and the tensor inspector for every stage's activation and gradient.
5. **Training controls and the one-step view.** Play, pause, step, learning-rate and batch-size controls; the before-and-after view of a single update.
6. **Attention views.** Matrix, arcs over text, head gallery, Q·K explorer.
7. **Generation.** Output panel, sampling sliders, probability-coloured text, time travel, visible 100-character sliding window.
8. **Residual stream and logit lens,** including the gradient highway.
9. **Network atlas.** WebGL bitmap with weight, activation, gradient and update modes, live during training.
10. **Interventions and diff mode.** Head, neuron and block ablation; value editing; before-and-after views.
11. **Lessons, replay and sharing.** Lesson player and starter lessons; checkpoint replay slider; manual-backward mode; static browser demo with parity test; installers.

12) **Concepts profile and explorer.** Train the BPE TinyStories model; then prompt sets, capture, similarity matrix, projection map, neighbours and probes; then steering, attribution and feature search.

## Prior art and open decisions

Existing tools either show a large model at a high level of abstraction or a real-numbers model on a toy task; this design's niche is a real text generator small enough to show every number, trained live in the same app with every gradient visible, plus interventions.

| Tool | Model shown | Strength | Gap this design fills |
| --- | --- | --- | --- |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer) (Georgia Tech) | GPT-2 small, live in the browser | Smooth moves between overview and maths; your own prompt | Model too large to show every value; no training view |
| [LLM Visualization](https://github.com/bbycroft/llm-viz) (Brendan Bycroft) | minGPT sorting model, about 85k parameters, in 3D | Every block and value present, guided walkthrough | Sorts the letters A, B, C rather than writing text |
| [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy) | Character-level Shakespeare GPT | Shortest path to training one; source of our model code | Code only, no visualisation (now superseded by nanochat) |

Decisions to make before milestone 1:

- [ ] **Tokenizer:** characters for the mechanics profiles and BPE for Concepts is settled; still open are the exact vocabulary size (2k–8k) and the library (Hugging Face tokenizers is the default choice).
- [ ] **Default width:** 64 (everything visible) or 128 (better text, atlas needs more zoom).
- [ ] **Corpus:** TinyStories for Concepts is settled; for the mechanics profiles, Tiny Shakespeare or a slice of TinyStories, which would let both profiles share a domain and make comparisons easier.
- [ ] **UI framework:** Svelte or React.
- [ ] **App shell:** pywebview (simplest, all Python) or Tauri (smaller, more polished, more setup).
- [ ] **Target hardware:** CPU-only baseline, plus CUDA, Apple MPS, or both.
- [ ] **Manual-backward mode:** a core feature for builders, or an advanced extra.

A suggested first step: write the explicit-ops model and the trainer with play, pause and single-step, then the tracer, since every view depends on the trace format.

Sources: [nanoGPT README](https://github.com/karpathy/nanoGPT) · [Transformer Explainer repo](https://github.com/poloclub/transformer-explainer) · [llm-viz repo](https://github.com/bbycroft/llm-viz) · [GIGAZINE review of LLM Visualization](https://gigazine.net/gsc_news/en/20231204-llm-visualization)
