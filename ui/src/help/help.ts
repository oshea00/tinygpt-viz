// In-app help: a short explanation of whichever tab is open, and the always-visible panels.
// The full guide is in docs/guide/ (served by the app at /docs/). Text uses the lessons'
// formatting: **bold** and `code`.

export interface HelpTopic {
  title: string;
  what: string;          // what this view shows
  how: string[];         // how to use it
  tryThis: string[];     // things to try
  guide: string;         // page under docs/guide/
}

export const LAYOUT: HelpTopic = {
  title: "The screen",
  what: "One real, trained GPT, every number of it. Everything on screen shows the same forward pass of the prompt at the top, and shares one **selection**: a stage, a position (token) and a dimension. Clicking in any view changes the selection, and every other view follows.",
  how: [
    "**Prompt box** (top): every keystroke re-runs the model. Click a token chip to follow that position through every stage.",
    "**Pipeline map** (left): the network from input to output, one thumbnail per stage. Click one to select it.",
    "**Stepper**: walk the stages in order, with ← and →, past the loss and back up through the gradients.",
    "**Tabs** (middle): different ways of looking at the same pass. **Next token** (right) is always visible, and **training** under it when the server runs with `--train`.",
    "**lessons** (header): guided tours that drive all of these for you.",
  ],
  tryThis: ["Start with **lessons → 1. Text to numbers**.", "Type your own prompt and watch the next-token bars on the right change."],
  guide: "01-getting-started.md",
};

/** The always-visible panels, reached from the help panel's buttons or each panel's ?. */
export const PANELS: Record<string, HelpTopic> = {
  pipeline: {
    title: "Pipeline map and stepper",
    what: "The whole forward pass, top to bottom. The vertical line with ⊕ nodes is the **residual stream**, the vector each position carries; each block's attention and MLP branch off it, compute, and add their result back. Each stage has a live thumbnail of its output.",
    how: [
      "Click a thumbnail to select that stage: every view follows. Per-head stages show one thumbnail per head, bordered in the head's colour.",
      "**Stepper** (under the prompt): ◀ / ▶ or the ← → keys walk the stages in order; **play** steps automatically; **jump to** goes to a block, the output, or the start of the backward pass.",
      "Past the loss, the stepper walks **back up** the network: each stage then shows its gradient.",
      "Orange outlines: stages changed by an intervention. Grey: rows that came from the KV cache.",
      "**Picking an MLP neuron**: select an `L·.mlp.gelu` thumbnail and click a cell in the inspector. Each column there is one neuron's output, so the interventions tab then offers **use L… neuron … from the inspector**. Other MLP stages are not neurons: `mlp.up` is before the GELU, and `mlp.down` and the stream mix every neuron together.",
    ],
    tryThis: ["Press **play** in the stepper and watch one forward pass run, then the backward pass."],
    guide: "02-pipeline-and-stepper.md",
  },
  generation: {
    title: "Next token and generation",
    what: "What the model predicts next: one bar per possible next token, after the selected position (the last, unless you clicked a token chip). The token that actually comes next is outlined.",
    how: [
      "**temperature** sharpens (below 1) or flattens (above 1) the distribution; 0 always picks the top token. **top-k** and **top-p** cut off the unlikely tail (greyed).",
      "**sample one** appends one token drawn from the bars; **auto** keeps going until **stop** (or the end of a story, with **stop at ␄**).",
      "Generated text is redder where the choice was less likely. Click a token to load the trace that produced it.",
      "**KV cache**: compute only the new position, reusing every earlier one; the panel counts the work saved, until the window slides.",
    ],
    tryThis: ["Generate with temperature 0.3, then 1.6, and compare the text."],
    guide: "07-generation.md",
  },
  training: {
    title: "Training",
    what: "Train the model while you watch. Training runs on the server between your views' requests, so every view always shows one consistent set of weights, refreshed every **trace every** steps.",
    how: [
      "**play** / **pause** train continuously; **one step** does a single update; **reset** goes back to random weights (saved checkpoints are kept).",
      "The chart: training loss for each step (blue) and loss on held-out text (orange). A new sample appears every 500 steps.",
      "**learning rate** and **batch size** can change while it runs.",
      "**replay**: jump to any saved checkpoint, from random weights at step 0 to the end; **play through** steps through them all. Unsaved weights are saved before you jump away.",
    ],
    tryThis: ["Reset, press play, and watch the attention patterns and the samples form.", "Replay from step 0 with the head gallery open."],
    guide: "08-training.md",
  },
};

export const TOPICS: Record<string, HelpTopic> = {
  inspector: {
    title: "Inspector",
    what: "The selected stage's numbers as a heatmap. Rows are positions, labelled with their token; columns are dimensions (or key positions for attention, or vocabulary entries for logits). Blue is negative, red positive.",
    how: [
      "Hover a cell for its value; **click** it to see how it was computed: the input row and weight column of a matrix multiply, a layer norm's mean and spread, a softmax's terms.",
      "**value / gradient**: switch to ∂L/∂x, how the loss would change if this number changed. The gradient view also shows the stage's backward rule, checked against a hand-written backward pass.",
      "The scale (per tensor, per stage type, or global) and its range are always printed above the heatmap; clipped cells are marked •.",
    ],
    tryThis: ["Select `L0.q` and click a cell: the working shows the dot product that produced it.", "Step to `logits`, switch to **gradient**, and click the row of a position: p − y."],
    guide: "03-inspector.md",
  },
  attention: {
    title: "Attention",
    what: "Which earlier tokens each position looks at, for one layer and head at a time, and why.",
    how: [
      "**Arcs**: pick a token; arcs fan back to the tokens it attends to, thicker for more weight.",
      "**Matrix**: every query (row) against every key (column). The hatched triangle is the causal mask: no looking ahead.",
      "**Q·K explorer**: one query against every key: products, dot product, scaled score and softmax weight.",
      "**Head gallery**: every head at once, with a badge for its habit. Name heads, or **ablate** one to see what breaks.",
    ],
    tryThis: ["Find the sharpest head in the gallery and ablate it; then open the interventions tab to see the cost."],
    guide: "04-attention.md",
  },
  residual: {
    title: "Residual stream",
    what: "The shared workspace: one position's vector after the embedding and after every attention and MLP sublayer, which each read it and add to it.",
    how: [
      "**snapshots of x** or **what each sublayer writes** (the rows then sum to the final vector).",
      "**Write size**: how big each sublayer's addition is at this position, and at every position.",
      "**Gradient highway**: on the way back, how much gradient passes straight through each add and how much returns through the sublayer.",
      "**Dimension sizes**: the average size of each of the stream's dimensions; a few are far larger than the rest. Click a bar, or a column of the stream, to **track** that dimension: a chart follows it through every snapshot, and the inspector and atlas mark it wherever the stream's coordinates apply.",
      "**Trajectories**: every position's path through the snapshots, projected to 2D. Paths that run together are positions the model treats alike at that depth.",
    ],
    tryThis: [
      "Click different tokens in the prompt: which sublayer writes most for punctuation, or for the start of a word?",
      "Track the largest dimension: is it large for every position, and which sublayer makes it so?",
    ],
    guide: "05-residual-and-lens.md",
  },
  embeddings: {
    title: "Embeddings",
    what: "The model's two learned input tables as maps: the position table (one point per position) and the token table (one point per vocabulary entry), each projected onto its two main directions.",
    how: [
      "**Position table**: the points are joined in position order; the prompt's positions are larger and the selected one is ringed. Click one to select it.",
      "**Token table**: coloured by what the token looks like; the prompt's tokens are labelled. Type in the box to highlight tokens containing some text, and click a point to list its nearest rows.",
      "While training runs the maps keep the weights they were drawn from; **refresh the maps** redraws them.",
    ],
    tryThis: ["Hover the lone dot at the edge of the token map: it is many tokens the model never saw in training, stacked on one spot."],
    guide: "12-embeddings.md",
  },
  lens: {
    title: "Logit lens",
    what: "The final layer norm and unembedding applied to every residual snapshot, not just the last: what the stream would predict if the network stopped there.",
    how: [
      "Each cell is the top guess for the next token at that position and depth; underlined when it is the actual next token.",
      "Shade by confidence or by the probability of the actual next token. The right column is each row's accuracy.",
      "Click a cell for its top guesses, and the actual token's probability through the layers.",
    ],
    tryThis: ["Find a column where the guess is right from the first rows (an easy prediction), and one where it only forms at the end."],
    guide: "05-residual-and-lens.md",
  },
  atlas: {
    title: "Network atlas",
    what: "The whole model as one zoomable image: every weight, one pixel per number, with this prompt's activations under each column. It is a map: position has no meaning of its own.",
    how: [
      "Scroll to zoom, drag to pan. Labels appear as you zoom in, then the numbers themselves.",
      "**Click** a tile to select its stage (the pipeline follows); **double-click** to open it in the inspector.",
      "Modes: **values**, **gradients**, **weight × input** for the selected position, **diff** against a pinned snapshot, **recent change** during training.",
      "The legend under the atlas explains every mark.",
    ],
    tryThis: ["Pin, edit one word of the prompt, then choose **diff vs pinned**: only positions from the edit onward change."],
    guide: "06-atlas.md",
  },
  interventions: {
    title: "Interventions",
    what: "Change one thing inside the network and compare with the original: the loss, the next-token distribution, and where the change spreads.",
    how: [
      "Ablate a head, switch off a sublayer, force a head's attention, set any cell, shuffle the position embeddings.",
      "**Ablate an MLP neuron**: pick a layer in the tile's dropdown and click **open it in the inspector**, which opens that layer's `mlp.gelu` (or select any `L·.mlp.gelu` thumbnail). Click a cell there, come back, and **use L… neuron … from the inspector** fills in the layer and neuron. You can also type them. GELU's output is the place to do it: each column is one neuron, and that column is all the neuron passes to `mlp.down`. `mlp.up` is the same neuron before its non-linearity; `mlp.down` and the stream are mixes of every neuron.",
      "**Edit a value** sets one cell of any stage, selected in the inspector or the atlas. The stage decides what it means: in the stream it nudges one position; in `masked` it can unmask a later token and break causality; in `probs` it changes only the display, because the loss reads the logits.",
      "**zero** removes the unit's output entirely; **mean** keeps its average over the prompt and removes only what differs from position to position, a gentler test of what it does here.",
      "Every view then shows the changed model; changed stages are outlined orange in the pipeline. The inspector's **change from intervention** shows any stage's difference.",
      "Shortcuts: **ablate** in the head gallery; the concept explorer's **Neurons** panel shows what a neuron responds to.",
      "Interventions apply only to what you look at: training never sees them.",
    ],
    tryThis: [
      "Shuffle the position embeddings and watch the predictions fall apart.",
      "Select `L1.mlp.gelu`, click a bright cell, ablate that neuron here and look at **biggest change**. Then try **mean** instead of **zero**.",
    ],
    guide: "09-interventions.md",
  },
  concepts: {
    title: "Concept explorer",
    what: "How the model represents words. Capture runs a set of template prompts through the model and records the vector of each filled-in word at every snapshot; the panels then compare those vectors. Best on a model with whole-word tokens (`--config bpe-small`).",
    how: [
      "**Prompt set**: `@category: word word …` lines and templates with one `{category}` slot. **capture** after editing it, and after training.",
      "Keep **also the untrained model** ticked: structure it also shows was not learned.",
      "**similarity**, **projection map**, **neighbours**, **context drift** and **probes** analyse the captured vectors; **directions & steering** tests one by adding it during generation.",
      "**attribution** and **neurons** work on the main prompt and need no capture.",
      "**activation patching** compares two prompts that differ in one word: it copies one activation from the clean run into the corrupted one and shows how much of the clean answer comes back. Click a cell to apply that patch to the live model.",
    ],
    tryThis: ["Fit the probes, then steer with the direction of the category that probes best.",
              "Run activation patching on the example, then switch to **heads** to find the heads that carry the name to the end."],
    guide: "10-concepts.md",
  },
  beliefs: {
    title: "Belief trajectories",
    what: "What the model believes about a story's mood, sentence by sentence, and where that [[belief]] lives in the residual stream. After each sentence the story so far is followed by a [[cloze]] (\"Everyone was very\"), and the next-token probabilities of a word set, rescaled to sum to 1, are the belief. After the paper \"Stories in Space\" (Bigelow et al., 2026).",
    how: [
      "**read stories** runs the first N validation stories (or your own, one per paragraph) one sentence at a time. Keep **also the untrained model** ticked for a baseline. The median probability on the words says whether the cloze lands.",
      "**trajectories**: one story's belief after each sentence, and its path through two maps of every sentence: [[belief space]] and the residual stream at the snapshot.",
      "**probes**: a [[ridge regression]] per snapshot from the vectors to the belief, tested on held-out stories, beside guessing the average and the untrained model.",
      "**geometry**: each word's [[centroid]] in both spaces, their [[distance matrices]], and how well the two agree at every snapshot.",
      "**steering & entanglement**: add a word's direction ([[mean difference]] or probe weights, at one snapshot or a band of layers) and measure what moves on held-out stories. **steer each word** fills the [[entanglement]] matrix.",
    ],
    tryThis: ["Fit the probes, then pick a held-out story in **trajectories** and move the snapshot slider: the dashed probe lines close in on the readout.",
              "Steer **sad** with probe weights at one snapshot, then over a band of layers (L1 to L3), and compare. Lesson 13 walks through it all."],
    guide: "13-beliefs.md",
  },
  onestep: {
    title: "Training step",
    what: "One training update on the prompt itself: forward, loss, backward, then the optimiser changes every weight. Then the same prompt again.",
    how: [
      "Take a step: the loss before and after, and each position's probability of the correct next token.",
      "For any matrix: its weights, their gradient, and the change Adam actually applied, side by side.",
      "This changes the model's weights (in memory, until a checkpoint is saved).",
    ],
    tryThis: ["Take a few steps on one prompt: the probabilities climb towards 1. That is memorisation, not learning in general."],
    guide: "08-training.md",
  },
};
