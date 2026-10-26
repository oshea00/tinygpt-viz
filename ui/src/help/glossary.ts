// The glossary: every weight, every kind of stage, and the ideas the views and lessons use.
// Stage entries take their formula, backward rule and explanation from the live trace (the
// same text the inspector shows, from core/stages.py), so they never drift from the code.
// In lesson and help text, a `code` span naming an entry, or [[a term]], links here.

export type Section = "Weights" | "Stages" | "Concepts";

export interface GlossEntry {
  key: string;          // anchor: safe in an attribute and an element id
  section: Section;
  title: string;
  names: string[];      // shown in mono, and matched in text; L· and H· stand for any layer or head
  what?: string;        // formatted text (**bold**, `code`)
  tex?: string;         // for concepts; stages and weights show their stage's TeX instead
  stage?: string;       // a concrete stage id whose formula, backward rule and explanation to show
  weights?: string[];   // concrete weight names whose shapes to show (layer 0 stands for every layer)
}

const W = (title: string, names: string[], weights: string[], stage: string, what: string): GlossEntry =>
  ({ key: "", section: "Weights", title, names, weights, stage, what });

const WEIGHTS: GlossEntry[] = [
  W("Token table", ["wte", "W_te", "token table wte"], ["wte"], "embed.tok",
    "One learned row per vocabulary entry. Looking up a token's row starts its vector. The embeddings are **tied**: the same table, transposed, turns the last layer's vector back into a score per token (see `logits`)."),
  W("Position table", ["wpe", "W_pe", "position table wpe"], ["wpe"], "embed.pos",
    "One learned row per position, 0 to 99, added to the token's row so the model knows where each token is."),
  W("LayerNorm 1 scale and shift", ["L·.ln1_g", "L·.ln1_b", "LN1 γ β"], ["L0.ln1_g", "L0.ln1_b"], "L0.ln1",
    "γ scales and β shifts each dimension after layer norm 1 normalises a row, in every block."),
  W("Query projection", ["L·.W_q", "L·.b_q", "W_Q", "W_q"], ["L0.W_q", "L0.b_q"], "L0.q",
    "Turns each position's normalised vector into its queries, one 32-wide band per head: what the position is looking for."),
  W("Key projection", ["L·.W_k", "L·.b_k", "W_K", "W_k"], ["L0.W_k", "L0.b_k"], "L0.k",
    "Turns each position's vector into its keys: what it offers to queries from later positions."),
  W("Value projection", ["L·.W_v", "L·.b_v", "W_V", "W_v"], ["L0.W_v", "L0.b_v"], "L0.v",
    "Turns each position's vector into its values: the content it hands over when attended to."),
  W("Output projection", ["L·.W_o", "L·.b_o", "W_O", "W_o"], ["L0.W_o", "L0.b_o"], "L0.attn.out",
    "Mixes the heads' concatenated outputs into one vector to add back to the residual stream."),
  W("LayerNorm 2 scale and shift", ["L·.ln2_g", "L·.ln2_b", "LN2 γ β"], ["L0.ln2_g", "L0.ln2_b"], "L0.ln2",
    "γ and β of the layer norm before the MLP."),
  W("MLP up projection", ["L·.W_up", "L·.b_up", "W_up"], ["L0.W_up", "L0.b_up"], "L0.mlp.up",
    "Expands each 256-wide vector to the MLP's 1,024 hidden neurons. Each column is one neuron's input weights."),
  W("MLP down projection", ["L·.W_down", "L·.b_down", "W_down"], ["L0.W_down", "L0.b_down"], "L0.mlp.down",
    "Projects the 1,024 neurons back to 256 dimensions to add to the stream. Each row is what one neuron writes."),
  W("Final LayerNorm scale and shift", ["ln_f_g", "ln_f_b", "final LN γ β"], ["ln_f_g", "ln_f_b"], "ln_f",
    "γ and β of the last layer norm, before the scores per token are read out."),
  W("Unembedding", ["W_U", "W_unembed", "unembedding W_U"], [], "logits",
    "Turns the final vector into one score per token. `bpe-small` has no separate matrix: its embeddings are tied, so this is `wte` transposed."),
];

/** One entry per kind of stage; the id uses layer 0 and head 0 for all layers and heads. */
const STAGES: [string, string][] = [
  ["embed.tok", "Token embedding"], ["embed.pos", "Position embedding"], ["embed.sum", "x₀ = tok + pos"],
  ["L0.ln1", "LayerNorm 1"], ["L0.q", "Queries Q"], ["L0.k", "Keys K"], ["L0.v", "Values V"],
  ["L0.attn.H0.q", "Split Q into heads"], ["L0.attn.H0.k", "Split K into heads"], ["L0.attn.H0.v", "Split V into heads"],
  ["L0.attn.H0.scores", "Attention scores Q·Kᵀ/√d"], ["L0.attn.H0.masked", "Causal mask"],
  ["L0.attn.H0.weights", "Softmax → attention weights"], ["L0.attn.H0.mix", "Mix values"],
  ["L0.attn.concat", "Concatenate heads"], ["L0.attn.out", "Output projection"], ["L0.resid_attn", "Residual add (attention)"],
  ["L0.ln2", "LayerNorm 2"], ["L0.mlp.up", "MLP up"], ["L0.mlp.gelu", "GELU"], ["L0.mlp.down", "MLP down"],
  ["L0.resid_mlp", "Residual add (MLP)"], ["ln_f", "Final LayerNorm"], ["logits", "Logits"], ["probs", "Probabilities"], ["loss", "Loss"],
];

const C = (title: string, names: string[], what: string, tex?: string): GlossEntry =>
  ({ key: "", section: "Concepts", title, names, what, tex });

const CONCEPTS: GlossEntry[] = [
  C("Token", ["token", "tokens"], "A piece of text the model reads as one unit: usually a whole word with the space before it (`␣dog`), sometimes a word piece, a punctuation mark or a single byte. Each position holds one token."),
  C("Vocabulary", ["vocabulary", "vocab"], "Every token the model knows: 4,096 for `bpe-small`, learned from TinyStories by byte-level BPE, which merges the most frequent pairs of pieces until the vocabulary is full."),
  C("Embedding", ["embedding", "embeddings"], "A learned vector standing for a token (a row of `wte`) or a position (a row of `wpe`). Their sum is the residual stream's starting point.", String.raw`x_0 = W_{te}[\text{token}_t] + W_{pe}[t]`),
  C("Residual stream", ["residual stream", "stream"], "The vector each position carries through the network. Every attention and MLP sublayer reads a normalised copy and **adds** its result; nothing overwrites it.", String.raw`x \leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x)), \qquad x \leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x))`),
  C("Snapshot", ["snapshot", "snapshots"], "The residual stream at one point: after the embedding, then after each attention and each MLP. `bpe-small` has 13: x₀ and two per block."),
  C("Layer norm", ["layer norm", "LayerNorm"], "Shifts each row to mean 0 and scales it to standard deviation 1, then applies a learned scale γ and shift β per dimension. It keeps the numbers in a steady range from layer to layer.", String.raw`\hat x = \gamma \odot \frac{x - \mu}{\sigma} + \beta`),
  C("Attention head", ["head", "heads", "attention head"], "One of 8 attention units per block, each 32 dimensions wide. A head scores every earlier position against the current one and takes a weighted average of their values.", String.raw`\mathrm{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{d_{\text{head}}}} + M\right) V_h`),
  C("Query, key and value", ["query", "key", "value", "queries", "keys", "values"], "A head's three views of each position: the **query** says what it looks for, the **key** what it offers, and the **value** what it passes on. A query–key dot product decides how much of a value is taken."),
  C("Causal mask", ["causal mask", "mask"], "Stops a position from attending to later ones, so a prediction can only use the text before it. Masked scores are set to −∞, so softmax gives them weight 0.", String.raw`M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`),
  C("Softmax", ["softmax"], "Turns a row of scores into probabilities: all positive, summing to 1, with larger scores getting more.", String.raw`p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}`),
  C("MLP and neurons", ["MLP", "neuron", "neurons"], "The second sublayer of each block: each position's vector is expanded to 1,024 hidden **neurons**, passed through GELU, and projected back. It works on each position separately.", String.raw`\mathrm{MLP}(x) = \mathrm{GELU}(x W_{up} + b_{up})\, W_{down} + b_{down}`),
  C("GELU", ["GELU", "gelu"], "The MLP's non-linearity: passes positive values almost unchanged and squashes negative ones towards 0. Φ is the standard normal distribution function.", String.raw`\mathrm{GELU}(u) = u\,\Phi(u)`),
  C("Logit", ["logit", "logits"], "A raw score for one token, before softmax. The model produces 4,096 per position, one per vocabulary entry."),
  C("Tied embeddings", ["tied embeddings", "tied"], "Using the token table both to read tokens in and, transposed, to score them on the way out. It saves a 4,096 × 256 matrix.", String.raw`z = \hat x\, W_{te}^{\top}`),
  C("Cross-entropy loss", ["loss", "cross-entropy"], "The average surprise at the actual next token: minus the log of the probability the model gave it. Its gradient with respect to the logits is p − y.", String.raw`L = -\frac{1}{T}\sum_t \log p_t[y_t]`),
  C("Gradient", ["gradient", "gradients"], "How much the loss would change per small change in a number. Every stage and weight has one, the same shape as itself; training moves weights against it."),
  C("AdamW", ["AdamW", "Adam"], "The optimiser: it scales each weight's step by that weight's running gradient size, so small but consistent gradients still get sizeable steps, and decays weights slightly towards 0."),
  C("Context window", ["context window", "window"], "The 100 positions the model can see at once. Longer text keeps its last 100 tokens, so the window slides during generation."),
  C("KV cache", ["KV cache"], "Keeping each layer's keys and values from earlier positions, so generating a token computes only the new position. A sliding window shifts every position and invalidates it."),
  C("Temperature, top-k and top-p", ["temperature", "top-k", "top-p"], "Ways to reshape the next-token distribution before sampling: temperature τ sharpens (below 1) or flattens (above 1) it; top-k keeps the k likeliest tokens; top-p keeps the smallest set whose probability reaches p.", String.raw`p_i \propto e^{z_i / \tau}`),
  C("Logit lens", ["logit lens"], "Reading a prediction out of every snapshot, not just the last, by applying the final layer norm and unembedding early. It shows when the guess forms."),
  C("Principal components", ["PCA", "principal component", "principal components"], "The directions along which a set of vectors varies most. The maps in the embeddings tab, the concept explorer and the trajectories project onto the top two, and say how much of the variation they keep."),
  C("Ablation", ["ablation", "ablate"], "Switching part of the model off (zeroing a head or neuron, or replacing it with its average) to see what depends on it."),
  C("Steering", ["steering", "steer"], "Adding a direction to the residual stream during generation, to test whether that direction changes what the model writes.", String.raw`x \leftarrow x + \alpha\, v`),
  C("Probe", ["probe", "probes"], "A simple classifier (here logistic regression) trained to read a concept from vectors at one snapshot. Accuracy on held-out words shows where a concept becomes readable, but not that the model uses it."),
];

/** Strip layer and head numbers: L3.attn.H5.weights → L·.attn.H·.weights. */
export const generic = (name: string) => name.trim().replace(/\bL\d+\b/g, "L·").replace(/\bH\d+\b/g, "H·");
const slug = (s: string) => s.replace(/·/g, "x").replace(/[^\w.-]+/g, "-");

export const GLOSSARY: GlossEntry[] = [
  ...WEIGHTS.map((e) => ({ ...e, key: `w-${slug(e.names[0])}` })),
  ...STAGES.map(([id, title]): GlossEntry => ({ key: `s-${slug(generic(id))}`, section: "Stages", title, names: [generic(id)], stage: id })),
  ...CONCEPTS.map((e) => ({ ...e, key: `c-${slug(e.names[0])}` })),
];

// Exact names first (code is case-sensitive: W_Q, wte), then concepts case-insensitively.
// A name can belong to a stage and a concept (logits, loss): the code name wins.
const exact = new Map<string, string>();
const loose = new Map<string, string>();
for (const e of GLOSSARY) for (const n of e.names) {
  if (!exact.has(n)) exact.set(n, e.key);
  if (e.section === "Concepts" && !loose.has(n.toLowerCase())) loose.set(n.toLowerCase(), e.key);
}

/** The glossary entry a name in text refers to, or null. */
export function glossKey(name: string): string | null {
  const g = generic(name);
  return exact.get(g) ?? loose.get(g.toLowerCase()) ?? null;
}
