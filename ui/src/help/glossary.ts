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

// An adapter's weights and stages exist only while one is switched on (the fine-tune tab).
const LORA_SITES = ["q", "k", "v", "attn.out", "mlp.up", "mlp.down"];
const ADAPTER: GlossEntry[] = [
  { key: "w-lora", section: "Weights", title: "Adapter matrices A and B", names: LORA_SITES.flatMap((t) => [`L·.${t}.lora_A`, `L·.${t}.lora_B`]),
    what: "A low-rank adapter's two thin matrices beside a frozen weight: A [d_in × r] squeezes each row to r numbers, B [r × d_out] expands them back. Only these train; B starts at zero." },
  { key: "s-lora", section: "Stages", title: "Adapter term", names: LORA_SITES.map((t) => `L·.${t}.lora`),
    what: "An adapter's contribution to the stage it is named after, (x A) B · s, which that stage adds to its own x W + b. It appears only while an adapter is switched on." },
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
  C("Causal mask", ["causal mask", "mask"], "Stops a position from attending to later ones, so a prediction can only use the text before it. Masked scores are set to −∞, so softmax gives them weight 0. It also means text added later never changes an earlier position's vectors.", String.raw`M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`),
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
  C("Principal components", ["PCA", "principal component", "principal components"], "The directions along which a set of vectors varies most. The maps in the embeddings tab, the concept explorer and the trajectories project onto the top two, and say how much of the variation they keep. **UMAP**, a non-linear alternative that the belief-trajectories paper also uses, is not in the app."),
  C("Ablation", ["ablation", "ablate"], "Switching part of the model off (zeroing a head or neuron, or replacing it with its average) to see what depends on it."),
  C("Activation patching", ["activation patching", "patching"], "Running a corrupted prompt with one activation copied in from a clean prompt, to find where the model carries the information the two prompts differ in. Ablation asks what is **necessary**; patching asks what is **sufficient** to restore the answer."),
  C("Induction head", ["induction head", "induction heads"], "A head that finds an earlier copy of the current token and attends to the token after it, so “A B … A” predicts B. It needs an earlier head that moves each token's predecessor into its position (a previous-token head)."),
  C("Steering", ["steering", "steer", "activation addition", "steering vector"], "Adding a direction v to the residual stream while the model runs, to test whether that direction changes what it does. α sets the strength; belief trajectories gives it in units of the snapshot's typical vector length. The direction can be a probe's weights or a difference of means, added at one snapshot or over a **band** of layers.", String.raw`x \leftarrow x + \alpha\, v`),
  C("Probe", ["probe", "probes", "linear probe"], "A simple model trained to read something from the vectors at one snapshot: logistic regression for word categories, ridge regression for beliefs. Success on held-out data shows the information is **readable** there, not that the model **uses** it; steering or ablation tests that."),
  C("In-context learning", ["in-context learning", "ICL"], "The model changing its predictions based on what it has read so far in the prompt, with no change to its weights. Reading a story and updating its sense of the story's mood is a small case of it. One view, the **Bayesian** one, treats the model as starting from prior expectations and updating them on the evidence in the text."),
  C("Belief", ["belief", "beliefs", "belief update"], "In belief trajectories: what the model expects about a story so far, as a probability distribution over a set of words (happy, sad, …), read with a cloze after each sentence. A **belief update** is how that distribution changes after one more sentence."),
  C("Belief space", ["belief space", "conceptual belief space", "belief trajectory", "belief trajectories"], "A space where each point is one belief: with six words, six numbers that sum to 1, so every point lies on a simplex whose corners mean \"only this word\". A story traces a **trajectory** through it, one point per sentence. Beliefs don't fill the space; they lie near a few directions (a **low-dimensional manifold**), so a 2-D map shows most of the motion."),
  C("Cloze readout", ["cloze", "readout", "behavioural readout", "renormalised"], "Measuring the model by what it outputs rather than by its internals. A **cloze** is a fill-in-the-blank continuation such as \"Everyone was very\"; the next-token probabilities of the chosen words, **renormalised** to sum to 1, are the readout. Their total probability before renormalising (the **mass**) says whether the cloze lands: if it is low, the model expects some other word.", String.raw`b_k = \frac{p(w_k)}{\sum_j p(w_j)}`),
  C("Ridge regression", ["ridge regression", "ridge"], "Linear regression with a penalty on the size of the weights, which keeps a probe from fitting noise when there are many dimensions and few examples. Features are standardised first, using the training rows only.", String.raw`W = (X^{\top}X + \lambda I)^{-1} X^{\top} Y`),
  C("Held-out data", ["held out", "held-out", "train/test split"], "Examples kept out of training and used only to test. Belief probes hold out whole stories (every fifth), so a probe can't score well by memorising a story's sentences."),
  C("RMSE", ["RMSE", "root-mean-square error"], "Root-mean-square error: the typical size of a prediction's mistake. Compare it with the error of always guessing the training average, the baseline a probe must beat.", String.raw`\mathrm{RMSE} = \sqrt{\tfrac{1}{n}\textstyle\sum (\hat y - y)^2}`),
  C("Correlation", ["correlation", "Pearson r", "Pearson correlation"], "How closely two sets of numbers rise and fall together: r = 1 when they move perfectly together, 0 when unrelated, −1 when opposite. With few pairs (15 for six words) it is a rough guide."),
  C("Centroid", ["centroid", "centroids"], "The average point of a group. In belief trajectories, each word's centroid is the mean over the sentences that express it most, at most 3 from any one story."),
  C("Representational similarity", ["representational similarity", "distance matrix", "distance matrices"], "Comparing two spaces by the arrangement of the same items in each: compute the distances between every pair (a **distance matrix**) in both, and correlate them. If behaviour and the residual stream place the emotions alike, the matrices correlate."),
  C("Difference of means", ["difference of means", "mean difference"], "A direction for a concept: the average vector where it is present minus the average where it isn't. Simple, and often a better steering direction than a probe's weights."),
  C("Self-repair", ["self-repair"], "Later layers undoing a change made at an earlier one, so an intervention at a single layer has less effect than expected. It is why the paper steers with a probe direction over a band of layers."),
  C("Entanglement", ["entanglement"], "Steering one concept also moving others. The paper finds concepts close together in belief space move together; the entanglement matrix shows each word's change in log probability when each word is steered."),
  C("Fine-tuning", ["fine-tuning", "fine-tune", "fine-tuned"], "Training a model that has already been trained, on a smaller set of examples, to change what it does: write about one topic, follow instructions, refuse some requests. A full fine-tune updates every weight; LoRA updates a few small added matrices."),
  C("LoRA", ["LoRA", "low-rank adapter", "adapter", "adapters"], "Low-rank adaptation (Hu et al., 2021). Beside a frozen weight W, train two thin matrices A and B; the stage computes x W + (x A) B. Their product ΔW = A B is a change to W of **rank** at most r, described by r (d_in + d_out) numbers instead of d_in × d_out. B starts at zero, so training starts from the model as it was. Many adapters can share one base model and be switched on and off.", String.raw`y = x W + b + (x A) B \cdot s`),
  C("Rank", ["rank", "low-rank"], "The number of independent directions a matrix can map onto. A product A B through r numbers has rank at most r, however large the matrix; its **singular values** (at most r nonzero) say how much it uses each direction."),
  C("Singular values", ["singular value", "singular values"], "How much a matrix stretches along each of its independent directions, largest first. A rank-r matrix has r nonzero singular values; the square root of the sum of their squares is its size (Frobenius norm)."),
  C("RLHF", ["RLHF", "reinforcement learning from human feedback", "human feedback"], "Tuning a language model on what people prefer. People compare pairs of answers; a **reward model** learns to score answers the way they do; the language model is then optimised (with PPO, a reinforcement-learning method) to earn high reward while staying close to the original model (Ouyang et al., 2022). It is how chat models learn to be helpful and to decline harmful requests."),
  C("DPO", ["DPO", "direct preference optimisation", "direct preference optimization"], "Direct preference optimisation (Rafailov et al., 2023), now the usual way to preference-tune: RLHF's goal without a separate reward model or reinforcement learning. For a request with a preferred and a rejected answer, the loss widens the gap between their log-probability gains over the original model. β sets how far the model may move from the original.", String.raw`\mathcal L = -\log \sigma\!\Big(\beta\big[\log\tfrac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)} - \log\tfrac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}\big]\Big)`),
  C("Reference model", ["reference model", "π_ref", "reference"], "In RLHF and DPO, the frozen model that tuning starts from. Rewards are measured against it, so the tuned model is pulled towards preferred answers without drifting into text the original would never write. With an adapter, it is the same network with the adapter off."),
  C("Implicit reward", ["implicit reward", "reward margin", "reward model"], "In DPO, β times how much more likely the tuned model makes an answer than the reference does. It plays the part of RLHF's reward model, which is never trained separately: the policy contains it. The margin is the preferred answer's implicit reward minus the rejected one's.", String.raw`r(y) = \beta \log \frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}`),
  C("Refusal direction", ["refusal direction"], "A single direction in the residual stream that carries a model's decision to refuse (Arditi et al., 2024). Removing it stops refusals; adding it makes harmless requests refused. Found here as a difference of means between unsafe and safe requests' last-token vectors."),
  C("Nat", ["nat", "nats"], "The unit of a difference of natural-log probabilities. Raising an answer's log-probability by 1 nat makes it e ≈ 2.7 times as likely; by 10 nats, about 22,000 times. An answer's log-probability is the sum over its tokens, so long answers move by many nats.", String.raw`\log p' - \log p = 1 \iff p' = e\,p`),
  C("Over-refusal", ["over-refusal", "over-refuse"], "Refusing requests that are fine, because they look like ones that are not: the classic cost of safety tuning. Measured on safe requests the model never trained on."),
  C("Valence and arousal", ["valence", "arousal"], "Two classic axes for describing emotion in psychology: **valence**, pleasant to unpleasant, and **arousal**, calm to excited. The paper compares its map of emotions with them."),
];

/** Strip layer and head numbers: L3.attn.H5.weights → L·.attn.H·.weights. */
export const generic = (name: string) => name.trim().replace(/\bL\d+\b/g, "L·").replace(/\bH\d+\b/g, "H·");
const slug = (s: string) => s.replace(/·/g, "x").replace(/[^\w.-]+/g, "-");

export const GLOSSARY: GlossEntry[] = [
  ...WEIGHTS.map((e) => ({ ...e, key: `w-${slug(e.names[0])}` })),
  ...STAGES.map(([id, title]): GlossEntry => ({ key: `s-${slug(generic(id))}`, section: "Stages", title, names: [generic(id)], stage: id })),
  ...ADAPTER,
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
