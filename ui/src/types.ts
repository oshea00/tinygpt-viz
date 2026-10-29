// The trace contract shared with core/trace_io.py.

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export type StageKind =
  | "embed" | "norm" | "linear" | "split" | "attn_scores" | "mask" | "softmax"
  | "attn_mix" | "concat" | "add" | "activation" | "logits" | "probs" | "loss" | "lora";

export interface Stage {
  id: string;
  kind: StageKind;
  layer?: number;
  head?: number;
  output: Tensor;
  grad?: Tensor;
  inputs: string[];
  weights: string[];
  formula: string;
  explain: string;
  backward: string;      // TeX for the stage's backward rule
}

export interface Trace {
  step: number;
  loss: number | null;
  tokens: number[];
  text: string[];
  targets: number[];     // next-token target per position, -1 where there is none
  stages: Stage[];
  byId: Map<string, Stage>;
  weightGrads: Record<string, Tensor>;
  prompt?: string;
  reason?: string;
  weightsVersion?: number;
  tag?: string;                     // "baseline": the unmodified run compared against
  incremental?: Incremental;        // set when the trace came from the KV cache
  interventions?: Intervention[];   // what was changed in this trace
  adapter?: AdapterChoice | null;   // the low-rank adapter switched on, if any
}

/** Which low-rank adapter (core/lora.py) a command runs with, and how strongly (1: as trained). */
export interface AdapterChoice {
  name: string;
  scale: number;
}

/** An adapter as the server describes it (core/lora.py's Adapter.header). */
export interface AdapterHeader {
  name: string;
  rank: number;
  targets: string[];          // stage ids it adds to ("L3.q")
  baseStep: number;           // the checkpoint it was trained on
  params: number;
  version: number;            // bumps as it trains
  info: AdapterInfo;
}

/** One point of an adapter's training curve: the step's loss, and every few steps the measures. */
export interface LoraPoint {
  step: number;
  loss?: number;
  trainLoss?: number;         // on its stories
  heldOutLoss?: number;       // on ordinary validation stories
  pTopic?: number;            // mean p(topic word) after the neutral openers
}

/** How an adapter was trained: live with LoRA on stories, or offline with DPO (core/dpo.py). */
export type AdapterInfo = Record<string, any> & { kind?: "lora" | "dpo"; curve?: LoraPoint[]; topic?: string; steps?: number };

/** The "adapter" frame: an adapter's header and its A and B tensors ("L3.q.lora_A"...). */
export interface AdapterMsg extends AdapterHeader {
  tensors: Record<string, Tensor>;
}

/** How a KV-cache trace was made (core/kvcache.py): rows before `reused` came from the cache. */
export interface Incremental {
  reused: number;
  computed: number;
  reason: string | null;          // why the cache could not be used, if it was not
  multiplyAdds: number;
  multiplyAddsFull: number;       // what a full pass over the same positions would cost
}

/** Changes to the forward pass (core/interventions.py). */
export type Intervention =
  | { kind: "head"; layer: number; head: number; mode: "zero" | "mean" }
  | { kind: "neuron"; layer: number; index: number; mode: "zero" | "mean" }
  | { kind: "block"; layer: number; part: "attn" | "mlp" }
  | { kind: "attend"; layer: number; head: number; position: number }
  | { kind: "set"; stage: string; row: number; col: number; value: number }
  | { kind: "shuffle_pos"; seed: number }
  | { kind: "steer"; stage: string; vector: number[]; scale: number; label?: string }
  | { kind: "patch"; stage: string; positions: number[]; values: number[][]; label?: string };

export interface Weights {
  step: number;
  version: number;
  tensors: Record<string, Tensor>;
}

export interface StepExplain {
  step: number;
  prompt: string;
  text: string[];
  lossBefore: number;
  lossAfter: number;
  pBefore: number[];
  pAfter: number[];
  lr: number;
  before: Record<string, Tensor>;
  grad: Record<string, Tensor>;
  delta: Record<string, Tensor>;
}

export interface ModelConfig {
  name: string;
  n_layer: number;
  n_head: number;
  d_model: number;
  d_mlp: number;
  block_size: number;
  vocab_size: number;
  tied: boolean;
}

export interface Status {
  step: number;
  playing: boolean;
  lr: number;
  batchSize: number;
  traceEvery: number;
  device: string;
  checkpoints: number[];
}

export interface Hello extends Status {
  type: "hello";
  config: ModelConfig;
  vocab: string[];
  params: number;
  paramGroups: Record<string, number>;
  probe: string;
  training: boolean;      // the server was started with --train
  adapters?: AdapterHeader[];
}

export type Command =
  | { type: "run"; prompt: string; withGrad?: boolean; targetIndex?: number; interventions?: Intervention[]; tag?: string; incremental?: boolean;
      adapter?: AdapterChoice | null }
  | { type: "train"; action: "play" | "pause" | "step" | "reset" }
  | { type: "set"; lr?: number; batchSize?: number; traceEvery?: number; probe?: string }
  | { type: "sample"; prompt?: string; temperature: number; topK?: number; topP?: number; n: number; adapter?: AdapterChoice | null }
  | { type: "loadCheckpoint"; step: number }
  | { type: "save" }
  | { type: "manualBackward"; prompt: string }
  | { type: "lens"; prompt: string; adapter?: AdapterChoice | null }
  | { type: "view"; weightGrads: boolean }   // whether traces carry the weights' gradients
  | { type: "capture"; prompts: { text: string; targets: { start: number; end: number; word: string; category: string }[] }[];
      pool: "last" | "mean"; model: "current" | "untrained"; tag?: string; adapter?: AdapterChoice | null }
  | { type: "generateWith"; prompt: string; n: number; interventions?: Intervention[]; temperature?: number; seed?: number; tag?: string;
      adapter?: AdapterChoice | null }
  | { type: "occlude"; prompt: string; position: number }
  | { type: "neuronTop"; layer: number; neuron: number }
  | { type: "patch"; clean: string; corrupt: string; answer: string; wrong: string; what: "resid" | "heads" }
  | { type: "stepExplain"; prompt?: string }
  | { type: "beliefs"; stories?: string[]; n?: number; cloze: string; words: string[]; model: "current" | "untrained" }
  | { type: "beliefSteer"; stories: string[]; cloze: string; words: string[]; interventions: Intervention[]; tag: string }
  | { type: "loraTrain"; name: string; texts: string[]; topic: string; rank?: number; targets?: string[]; steps?: number;
      lr?: number; batch?: number; seed?: number }
  | { type: "loraStop" }
  | { type: "adapterWeights"; name: string }
  | { type: "scorePairs"; name: string; scale?: number; pairs: { prompt: string; chosen: string; rejected: string }[]; beta: number; tag?: string };

/** Commands that can run with an adapter switched on. */
export const ADAPTED_COMMANDS = new Set(["run", "sample", "lens", "capture", "generateWith", "occlude", "neuronTop", "patch",
                                        "beliefs", "beliefSteer"]);

export interface MetricsMsg {
  type: "metrics";
  step: number;
  points: [number, number][];
  tokensPerSec: number;
  valLoss?: number;
}

export interface SampleMsg {
  type: "sample";
  step: number;
  text: string;
  auto?: boolean;
  prompt?: string;
  tokens?: string[];
  chosenProbs?: number[];
  probs?: number[];
}

export interface GradCheck {
  maxAbsDiff: number;
  relative: number;
  nearZero: boolean;     // zero in exact arithmetic; both sides hold only rounding noise
}

export interface ManualBackwardMsg {
  type: "manualBackward";
  prompt: string;
  step: number;
  stages: Record<string, GradCheck>;
  weights: Record<string, GradCheck>;
  worstRelative: number;
}

/** Server-side logit lens summary (core/lens.py), used for large vocabularies. */
export interface LensMsg {
  type: "lens";
  prompt: string;
  step: number;
  snapshots: string[];
  topIds: number[][][];    // [snapshot][position][k]
  topP: number[][][];
  pActual: number[][];     // [snapshot][position], −1 where there is no next token
}

/** An activation-patching sweep (core/concepts.py patch_sweep). */
export interface PatchResult {
  clean: string;
  corrupt: string;
  cleanText: string[];
  corruptText: string[];
  what: "resid" | "heads";
  stages: string[];          // the stage each grid row (resid) or cell (heads, layer-major) patches
  answer: string;            // the clean prompt's answer token, and the token it competes with
  wrong: string;
  ldClean: number;           // logit[answer] − logit[wrong] at the last position, in each run
  ldCorrupt: number;
  step: number;
  grid: Tensor;              // recovery: [snapshots, positions] or [layers, heads]
  source: Tensor;            // the clean activations: [snapshots, T, d] or [layers, heads, T, d_head]
}

export interface NeuronExample { value: number; before: string; token: string; after: string; }

export type ConceptMsg =
  | { type: "generated"; tag?: string; prompt: string; text: string }
  | { type: "occlusion"; prompt: string; position: number; step: number; drops: number[] }
  | { type: "neuronTop"; layer: number; neuron: number; step: number; examples: NeuronExample[] }
  | { type: "beliefSteer"; tag: string; readout: number[][]; mass: number[] };

/** Preference pairs scored with an adapter and without it (core/dpo.py's score). */
export interface PairScoresMsg {
  type: "pairScores";
  name: string;
  scale: number;
  tag?: string;
  ref_chosen: number[];
  ref_rejected: number[];
  policy_chosen: number[];
  policy_rejected: number[];
  margin: number[];          // β · ((log π − log π_ref)(chosen) − (log π − log π_ref)(rejected))
  beta: number;
}

export interface LoraProgressMsg {
  type: "loraProgress";
  name: string;
  steps: number;
  running: boolean;
  points: LoraPoint[];
}

export type JsonMsg =
  | ConceptMsg
  | LoraProgressMsg
  | PairScoresMsg
  | ManualBackwardMsg
  | LensMsg
  | Hello
  | MetricsMsg
  | SampleMsg
  | ({ type: "status" } & Status)
  | { type: "reset" }
  | { type: "error"; message: string };
