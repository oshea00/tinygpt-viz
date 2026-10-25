// The trace contract shared with core/trace_io.py.

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export type StageKind =
  | "embed" | "norm" | "linear" | "split" | "attn_scores" | "mask" | "softmax"
  | "attn_mix" | "concat" | "add" | "activation" | "logits" | "probs" | "loss";

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
  interventions?: Intervention[];   // what was changed in this trace
}

/** Changes to the forward pass (core/interventions.py). */
export type Intervention =
  | { kind: "head"; layer: number; head: number; mode: "zero" | "mean" }
  | { kind: "neuron"; layer: number; index: number; mode: "zero" | "mean" }
  | { kind: "block"; layer: number; part: "attn" | "mlp" }
  | { kind: "attend"; layer: number; head: number; position: number }
  | { kind: "set"; stage: string; row: number; col: number; value: number }
  | { kind: "shuffle_pos"; seed: number }
  | { kind: "steer"; stage: string; vector: number[]; scale: number; label?: string };

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
  tokenizer: "char" | "bpe";
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
}

export type Command =
  | { type: "run"; prompt: string; withGrad?: boolean; targetIndex?: number; interventions?: Intervention[]; tag?: string }
  | { type: "train"; action: "play" | "pause" | "step" | "reset" }
  | { type: "set"; lr?: number; batchSize?: number; traceEvery?: number; probe?: string }
  | { type: "sample"; prompt?: string; temperature: number; topK?: number; topP?: number; n: number }
  | { type: "loadCheckpoint"; step: number }
  | { type: "save" }
  | { type: "manualBackward"; prompt: string }
  | { type: "lens"; prompt: string }
  | { type: "capture"; prompts: { text: string; targets: { start: number; end: number; word: string; category: string }[] }[];
      pool: "last" | "mean"; model: "current" | "untrained"; tag?: string }
  | { type: "generateWith"; prompt: string; n: number; interventions?: Intervention[]; temperature?: number; seed?: number; tag?: string }
  | { type: "occlude"; prompt: string; position: number }
  | { type: "neuronTop"; layer: number; neuron: number }
  | { type: "stepExplain"; prompt?: string };

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

export interface NeuronExample { value: number; before: string; token: string; after: string; }

export type ConceptMsg =
  | { type: "generated"; tag?: string; prompt: string; text: string }
  | { type: "occlusion"; prompt: string; position: number; step: number; drops: number[] }
  | { type: "neuronTop"; layer: number; neuron: number; step: number; examples: NeuronExample[] };

export type JsonMsg =
  | ConceptMsg
  | ManualBackwardMsg
  | LensMsg
  | Hello
  | MetricsMsg
  | SampleMsg
  | ({ type: "status" } & Status)
  | { type: "reset" }
  | { type: "error"; message: string };
