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
}

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
}

export interface Hello extends Status {
  type: "hello";
  config: ModelConfig;
  vocab: string[];
  params: number;
  paramGroups: Record<string, number>;
  checkpoints: number[];
  probe: string;
}

export type Command =
  | { type: "run"; prompt: string; withGrad?: boolean; targetIndex?: number }
  | { type: "train"; action: "play" | "pause" | "step" | "reset" }
  | { type: "set"; lr?: number; batchSize?: number; traceEvery?: number; probe?: string }
  | { type: "sample"; prompt?: string; temperature: number; topK?: number; topP?: number; n: number }
  | { type: "loadCheckpoint"; step: number }
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

export type JsonMsg =
  | Hello
  | MetricsMsg
  | SampleMsg
  | ({ type: "status" } & Status)
  | { type: "reset" }
  | { type: "error"; message: string };
