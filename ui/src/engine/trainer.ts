// Training in the browser (core/train.py's Trainer): fresh weights, batches from train.bin, the
// learning-rate warmup and the checkpoint schedule. The engine does the steps (Engine.trainStep);
// LocalModel drives this from the training panel's commands.

import { Tensor } from "./autograd";
import { mulberry32 } from "./interventions";
import type { ModelConfig, Weights } from "./model";

/** core/config.py's TrainConfig. */
export interface TrainConfig {
  lr: number;
  batchSize: number;
  warmupSteps: number;
  weightDecay: number;
  gradClip: number;
  traceEvery: number;
  evalEvery: number;
  evalBatches: number;
  seed: number;
}

export const TRAIN_DEFAULTS: TrainConfig = {
  lr: 6e-4, batchSize: 128, warmupSteps: 100, weightDecay: 0.1, gradClip: 1.0,
  traceEvery: 50, evalEvery: 250, evalBatches: 20, seed: 1337,
};

/** The learning rate for the step after `step`: linear warmup, then constant. */
export const lrAt = (c: TrainConfig, step: number) => c.lr * Math.min(1, (step + 1) / Math.max(1, c.warmupSteps));

/** Roughly log-spaced steps from 0, so the replay slider sees early learning in detail. */
export function checkpointSchedule(maxStep = 20000, n = 30): Set<number> {
  const steps = new Set([0]);
  for (let i = 1; i < n; i++) steps.add(Math.round(Math.exp((Math.log(maxStep) * i) / (n - 1))));
  return steps;
}

/** Standard normal draws from a seeded generator (Box–Muller). */
export function normals(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = 1 - rand(), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** Random initial weights, as core/model.py's GPT: N(0, 0.02) matrices, with the residual writers
 * (W_o, W_down) smaller, layer-norm gains 1 and biases 0. Named as GPT.named_weights(). */
export function initWeights(cfg: ModelConfig & { d_mlp: number }, seed: number): Weights {
  const randn = normals(mulberry32(seed));
  const W: Weights = new Map();
  const normal = (name: string, shape: number[], std = 0.02) =>
    W.set(name, new Tensor(Float32Array.from({ length: shape[0] * shape[1] }, () => randn() * std), shape));
  const fill = (name: string, n: number, v: number) => W.set(name, new Tensor(new Float32Array(n).fill(v), [n]));
  const d = cfg.d_model, m = cfg.d_mlp, proj = 0.02 / Math.sqrt(2 * cfg.n_layer);
  normal("wte", [cfg.vocab_size, d]);
  normal("wpe", [cfg.block_size, d]);
  for (let l = 0; l < cfg.n_layer; l++) {
    const P = `L${l}`;
    fill(`${P}.ln1_g`, d, 1);
    fill(`${P}.ln1_b`, d, 0);
    for (const x of ["q", "k", "v"]) {
      normal(`${P}.W_${x}`, [d, d]);
      fill(`${P}.b_${x}`, d, 0);
    }
    normal(`${P}.W_o`, [d, d], proj);
    fill(`${P}.b_o`, d, 0);
    fill(`${P}.ln2_g`, d, 1);
    fill(`${P}.ln2_b`, d, 0);
    normal(`${P}.W_up`, [d, m]);
    fill(`${P}.b_up`, m, 0);
    normal(`${P}.W_down`, [m, d], proj);
    fill(`${P}.b_down`, d, 0);
  }
  fill("ln_f_g", d, 1);
  fill("ln_f_b", d, 0);
  if (!cfg.tied) normal("W_unembed", [d, cfg.vocab_size]);
  return W;
}

/** Random windows of a token stream: x and its next tokens y, B sequences of T, flat. */
export function batch(data: Uint16Array, B: number, T: number, rand: () => number): { x: number[]; y: number[] } {
  const x: number[] = [], y: number[] = [];
  for (let b = 0; b < B; b++) {
    const i = Math.floor(rand() * (data.length - T - 1));
    for (let t = 0; t < T; t++) {
      x.push(data[i + t]);
      y.push(data[i + t + 1]);
    }
  }
  return { x, y };
}

/** A copy of a weight set (checkpoints must not move with the live weights). */
export const copyWeights = (W: Weights): Weights => new Map([...W].map(([n, w]) => [n, new Tensor(w.data.slice(), w.shape)]));
