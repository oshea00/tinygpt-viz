// The GPT of core/model.py, on the autograd in ./autograd, and the tracer of core/tracer.py:
// every stage passes through tap(), which records it (and later its gradient).

import { Ops, Tape, Tensor, type Node, type OpSet } from "./autograd";
import { apply, compileAll } from "./interventions";
import type { Intervention } from "../types";

export interface ModelConfig {
  n_layer: number;
  n_head: number;
  d_model: number;
  block_size: number;
  vocab_size: number;
  tied: boolean;
}

export type Weights = Map<string, Tensor>;

/** Parameters under their display names ("wte", "L0.W_q", ...) from flat arrays and shapes. */
export function makeWeights(arrays: Record<string, { data: Float32Array; shape: number[] }>): Weights {
  return new Map(Object.entries(arrays).map(([n, a]) => [n, new Tensor(a.data, a.shape)]));
}

export interface StageRecord {
  id: string;
  output: Tensor;
  grad: Float32Array | null;
}

export interface TraceResult {
  tokens: number[];
  targets: number[];
  stages: StageRecord[];
  loss: number | null;
  weightGrads: Map<string, { data: Float32Array; shape: number[] }>;
}

export interface TraceOptions {
  withGrad?: boolean;
  nextToken?: number | null;
  targetIndex?: number | null;
  interventions?: Intervention[];
}

/** The window and targets for a trace: the prompt's own next tokens, the last position's only
 * if `nextToken` is given, all but one cleared when `targetIndex` is set. */
export function prepare(cfg: ModelConfig, tokens: number[], opts: TraceOptions) {
  tokens = tokens.slice(-cfg.block_size);
  let targets = [...tokens.slice(1), opts.nextToken ?? -1];
  if (opts.targetIndex != null) {
    const keep = targets[opts.targetIndex];
    targets = targets.map((_, i) => (i === opts.targetIndex ? keep : -1));
  }
  const hasTarget = targets.some((t) => t >= 0);
  return { tokens, targets, hasTarget, withGrad: (opts.withGrad ?? true) && hasTarget };
}

/** One forward (and backward) pass of tokens on the CPU, keeping every stage. Mirrors core/tracer.py. */
export function trace(cfg: ModelConfig, W: Weights, tokens: number[], opts: TraceOptions = {}): TraceResult {
  const { tokens: idx, targets, hasTarget, withGrad } = prepare(cfg, tokens, opts);
  const tape = new Tape();
  tape.enabled = withGrad;
  for (const w of W.values()) {
    w.requiresGrad = withGrad;
    w.grad = null;
  }
  const live = new Map<string, Tensor>();
  const op = new Ops(tape);
  const loss = forward(cfg, (n) => W.get(n)!, op, idx, hasTarget ? targets : null, recorder(op, live, opts.interventions));
  if (withGrad && loss) {
    op.seed(loss);
    tape.backward();
  }

  const stages = [...live].map(([id, x]) => ({
    id, output: x,
    grad: withGrad && x.requiresGrad ? x.g() : null,   // zeros if the stage did not affect the loss
  }));
  const weightGrads = new Map<string, { data: Float32Array; shape: number[] }>();
  // As core/tracer.py: a weight cut off from the loss (by an intervention) has no gradient at all.
  if (withGrad) for (const [n, w] of W) if (w.grad) weightGrads.set(n, { data: w.grad, shape: w.shape });
  for (const w of W.values()) {
    w.requiresGrad = false;
    w.grad = null;
  }
  return { tokens: idx, targets, stages, loss: loss ? loss.data[0] : null, weightGrads };
}

/** The tap that applies interventions to a stage and keeps the result (core/tracer.py's record). */
export function recorder<T extends Node>(op: OpSet<T>, live: Map<string, T>, interventions?: Intervention[]) {
  const rewrites = compileAll<T>(interventions);
  return (id: string, x: T): T => {
    x = apply(rewrites, op, id, x);
    live.set(id, x);
    return x;
  };
}

/** The model (core/model.py GPT.forward), on either op set. Returns the loss if targets are given. */
export function forward<T extends Node>(cfg: ModelConfig, w: (name: string) => T, op: OpSet<T>, idx: number[],
                                        targets: number[] | null, tap: (id: string, x: T) => T): T | null {
  const T = idx.length, H = cfg.n_head, dh = cfg.d_model / H;
  const tok = tap("embed.tok", op.embed(w("wte"), idx));
  const pos = tap("embed.pos", op.rows(w("wpe"), 0, T));
  let x = tap("embed.sum", op.add(tok, pos));
  for (let l = 0; l < cfg.n_layer; l++) {
    const P = `L${l}`;
    const ln1 = tap(`${P}.ln1`, op.layerNorm(x, w(`${P}.ln1_g`), w(`${P}.ln1_b`)));
    const q = tap(`${P}.q`, op.linear(ln1, w(`${P}.W_q`), w(`${P}.b_q`)));
    const k = tap(`${P}.k`, op.linear(ln1, w(`${P}.W_k`), w(`${P}.b_k`)));
    const v = tap(`${P}.v`, op.linear(ln1, w(`${P}.W_v`), w(`${P}.b_v`)));
    const heads: T[] = [];
    for (let h = 0; h < H; h++) {
      const p = `${P}.attn.H${h}`;
      const qh = tap(`${p}.q`, op.cols(q, h * dh, (h + 1) * dh));
      const kh = tap(`${p}.k`, op.cols(k, h * dh, (h + 1) * dh));
      const vh = tap(`${p}.v`, op.cols(v, h * dh, (h + 1) * dh));
      const scores = tap(`${p}.scores`, op.scores(qh, kh, 1 / Math.sqrt(dh)));
      const masked = tap(`${p}.masked`, op.causalMask(scores));
      const weights = tap(`${p}.weights`, op.softmax(masked));
      heads.push(tap(`${p}.mix`, op.mix(weights, vh)));
    }
    const concat = tap(`${P}.attn.concat`, op.concatCols(heads));
    const attn = tap(`${P}.attn.out`, op.linear(concat, w(`${P}.W_o`), w(`${P}.b_o`)));
    x = tap(`${P}.resid_attn`, op.add(x, attn));
    const ln2 = tap(`${P}.ln2`, op.layerNorm(x, w(`${P}.ln2_g`), w(`${P}.ln2_b`)));
    const up = tap(`${P}.mlp.up`, op.linear(ln2, w(`${P}.W_up`), w(`${P}.b_up`)));
    const act = tap(`${P}.mlp.gelu`, op.gelu(up));
    const down = tap(`${P}.mlp.down`, op.linear(act, w(`${P}.W_down`), w(`${P}.b_down`)));
    x = tap(`${P}.resid_mlp`, op.add(x, down));
  }
  const lnf = tap("ln_f", op.layerNorm(x, w("ln_f_g"), w("ln_f_b")));
  const logits = tap("logits", cfg.tied ? op.linearT(lnf, w("wte")) : op.linear(lnf, w("W_unembed"), null));
  // probs is shown, not differentiated through: the loss reads the logits directly.
  tap("probs", op.probs(logits));
  if (!targets) return null;
  return tap("loss", op.crossEntropy(logits, targets));
}
