// What the in-browser backend computes with: the model on the CPU (./model) or on WebGPU
// (./gpu), behind one interface. Analyses (./analyses) are written against it once.

import { Ops, Tape, Tensor, type Node, type OpSet } from "./autograd";
import { forward, trace, type KV, type ModelConfig, type TraceOptions, type TraceResult, type Weights } from "./model";
import { apply, compileAll } from "./interventions";
import type { FrameSpec } from "./gpu";
import { encodeTrace } from "./frames";
import type { Intervention } from "../types";

/** Rewrites a stage on the way through, after any interventions (occlusion, patching...). */
export type Rewrite = <T extends Node>(id: string, x: T, op: OpSet<T>) => T;

/** Chooses (or computes, with the op set and the weights) the tensors a forward pass returns. */
export type Read = <T extends Node>(live: Map<string, T>, op: OpSet<T>, weight: (name: string) => T) => Record<string, T>;

/** Which weights a forward pass uses: the loaded checkpoint, or step 0 (the untrained baseline). */
export type WeightSet = "current" | "untrained";

export interface ForwardOptions {
  weights?: WeightSet;
  interventions?: Intervention[];
  rewrite?: Rewrite;
  read: Read;
  /** The KV cache: each layer's keys and values ([cached positions, d]) for the positions before
   * `tokens`, which then continue from there. */
  past?: { k: Float32Array; v: Float32Array }[];
  /** Also return every layer's keys and values (cached plus new), as "kv.L<l>.k" and ".v". */
  keepKV?: boolean;
}

/** One AdamW step's settings, as core/train.py's optimizer (betas 0.9, 0.99; decay on matrices). */
export interface StepSettings {
  lr: number;
  weightDecay: number;
  gradClip: number;
}

export const BETAS = [0.9, 0.99] as const;
export const ADAM_EPS = 1e-8;

export interface Engine {
  readonly name: string;
  /** One training step on B sequences (x and y flat, B · T each): forward, backward, clipping
   * the gradient's global norm, then AdamW. Returns the loss and the norm before clipping. */
  trainStep(x: number[], y: number[], B: number, s: StepSettings): Promise<{ loss: number; gradNorm: number }>;
  /** The mean loss on B sequences, without gradients (the validation curve). */
  loss(x: number[], y: number[], B: number): Promise<number>;
  /** The current weights, and the last step's (clipped) gradients, on the host. */
  readWeights(): Promise<Weights>;
  readGrads(): Promise<Weights>;
  /** Fresh optimizer moments, as after loading a checkpoint or a reset. */
  resetOptimizer(): void;
  setWeights(W: Weights, set?: WeightSet): void;
  hasWeights(set: WeightSet): boolean;
  /** A trace with its frame (see core/trace_io.py). */
  trace(tokens: number[], opts: TraceOptions, frame: FrameSpec): Promise<TraceResult & { frame: ArrayBuffer }>;
  /** A forward pass without gradients, returning only what `read` asks for. */
  forward(tokens: number[], opts: ForwardOptions): Promise<Record<string, Float32Array>>;
}

/** The engine with every forward pass on `weights` (the untrained baseline, say). */
export function using(e: Engine, weights: WeightSet): Engine {
  if (weights === "current") return e;
  return Object.assign(Object.create(e) as Engine, {
    forward: (tokens: number[], opts: ForwardOptions) => e.forward(tokens, { ...opts, weights }),
  });
}

/** One forward pass on any op set, with the cache and interventions of `opts`; returns what
 * `opts.read` asks for, plus the keys and values with `keepKV`. Both engines run this. */
export function runForward<T extends Node>(cfg: ModelConfig, W: Map<string, T>, op: OpSet<T>, tokens: number[],
                                           opts: ForwardOptions): Record<string, T> {
  const live = new Map<string, T>();
  const t0 = opts.past?.[0].k.length ? opts.past[0].k.length / cfg.d_model : 0;
  const kv: KV<T> = {
    past: t0 ? opts.past!.map((p) => [op.constant(p.k, [t0, cfg.d_model]), op.constant(p.v, [t0, cfg.d_model])]) : undefined,
    keep: opts.keepKV ? [] : undefined,
  };
  forward(cfg, (n) => W.get(n)!, op, tokens.slice(-(cfg.block_size - t0)), null,
          forwardTap(op, live, opts.interventions, opts.rewrite), { kv });
  const out = opts.read(live, op, (n) => W.get(n)!);
  kv.keep?.forEach(([k, v], l) => Object.assign(out, { [`kv.L${l}.k`]: k, [`kv.L${l}.v`]: v }));
  return out;
}

/** The tap for a forward pass: interventions, then `rewrite`, then keep the stage. */
export function forwardTap<T extends Node>(op: OpSet<T>, live: Map<string, T>, interventions?: Intervention[], rewrite?: Rewrite) {
  const rewrites = compileAll<T>(interventions);
  return (id: string, x: T): T => {
    x = apply(rewrites, op, id, x);
    if (rewrite) x = rewrite(id, x, op);
    live.set(id, x);
    return x;
  };
}

export class CpuEngine implements Engine {
  readonly name = "browser CPU";
  private sets = new Map<WeightSet, Weights>();
  private adam = { t: 0, m: new Map<string, Float32Array>(), v: new Map<string, Float32Array>() };
  private grads: Weights = new Map();

  constructor(private cfg: ModelConfig) {}

  private get W(): Weights {
    return this.sets.get("current") ?? new Map();
  }

  setWeights(W: Weights, set: WeightSet = "current"): void {
    this.sets.set(set, W);
  }

  hasWeights(set: WeightSet): boolean {
    return this.sets.has(set);
  }

  async trace(tokens: number[], opts: TraceOptions, frame: FrameSpec) {
    const t = trace(this.cfg, this.W, tokens, opts);
    return { ...t, frame: encodeTrace(t, frame.meta, frame.step, frame.text(t.tokens), frame.extra) };
  }

  async trainStep(x: number[], y: number[], B: number, s: StepSettings) {
    const W = this.W, tape = new Tape(), op = new Ops(tape);
    for (const w of W.values()) {
      w.requiresGrad = true;
      w.grad = null;
    }
    const loss = forward(this.cfg, (n) => W.get(n)!, op, x, y, (_, t) => t, { B })!;
    op.seed(loss);
    tape.backward();
    let sq = 0;
    for (const w of W.values()) for (const g of w.g()) sq += g * g;
    const gradNorm = Math.sqrt(sq);
    const coef = Math.min(1, s.gradClip / (gradNorm + 1e-6));     // torch.nn.utils.clip_grad_norm_
    const a = this.adam, [b1, b2] = BETAS;
    a.t++;
    const bc1 = 1 - b1 ** a.t, bc2 = Math.sqrt(1 - b2 ** a.t);
    this.grads = new Map();
    for (const [n, w] of W) {
      const p = w.data, g = w.g();
      const m = a.m.get(n) ?? a.m.set(n, new Float32Array(p.length)).get(n)!;
      const v = a.v.get(n) ?? a.v.set(n, new Float32Array(p.length)).get(n)!;
      const decay = w.shape.length >= 2 ? s.weightDecay : 0;          // matrices decay; biases and gains do not
      for (let i = 0; i < p.length; i++) {
        g[i] *= coef;
        p[i] *= 1 - s.lr * decay;
        m[i] += (g[i] - m[i]) * (1 - b1);
        v[i] = v[i] * b2 + (1 - b2) * g[i] * g[i];
        p[i] -= (s.lr / bc1) * m[i] / (Math.sqrt(v[i]) / bc2 + ADAM_EPS);
      }
      this.grads.set(n, new Tensor(g, w.shape));
      w.requiresGrad = false;
      w.grad = null;
    }
    return { loss: loss.data[0], gradNorm };
  }

  async loss(x: number[], y: number[], B: number): Promise<number> {
    const tape = new Tape();
    tape.enabled = false;
    return forward(this.cfg, (n) => this.W.get(n)!, new Ops(tape), x, y, (_, t) => t, { B })!.data[0];
  }

  async readWeights(): Promise<Weights> {
    return this.W;
  }

  async readGrads(): Promise<Weights> {
    return this.grads;
  }

  resetOptimizer(): void {
    this.adam = { t: 0, m: new Map(), v: new Map() };
  }

  async forward(tokens: number[], opts: ForwardOptions): Promise<Record<string, Float32Array>> {
    const tape = new Tape();
    tape.enabled = false;
    const out = runForward(this.cfg, this.sets.get(opts.weights ?? "current")!, new Ops(tape), tokens, opts);
    return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, t.data]));
  }
}
