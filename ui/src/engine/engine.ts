// What the in-browser backend computes with: the model on the CPU (./model) or on WebGPU
// (./gpu), behind one interface. Analyses (./analyses) are written against it once.

import { Ops, Tape, type Node, type OpSet } from "./autograd";
import { forward, trace, type ModelConfig, type TraceOptions, type TraceResult, type Weights } from "./model";
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
}

export interface Engine {
  readonly name: string;
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

  async forward(tokens: number[], opts: ForwardOptions): Promise<Record<string, Float32Array>> {
    const tape = new Tape();
    tape.enabled = false;
    const op = new Ops(tape);
    const live = new Map();
    const W = this.sets.get(opts.weights ?? "current")!;
    forward(this.cfg, (n) => W.get(n)!, op, tokens.slice(-this.cfg.block_size), null,
            forwardTap(op, live, opts.interventions, opts.rewrite));
    return Object.fromEntries(Object.entries(opts.read(live, op, (n) => W.get(n)!)).map(([k, t]) => [k, t.data]));
  }
}
