// Interventions (core/interventions.py): change one thing inside the forward pass. Each spec
// rewrites one stage's output on the way through the model's tap, with ops that carry the
// right gradient (an ablated unit passes none back).

import type { Node, OpSet } from "./autograd";
import type { Intervention } from "../types";

export type Rewrite<T extends Node> = (op: OpSet<T>, x: T & { shape: number[] }) => T;

/** mulberry32: a small seeded generator that core/interventions.py reproduces exactly. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seeded Fisher–Yates permutation of 0..n-1, the same as core/interventions.py's. */
export function permutation(n: number, seed: number): number[] {
  const perm = Array.from({ length: n }, (_, i) => i);
  const rand = mulberry32(seed);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  return perm;
}

/** One spec → the stage it rewrites and the rewrite. */
function compile<T extends Node>(spec: Intervention): [string, Rewrite<T>] {
  switch (spec.kind) {
    case "head":
      return [`L${spec.layer}.attn.H${spec.head}.mix`, (op, x) => op.replaceCols(x, 0, x.shape[1], spec.mode ?? "zero")];
    case "neuron":
      return [`L${spec.layer}.mlp.gelu`, (op, x) => op.replaceCols(x, spec.index, spec.index + 1, spec.mode ?? "zero")];
    case "block":
      return [`L${spec.layer}.${{ attn: "attn.out", mlp: "mlp.down" }[spec.part]}`,
              (op, x) => op.constant(new Float32Array(x.shape[0] * x.shape[1]), x.shape)];
    case "attend":
      // Each query attends only to `position`, or to itself if that is in its future.
      return [`L${spec.layer}.attn.H${spec.head}.weights`, (op, x) => {
        const T = x.shape[1], w = new Float32Array(x.shape[0] * T);
        for (let i = 0; i < x.shape[0]; i++) w[i * T + Math.min(spec.position, i)] = 1;
        return op.constant(w, x.shape);
      }];
    case "set":
      return [spec.stage, (op, x) => op.overwrite(x, [spec.row * x.shape[1] + spec.col], [spec.value])];
    case "steer":
      return [spec.stage, (op, x) => op.addRow(x, spec.vector, spec.scale ?? 1)];
    case "patch":
      return [spec.stage, (op, x) => {
        const n = x.shape[1];
        return op.overwrite(x, spec.positions.flatMap((p) => Array.from({ length: n }, (_, j) => p * n + j)), spec.values.flat());
      }];
    case "shuffle_pos":
      return ["embed.pos", (op, x) => op.permuteRows(x, permutation(x.shape[0], spec.seed ?? 0))];
    default:
      throw new Error(`unknown intervention ${JSON.stringify((spec as { kind: string }).kind)}`);
  }
}

/** Specs grouped by the stage they rewrite, in order. */
export function compileAll<T extends Node>(specs: Intervention[] | undefined): Map<string, Rewrite<T>[]> {
  const byStage = new Map<string, Rewrite<T>[]>();
  for (const spec of specs ?? []) {
    const [stage, rewrite] = compile<T>(spec);
    byStage.set(stage, [...(byStage.get(stage) ?? []), rewrite]);
  }
  return byStage;
}

export function apply<T extends Node>(byStage: Map<string, Rewrite<T>[]>, op: OpSet<T>, stage: string, x: T): T {
  for (const rewrite of byStage.get(stage) ?? []) x = rewrite(op, x as T & { shape: number[] });
  return x;
}
