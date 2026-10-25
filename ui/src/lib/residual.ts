// The residual stream: its snapshots, what each sublayer writes to it, how gradients flow
// back through it, and the logit lens (reading a prediction out of every snapshot).

import type { Trace, Weights } from "../types";
import { dims2, norm, row } from "./tensor";

export interface Snapshot {
  id: string;        // stage id holding x at this point
  label: string;
}

/** x₀, then x after each attention and each MLP: 1 + 2·layers snapshots. */
export function snapshots(trace: Trace): Snapshot[] {
  const out: Snapshot[] = [{ id: "embed.sum", label: "x₀ embed" }];
  for (let l = 0; trace.byId.has(`L${l}.resid_attn`); l++) {
    out.push({ id: `L${l}.resid_attn`, label: `L${l} +attn` }, { id: `L${l}.resid_mlp`, label: `L${l} +mlp` });
  }
  return out;
}

export interface Write {
  label: string;
  delta: string;     // stage id of what this sublayer adds to the stream
  before: string;    // residual snapshot it reads (and adds to)
  after: string;     // residual snapshot it produces
}

export function writes(trace: Trace): Write[] {
  const snaps = snapshots(trace);
  return snaps.slice(1).map((s, i) => {
    const attn = s.id.endsWith("resid_attn");
    const layer = s.id.split(".")[0];
    return { label: `${layer} ${attn ? "attn" : "mlp"}`, delta: attn ? `${layer}.attn.out` : `${layer}.mlp.down`, before: snaps[i].id, after: s.id };
  });
}

export const rowNorm = (t: { shape: number[]; data: Float32Array }, r: number) => norm(row(t, r));

export function tensorNorm(t: { data: Float32Array }): number {
  return norm(t.data);
}

export interface GradFlow {
  label: string;
  through: number;   // ‖∂L/∂x‖ after the add: passes straight back through it
  branch: number;    // ‖what the sublayer adds on the way back‖
  total: number;     // ‖∂L/∂x‖ before the add
}

/** At x' = x + f(x), ∂L/∂x = ∂L/∂x' + (the path back through f). Norms per sublayer, for one
 * position or (pos = -1) the whole sequence, from the last layer back to the first. */
export function gradientHighway(trace: Trace, pos: number): GradFlow[] | null {
  const out: GradFlow[] = [];
  for (const w of writes(trace).reverse()) {
    const after = trace.byId.get(w.after)!.grad;
    const before = trace.byId.get(w.before)!.grad;
    if (!after || !before) return null;
    const a = pos >= 0 ? row(after, pos) : after.data;
    const b = pos >= 0 ? row(before, pos) : before.data;
    const branch = Array.from(b, (v, i) => v - a[i]);
    out.push({ label: w.label, through: norm(a), branch: norm(branch), total: norm(b) });
  }
  return out;
}

export interface Lens {
  snaps: Snapshot[];
  T: number;
  V: number;
  probs: Float32Array;   // [snapshot, position, vocab]
}

/** Apply the final layer norm and unembedding to every residual snapshot. */
export function logitLens(trace: Trace, weights: Weights, eps = 1e-5): Lens {
  const snaps = snapshots(trace);
  const g = weights.tensors["ln_f_g"].data;
  const b = weights.tensors["ln_f_b"].data;
  const tied = !weights.tensors["W_unembed"];
  const U = tied ? weights.tensors["wte"] : weights.tensors["W_unembed"];
  const V = tied ? U.shape[0] : U.shape[1];
  const [T, d] = dims2(trace.byId.get("embed.sum")!.output);
  const probs = new Float32Array(snaps.length * T * V);
  const xhat = new Float64Array(d);
  const z = new Float64Array(V);
  snaps.forEach((s, si) => {
    const X = trace.byId.get(s.id)!.output;
    for (let t = 0; t < T; t++) {
      const x = row(X, t);
      let mean = 0;
      for (let i = 0; i < d; i++) mean += x[i];
      mean /= d;
      let v = 0;
      for (let i = 0; i < d; i++) v += (x[i] - mean) ** 2;
      const sd = Math.sqrt(v / d + eps);
      for (let i = 0; i < d; i++) xhat[i] = ((x[i] - mean) / sd) * g[i] + b[i];
      let max = -Infinity;
      for (let k = 0; k < V; k++) {
        let s2 = 0;
        for (let i = 0; i < d; i++) s2 += xhat[i] * (tied ? U.data[k * d + i] : U.data[i * V + k]);
        z[k] = s2;
        if (s2 > max) max = s2;
      }
      let total = 0;
      for (let k = 0; k < V; k++) total += (z[k] = Math.exp(z[k] - max));
      const base = (si * T + t) * V;
      for (let k = 0; k < V; k++) probs[base + k] = z[k] / total;
    }
  });
  return { snaps, T, V, probs };
}

export function lensRow(lens: Lens, s: number, t: number): Float32Array {
  const base = (s * lens.T + t) * lens.V;
  return lens.probs.subarray(base, base + lens.V);
}

export function argmax(a: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}
