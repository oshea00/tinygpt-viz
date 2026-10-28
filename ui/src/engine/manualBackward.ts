// The backward pass written by hand (core/manual_backward.py), in float64 from a trace's stored
// activations, and compared with the engine's autograd stage by stage. The comparison is what
// the inspector's manual-backward view shows.

import type { TraceResult } from "./model";

interface Meta {
  id: string;
  kind: string;
  head?: number;
  inputs: string[];
  weights: string[];
}

/** A float64 matrix [r, c]. */
interface M {
  d: Float64Array;
  r: number;
  c: number;
}

const mat = (d: ArrayLike<number>, shape: number[]): M =>
  ({ d: Float64Array.from(d), r: shape.length > 1 ? shape[0] : 1, c: shape[shape.length - 1] });
const zeros = (r: number, c: number): M => ({ d: new Float64Array(r * c), r, c });

/** op(A) op(B), with op transposing when asked. */
function mm(A: M, B: M, ta = false, tb = false): M {
  const m = ta ? A.c : A.r, k = ta ? A.r : A.c, n = tb ? B.r : B.c;
  const out = zeros(m, n);
  for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) {
    const a = ta ? A.d[p * A.c + i] : A.d[i * A.c + p];
    if (a === 0) continue;
    for (let j = 0; j < n; j++) out.d[i * n + j] += a * (tb ? B.d[j * B.c + p] : B.d[p * B.c + j]);
  }
  return out;
}

function colSum(g: M): Float64Array {
  const s = new Float64Array(g.c);
  for (let i = 0; i < g.r; i++) for (let j = 0; j < g.c; j++) s[j] += g.d[i * g.c + j];
  return s;
}

// -- one function per operation -----------------------------------------------------------

/** ∂L/∂z = (p − onehot(y)) / n over kept positions. */
function crossEntropyBackward(z: M, targets: number[]): M {
  const g = zeros(z.r, z.c), kept = targets.filter((t) => t >= 0).length;
  targets.forEach((t, i) => {
    if (t < 0) return;
    let max = -Infinity, s = 0;
    for (let j = 0; j < z.c; j++) max = Math.max(max, z.d[i * z.c + j]);
    for (let j = 0; j < z.c; j++) s += Math.exp(z.d[i * z.c + j] - max);
    for (let j = 0; j < z.c; j++) g.d[i * z.c + j] = Math.exp(z.d[i * z.c + j] - max) / s / kept;
    g.d[i * z.c + t] -= 1 / kept;
  });
  return g;
}

/** y = γ x̂ + β: ∂x = (ĝ − mean ĝ − x̂ mean(ĝ x̂)) / σ with ĝ = g γ; ∂γ = Σ g x̂; ∂β = Σ g. */
function layerNormBackward(g: M, x: M, gamma: ArrayLike<number>): [M, Float64Array, Float64Array] {
  const dx = zeros(x.r, x.c), dg = new Float64Array(x.c), d = x.c;
  for (let i = 0; i < x.r; i++) {
    let mu = 0, v = 0;
    for (let j = 0; j < d; j++) mu += x.d[i * d + j] / d;
    for (let j = 0; j < d; j++) v += (x.d[i * d + j] - mu) ** 2 / d;
    const sigma = Math.sqrt(v + 1e-5);
    let mg = 0, mgx = 0;
    for (let j = 0; j < d; j++) {
      const xh = (x.d[i * d + j] - mu) / sigma, gh = g.d[i * d + j] * gamma[j];
      mg += gh / d;
      mgx += (gh * xh) / d;
      dg[j] += g.d[i * d + j] * xh;
    }
    for (let j = 0; j < d; j++) {
      const xh = (x.d[i * d + j] - mu) / sigma;
      dx.d[i * d + j] = (g.d[i * d + j] * gamma[j] - mg - xh * mgx) / sigma;
    }
  }
  return [dx, dg, colSum(g)];
}

function geluBackward(g: M, u: M): M {
  const c = Math.sqrt(2 / Math.PI), out = zeros(u.r, u.c);
  for (let i = 0; i < u.d.length; i++) {
    const x = u.d[i], t = Math.tanh(c * (x + 0.044715 * x ** 3));
    out.d[i] = g.d[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x));
  }
  return out;
}

/** Masked entries were replaced by a constant (−∞): no gradient reaches them. */
function maskBackward(g: M): M {
  const out = zeros(g.r, g.c);
  for (let i = 0; i < g.r; i++) for (let j = 0; j <= Math.min(i, g.c - 1); j++) out.d[i * g.c + j] = g.d[i * g.c + j];
  return out;
}

function softmaxBackward(g: M, a: M): M {
  const out = zeros(a.r, a.c);
  for (let i = 0; i < a.r; i++) {
    let dot = 0;
    for (let j = 0; j < a.c; j++) dot += g.d[i * a.c + j] * a.d[i * a.c + j];
    for (let j = 0; j < a.c; j++) out.d[i * a.c + j] = a.d[i * a.c + j] * (g.d[i * a.c + j] - dot);
  }
  return out;
}

const scale = (x: M, s: number): M => ({ ...x, d: x.d.map((v) => v * s) });

// -- walking the network in reverse ---------------------------------------------------------

/** Gradients for every stage and weight, by hand from the trace's activations. */
export function manualBackward(t: TraceResult, metas: Map<string, Meta>, weights: Map<string, { data: Float32Array; shape: number[] }>) {
  const out = new Map(t.stages.map((s) => [s.id, mat(s.output.data, s.output.shape)]));
  const W = (n: string) => mat(weights.get(n)!.data, weights.get(n)!.shape);
  const grad = new Map<string, M>();
  const wgrad = new Map([...weights].map(([n, w]) => [n, new Float64Array(w.data.length)]));
  const give = (id: string, g: M) => {
    const have = grad.get(id);
    if (!have) grad.set(id, { ...g, d: g.d.slice() });     // a copy: later gradients add into it
    else for (let i = 0; i < g.d.length; i++) have.d[i] += g.d[i];
  };
  const addTo = (n: string, g: ArrayLike<number>) => {
    const w = wgrad.get(n)!;
    for (let i = 0; i < g.length; i++) w[i] += g[i];
  };

  give("logits", crossEntropyBackward(out.get("logits")!, t.targets));
  for (const s of [...t.stages].reverse()) {
    const m = metas.get(s.id)!;
    const g = grad.get(s.id);
    if (!g || m.kind === "loss" || m.kind === "probs") continue;
    const ins = m.inputs, x = () => out.get(ins[0])!;
    switch (m.kind) {
      case "linear":
      case "logits": {
        const tied = m.kind === "logits" && m.weights[0] === "wte";
        const w = W(m.weights[0]);
        give(ins[0], mm(g, w, false, !tied));            // g Wᵀ (W is wteᵀ when tied)
        addTo(m.weights[0], tied ? mm(g, x(), true).d : mm(x(), g, true).d);
        if (m.weights.length > 1) addTo(m.weights[1], colSum(g));
        break;
      }
      case "norm": {
        const [dx, dg, db] = layerNormBackward(g, x(), weights.get(m.weights[0])!.data);
        give(ins[0], dx);
        addTo(m.weights[0], dg);
        addTo(m.weights[1], db);
        break;
      }
      case "activation":
        give(ins[0], geluBackward(g, x()));
        break;
      case "add":
        give(ins[0], g);                                    // the residual "highway"
        give(ins[1], g);
        break;
      case "split": {
        const src = x(), full = zeros(src.r, src.c), dh = g.c;
        for (let i = 0; i < g.r; i++) for (let j = 0; j < dh; j++) full.d[i * src.c + m.head! * dh + j] = g.d[i * dh + j];
        give(ins[0], full);
        break;
      }
      case "concat": {
        const dh = g.c / ins.length;
        ins.forEach((head, h) => {
          const part = zeros(g.r, dh);
          for (let i = 0; i < g.r; i++) for (let j = 0; j < dh; j++) part.d[i * dh + j] = g.d[i * g.c + h * dh + j];
          give(head, part);
        });
        break;
      }
      case "attn_scores": {
        const q = out.get(ins[0])!, k = out.get(ins[1])!, sd = 1 / Math.sqrt(q.c);
        give(ins[0], scale(mm(g, k), sd));
        give(ins[1], scale(mm(g, q, true), sd));
        break;
      }
      case "mask":
        give(ins[0], maskBackward(g));
        break;
      case "softmax":
        give(ins[0], softmaxBackward(g, out.get(s.id)!));
        break;
      case "attn_mix": {
        give(ins[0], mm(g, out.get(ins[1])!, false, true));
        give(ins[1], mm(out.get(ins[0])!, g, true));
        break;
      }
      case "embed": {
        const name = m.weights[0], w = wgrad.get(name)!, d = g.c;
        if (name === "wte") t.tokens.forEach((id, i) => { for (let j = 0; j < d; j++) w[id * d + j] += g.d[i * d + j]; });
        else for (let i = 0; i < g.d.length; i++) w[i] += g.d[i];
        break;
      }
    }
  }
  for (const s of t.stages) {
    const m = metas.get(s.id)!;
    if (!grad.has(s.id) && m.kind !== "loss" && m.kind !== "probs") grad.set(s.id, zeros(out.get(s.id)!.r, out.get(s.id)!.c));
  }
  return { grad, wgrad };
}

/** Hand-written against autograd, per stage and weight: max |difference| relative to the
 * gradient's largest entry. Gradients that are zero in exact arithmetic (the key bias) hold only
 * rounding noise on both sides; those are judged against the trace's largest gradient. */
export function compare(t: TraceResult, metas: Map<string, Meta>, weights: Map<string, { data: Float32Array; shape: number[] }>) {
  const { grad, wgrad } = manualBackward(t, metas, weights);
  const absMax = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };
  const refs = [...t.stages.flatMap((s) => (s.grad ? [s.grad] : [])), ...[...t.weightGrads.values()].map((g) => g.data)];
  const largest = Math.max(0, ...refs.map(absMax));
  const floor = 1e-6 * largest;
  const stats = (mine: ArrayLike<number>, ref: ArrayLike<number>) => {
    let diff = 0;
    for (let i = 0; i < ref.length; i++) diff = Math.max(diff, Math.abs(mine[i] - ref[i]));
    const s = absMax(ref), nearZero = s < floor, denom = nearZero ? largest : s;
    return { maxAbsDiff: diff, relative: denom > 0 ? diff / denom : diff, nearZero };
  };
  const stages = Object.fromEntries(t.stages.filter((s) => s.grad && grad.has(s.id)).map((s) => [s.id, stats(grad.get(s.id)!.d, s.grad!)]));
  const weightStats = Object.fromEntries([...t.weightGrads].map(([n, g]) => [n, stats(wgrad.get(n)!, g.data)]));
  const worst = Math.max(0, ...[...Object.values(stages), ...Object.values(weightStats)].map((v) => v.relative));
  return { stages, weights: weightStats, worstRelative: worst };
}
