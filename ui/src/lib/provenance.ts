// "Where did this number come from?" Recomputes one cell of a stage from its inputs and
// weights, showing the working. The result should equal the stored value; the inspector
// prints both so a learner can check.

import type { Stage, Tensor, Trace, Weights } from "../types";
import { at, col, dims2, row } from "./tensor";

export interface Strip {
  label: string;
  values: number[];
}

export interface Provenance {
  title: string;
  strips: Strip[];
  lines: [string, number | string][];
  value: number;
  stored: number;
}

const EPS = 1e-5;

const gelu = (u: number) => 0.5 * u * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (u + 0.044715 * u ** 3)));

function input(trace: Trace, stage: Stage, i = 0): Tensor {
  const s = trace.byId.get(stage.inputs[i]);
  if (!s) throw new Error(`missing input ${stage.inputs[i]} of ${stage.id}`);
  return s.output;
}

function weight(w: Weights, name: string): Tensor {
  const t = w.tensors[name];
  if (!t) throw new Error(`missing weight ${name}`);
  return t;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): { products: number[]; sum: number } {
  const products = Array.from(a, (x, i) => x * b[i]);
  return { products, sum: products.reduce((s, p) => s + p, 0) };
}

function headDim(trace: Trace, stage: Stage): number {
  return dims2(trace.byId.get(`L${stage.layer}.attn.H0.q`)!.output)[1];
}

export function provenance(trace: Trace, weights: Weights | null, stage: Stage, r: number, c: number): Provenance | null {
  const stored = at(stage.output, r, c);
  const need = () => {
    if (!weights) throw new Error("weights not loaded yet");
    return weights;
  };

  switch (stage.kind) {
    case "embed": {
      const w = need();
      const isTok = stage.weights[0] === "wte";
      const tableRow = isTok ? trace.tokens[r] : r;
      const value = at(weight(w, stage.weights[0]), tableRow, c);
      return {
        title: isTok ? `Row ${tableRow} of the token table (token "${trace.text[r]}"), column ${c}` : `Row ${r} of the position table, column ${c}`,
        strips: [{ label: `${stage.weights[0]}[${tableRow}]`, values: Array.from(row(weight(w, stage.weights[0]), tableRow)) }],
        lines: [["looked-up value", value]],
        value, stored,
      };
    }
    case "linear":
    case "logits": {
      const w = need();
      const x = row(input(trace, stage), r);
      const name = stage.weights[0];
      const W = weight(w, name);
      // Tied unembedding uses the token table transposed: column c is row c of wte.
      const wcol = stage.kind === "logits" && name === "wte" ? Array.from(row(W, c)) : col(W, c);
      const { products, sum } = dot(x, wcol);
      const bias = stage.weights[1] ? at(weight(w, stage.weights[1]), 0, c) : 0;
      // With an adapter on this stage, its low-rank term (its own stage) is added too.
      const lora = trace.byId.get(`${stage.id}.lora`);
      const delta = lora ? at(lora.output, r, c) : 0;
      return {
        title: `Dot product of input row ${r} with column ${c} of ${name}${stage.weights[1] ? ", plus bias" : ""}${lora ? ", plus the adapter's term" : ""}`,
        strips: [
          { label: `${stage.inputs[0]} row ${r}`, values: Array.from(x) },
          { label: `${name} column ${c}`, values: wcol },
          { label: "element-wise products", values: products },
        ],
        lines: [["sum of products", sum], ...(stage.weights[1] ? [[`bias ${stage.weights[1]}[${c}]`, bias] as [string, number]] : []),
                ...(lora ? [[`adapter ${lora.id}[${r}, ${c}]`, delta] as [string, number]] : []), ["result", sum + bias + delta]],
        value: sum + bias + delta, stored,
      };
    }
    case "lora": {
      // (x A) B · s: x squeezed to r numbers by A, then expanded by column c of B.
      const w = need();
      const x = row(input(trace, stage), r);
      const A = weight(w, stage.weights[0]), B = weight(w, stage.weights[1]);
      const rank = A.shape[1], s = trace.adapter?.scale ?? 1;
      const h = Array.from({ length: rank }, (_, k) => dot(x, col(A, k)).sum);
      const { products, sum } = dot(h, col(B, c));
      return {
        title: `Input row ${r} through A (to ${rank} numbers), then column ${c} of B, times the strength ${s}`,
        strips: [
          { label: `${stage.inputs[0]} row ${r}`, values: Array.from(x) },
          { label: `x · A: row ${r} squeezed to ${rank} numbers`, values: h },
          { label: `${stage.weights[1]} column ${c}`, values: col(B, c) },
          { label: "element-wise products", values: products },
        ],
        lines: [["sum of products", sum], ["strength s", s], ["result", sum * s]],
        value: sum * s, stored,
      };
    }
    case "norm": {
      const w = need();
      const x = Array.from(row(input(trace, stage), r));
      const mean = x.reduce((a, b) => a + b, 0) / x.length;
      const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length;
      const std = Math.sqrt(variance + EPS);
      const xhat = (x[c] - mean) / std;
      const g = at(weight(w, stage.weights[0]), 0, c);
      const b = at(weight(w, stage.weights[1]), 0, c);
      return {
        title: `Normalise row ${r}, then scale and shift column ${c}`,
        strips: [{ label: `${stage.inputs[0]} row ${r}`, values: x }],
        lines: [["x", x[c]], ["row mean μ", mean], ["row std σ", std], ["(x − μ) / σ", xhat], ["γ", g], ["β", b], ["γ·x̂ + β", g * xhat + b]],
        value: g * xhat + b, stored,
      };
    }
    case "softmax":
    case "probs": {
      const x = Array.from(row(input(trace, stage), r));
      const max = Math.max(...x.filter(Number.isFinite));
      const exps = x.map((v) => Math.exp(v - max));
      const total = exps.reduce((a, b) => a + b, 0);
      return {
        title: `Softmax of row ${r}: exp of each score over the row's total`,
        strips: [{ label: `${stage.inputs[0]} row ${r}`, values: x.map((v) => (Number.isFinite(v) ? v : 0)) },
                 { label: "exp(score − max)", values: exps }],
        lines: [["score", x[c]], ["row max", max], ["exp(score − max)", exps[c]], ["sum of exps", total], ["exp / sum", exps[c] / total]],
        value: exps[c] / total, stored,
      };
    }
    case "add": {
      const a = at(input(trace, stage, 0), r, c);
      const b = at(input(trace, stage, 1), r, c);
      return {
        title: `Element-wise sum at row ${r}, column ${c}`,
        strips: [],
        lines: [[stage.inputs[0], a], [stage.inputs[1], b], ["sum", a + b]],
        value: a + b, stored,
      };
    }
    case "split": {
      const dh = dims2(stage.output)[1];
      const src = stage.head! * dh + c;
      const value = at(input(trace, stage), r, src);
      return { title: `Column ${src} of ${stage.inputs[0]} (head ${stage.head}'s band starts at ${stage.head! * dh})`,
               strips: [], lines: [["copied value", value]], value, stored };
    }
    case "attn_scores": {
      const q = row(input(trace, stage, 0), r);
      const k = row(input(trace, stage, 1), c);
      const { products, sum } = dot(q, k);
      const scale = Math.sqrt(q.length);
      return {
        title: `Query ${r} · key ${c}, divided by √${q.length}`,
        strips: [{ label: `query ${r}`, values: Array.from(q) }, { label: `key ${c}`, values: Array.from(k) },
                 { label: "products", values: products }],
        lines: [["q · k", sum], [`÷ √${q.length}`, scale], ["score", sum / scale]],
        value: sum / scale, stored,
      };
    }
    case "mask": {
      const s = at(input(trace, stage), r, c);
      const value = c > r ? -Infinity : s;
      return { title: c > r ? `Key ${c} is after query ${r}: masked` : `Key ${c} ≤ query ${r}: kept`,
               strips: [], lines: [["score", s], ["after mask", value]], value, stored };
    }
    case "attn_mix": {
      const wts = row(input(trace, stage, 0), r);
      const v = col(input(trace, stage, 1), c);
      const { products, sum } = dot(wts, v);
      return {
        title: `Weighted average of value column ${c}, using row ${r}'s attention weights`,
        strips: [{ label: `attention weights row ${r}`, values: Array.from(wts) }, { label: `values, column ${c}`, values: v },
                 { label: "weight × value", values: products }],
        lines: [["sum", sum]],
        value: sum, stored,
      };
    }
    case "concat": {
      const dh = headDim(trace, stage);
      const h = Math.floor(c / dh);
      const value = at(input(trace, stage, h), r, c % dh);
      return { title: `Column ${c % dh} of head ${h}'s output`, strips: [], lines: [["copied value", value]], value, stored };
    }
    case "activation": {
      const u = at(input(trace, stage), r, c);
      return { title: "GELU of one pre-activation", strips: [],
               lines: [["u", u], ["GELU(u) = ½u(1 + tanh(√(2/π)(u + 0.044715u³)))", gelu(u)]], value: gelu(u), stored };
    }
    case "loss": {
      const probs = trace.byId.get("probs")!.output;
      const terms = trace.targets.flatMap((t, i) => (t >= 0 ? [-Math.log(at(probs, i, t))] : []));
      const value = terms.reduce((a, b) => a + b, 0) / terms.length;
      return { title: "Mean over positions of −log p(actual next token)",
               strips: [{ label: "−log p per position", values: terms }],
               lines: [["positions with a target", terms.length], ["mean", value]], value, stored };
    }
  }
  return null;
}

/** Backward phase: where a gradient came from. Downstream stages that read this one each
 * contribute; residual adds pass their gradient straight through. */
export function gradProvenance(trace: Trace, stage: Stage, r: number, c: number): Provenance | null {
  if (!stage.grad) return null;
  const stored = at(stage.grad, r, c);
  if (stage.kind === "loss") return { title: "∂L/∂L = 1: the backward pass starts here", strips: [], lines: [], value: 1, stored };
  if (stage.kind === "logits") {
    const p = at(trace.byId.get("probs")!.output, r, c);
    const hasTarget = trace.targets[r] >= 0;
    const y = trace.targets[r] === c ? 1 : 0;
    const n = trace.targets.filter((t) => t >= 0).length;
    const value = hasTarget ? (p - y) / n : 0;
    return {
      title: hasTarget ? "Softmax + cross-entropy: (p − y), divided by the number of positions averaged" : "No target at this position, so no gradient",
      strips: [], lines: [["p", p], ["y (1 if this is the actual next token)", y], ["positions", n], ["(p − y) / n", value]],
      value, stored,
    };
  }
  const consumers = trace.stages.filter((s) => s.inputs.includes(stage.id) && s.grad);
  const lines: [string, number | string][] = [];
  let known = 0;
  for (const s of consumers) {
    if (s.kind === "add") {
      const g = at(s.grad!, r, c);
      lines.push([`from ${s.id} (add: passes straight through)`, g]);
      known += g;
    } else if (s.kind === "split") {
      const dh = dims2(s.output)[1];
      const h = s.head!;
      if (c >= h * dh && c < (h + 1) * dh) {
        const g = at(s.grad!, r, c - h * dh);
        lines.push([`from ${s.id} (head ${h}'s band)`, g]);
        known += g;
      }
    } else {
      lines.push([`from ${s.id} (${s.kind}: chain rule through its formula)`, "…"]);
    }
  }
  if (lines.some(([, v]) => v === "…")) lines.push(["those, together", stored - known]);
  lines.push(["total ∂L/∂x", stored]);
  return {
    title: consumers.length > 1 ? `Sum of gradients from the ${consumers.length} stages that read this one` : "Gradient from the stage that reads this one",
    strips: [], lines, value: stored, stored,
  };
}
