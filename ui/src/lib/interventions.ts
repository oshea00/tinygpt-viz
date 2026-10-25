// Describing interventions and measuring their effect against the baseline trace.

import type { Intervention, Tensor, Trace } from "../types";
import { at, dims2, norm } from "./tensor";

export function describe(iv: Intervention): string {
  switch (iv.kind) {
    case "head": return `L${iv.layer} head ${iv.head}: ${iv.mode === "zero" ? "zeroed" : "replaced by its mean"}`;
    case "neuron": return `L${iv.layer} MLP neuron ${iv.index}: ${iv.mode === "zero" ? "zeroed" : "replaced by its mean"}`;
    case "block": return `L${iv.layer} ${iv.part === "attn" ? "attention" : "MLP"} switched off`;
    case "attend": return `L${iv.layer} head ${iv.head}: forced to attend to position ${iv.position}`;
    case "set": return `${iv.stage}[${iv.row}, ${iv.col}] set to ${iv.value}`;
    case "shuffle_pos": return `position embeddings shuffled (seed ${iv.seed})`;
    case "steer": return `steering at ${iv.stage}${iv.label ? ` towards ${iv.label}` : ""} × ${iv.scale}`;
  }
}

/** after − before, or null if the shapes differ. Masked cells (−∞ in both) become 0. */
export function difference(after: Tensor, before: Tensor | undefined): Tensor | null {
  if (!before || after.data.length !== before.data.length) return null;
  const data = new Float32Array(after.data.length);
  for (let i = 0; i < data.length; i++) {
    const a = after.data[i], b = before.data[i];
    data[i] = a === b ? 0 : a - b;   // a === b also covers −∞ − −∞
  }
  return { shape: after.shape, data };
}

export interface StageChange {
  id: string;
  change: number;     // ‖after − before‖
  relative: number;   // ... divided by ‖before‖
}

/** How much every stage changed, in execution order: where the change enters and how it spreads. */
export function stageChanges(trace: Trace, baseline: Trace): StageChange[] {
  return trace.stages.flatMap((s) => {
    const d = difference(s.output, baseline.byId.get(s.id)?.output);
    if (!d) return [];
    const finite = (t: Tensor) => t.data.map((v) => (Number.isFinite(v) ? v : 0));
    const change = norm(finite(d));
    const base = norm(finite(baseline.byId.get(s.id)!.output));
    return [{ id: s.id, change, relative: base > 0 ? change / base : change > 0 ? Infinity : 0 }];
  });
}

/** p(actual next token) per position, before and after. */
export function targetProbs(trace: Trace, baseline: Trace): { before: number; after: number }[] {
  const pa = trace.byId.get("probs")!.output, pb = baseline.byId.get("probs")!.output;
  if (dims2(pa)[0] !== dims2(pb)[0]) return [];
  return trace.targets.flatMap((t, i) => (t >= 0 ? [{ before: at(pb, i, t), after: at(pa, i, t) }] : []));
}

/** Whether a baseline matches a trace well enough to compare (same prompt and weights). */
export function comparable(trace: Trace | null, baseline: Trace | null): baseline is Trace {
  return !!trace && !!baseline && baseline.prompt === trace.prompt && baseline.step === trace.step && baseline.tokens.length === trace.tokens.length;
}
