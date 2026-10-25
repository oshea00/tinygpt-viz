// Colour-scale ranges. The range in use is always printed next to a heatmap:
// silent rescaling is the commonest way such tools mislead.

import type { Stage, Tensor, Trace } from "../types";
import type { MapKind } from "./colormap";

export type ScaleMode = "tensor" | "stageType" | "global";

export interface Range {
  kind: MapKind;
  lo: number;
  hi: number;
  symlog: boolean;
  label: string;
}

const SAMPLE = 20_000;

/** p-th percentile of |x| (diverging) or x (sequential) over finite values, sampled.
 * Runs for every thumbnail on every trace, so it uses a native typed-array sort. */
function percentile(arrays: Float32Array[], p: number, abs: boolean, sample = SAMPLE): number {
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const stride = Math.max(1, Math.ceil(total / sample));
  const vals = new Float32Array(Math.ceil(total / stride) + arrays.length);
  let n = 0;
  let max = 0;
  let i = 0;
  for (const a of arrays) {
    for (let j = 0; j < a.length; j++, i++) {
      const v = abs ? Math.abs(a[j]) : a[j];
      if (!Number.isFinite(v)) continue;
      if (v > max) max = v;
      if (i % stride === 0) vals[n++] = v;
    }
  }
  if (p >= 100 || n === 0) return max || 1;
  const sorted = vals.subarray(0, n).sort();
  return sorted[Math.min(n - 1, Math.floor((p / 100) * n))] || max || 1;
}

export function isSequential(stage: Stage, grad: boolean): boolean {
  return !grad && (stage.kind === "softmax" || stage.kind === "probs");
}

/** Stages of one type across layers and heads share a key: L2.attn.H1.weights → L*.attn.H*.weights */
export const stageTypeKey = (id: string) => id.replace(/^L\d+/, "L*").replace(/\.H\d+\./, ".H*.");

export function tensorRange(arrays: Float32Array[], kind: MapKind, clip: number, symlog: boolean, label: string,
                            sample = SAMPLE): Range {
  const hi = percentile(arrays, clip, kind === "diverging", sample);
  return { kind, lo: kind === "diverging" ? -hi : 0, hi, symlog, label };
}

export function stageRange(trace: Trace, stage: Stage, grad: boolean, mode: ScaleMode, clip: number,
                           symlog: boolean): Range {
  const kind: MapKind = isSequential(stage, grad) ? "sequential" : "diverging";
  const pick = (s: Stage) => (grad ? s.grad : s.output)?.data;
  let group: Stage[];
  if (mode === "tensor") group = [stage];
  else if (mode === "stageType") {
    const key = stageTypeKey(stage.id);
    group = trace.stages.filter((s) => stageTypeKey(s.id) === key);
  } else group = trace.stages.filter((s) => isSequential(s, grad) === (kind === "sequential") && s.kind !== "loss");
  const arrays = group.map(pick).filter((a): a is Float32Array => !!a);
  const label = mode === "tensor" ? "this tensor" : mode === "stageType" ? `all ${stageTypeKey(stage.id)}` : "whole trace";
  return tensorRange(arrays, kind, clip, symlog, label);
}

export function weightRange(t: Tensor, clip: number): Range {
  return tensorRange([t.data], "diverging", clip, false, "this tensor");
}

/** Map a value into [-1, 1] (diverging) or [0, 1] (sequential); |t| > 1 means clipped. */
export function normalise(v: number, r: Range): number {
  if (r.kind === "sequential") return r.hi > 0 ? v / r.hi : 0;
  if (!r.symlog) return r.hi > 0 ? v / r.hi : 0;
  const s = r.hi / 50;
  return (Math.sign(v) * Math.log1p(Math.abs(v) / s)) / Math.log1p(r.hi / s);
}
