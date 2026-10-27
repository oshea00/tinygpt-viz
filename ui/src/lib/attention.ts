// Attention summaries computed from a trace: per-head pattern statistics for the gallery,
// and the query-against-every-key working for the Q·K explorer.

import type { Tensor, Trace } from "../types";
import { at, dims2, row } from "./tensor";

export interface HeadStats {
  prev: number;      // mean weight on the previous position
  self: number;      // mean weight on the position itself
  first: number;     // mean weight on position 0
  entropy: number;   // mean entropy of a row, in nats
  distance: number;  // mean look-back distance, weighted by attention
  peak: number;      // mean of each row's largest weight: how focused the head is
  induction: number; // weight on the token after an earlier copy of this one (rows with a copy only; 0 if none)
}

/** Statistics over query rows 1..T-1 (row 0 can only attend to itself, so it says nothing).
 * With the prompt's token ids, also the induction score: for each query whose token appeared
 * earlier at j, the weight on j + 1 (what followed it last time), averaged over those queries. */
export function headStats(weights: Tensor, tokens?: number[]): HeadStats {
  const T = dims2(weights)[0];
  const s: HeadStats = { prev: 0, self: 0, first: 0, entropy: 0, distance: 0, peak: 0, induction: 0 };
  if (T < 2) return { ...s, self: 1, peak: 1 };
  let repeats = 0;
  for (let r = 1; r < T; r++) {
    const w = row(weights, r);
    if (tokens) {
      let copied = 0, seen = false;
      for (let j = 0; j < r; j++) if (tokens[j] === tokens[r]) { copied += w[j + 1]; seen = true; }
      if (seen) { s.induction += copied; repeats++; }
    }
    s.prev += w[r - 1];
    s.self += w[r];
    s.first += w[0];
    let peak = 0;
    for (let j = 0; j <= r; j++) {
      peak = Math.max(peak, w[j]);
      if (w[j] > 0) s.entropy -= w[j] * Math.log(w[j]);
      s.distance += w[j] * (r - j);
    }
    s.peak += peak;
  }
  const n = T - 1;
  return { prev: s.prev / n, self: s.self / n, first: s.first / n, entropy: s.entropy / n, distance: s.distance / n, peak: s.peak / n,
           induction: repeats ? s.induction / repeats : 0 };
}

/** Whether any token appears twice, so the induction score means something. */
export const hasRepeats = (tokens: number[]) => new Set(tokens).size < tokens.length;

/** A short description of the head's pattern: a fixed target if it has one, otherwise
 * how focused it is and how far back it typically looks. */
export function headPattern(s: HeadStats): string {
  const [name, v] = ([["previous", s.prev], ["self", s.self], ["first", s.first]] as const).reduce((a, b) => (b[1] > a[1] ? b : a));
  if (v >= 0.4) return `${name} ${v.toFixed(2)}`;
  const back = `${s.distance < 10 ? s.distance.toFixed(1) : s.distance.toFixed(0)} back`;
  if (s.peak >= 0.5) return `sharp · ${back}`;
  if (s.distance <= 5) return `local · ${back}`;
  return `broad · ${back}`;
}

export interface QKRow {
  key: number;
  k: Float32Array;
  products: number[];
  dot: number;
  score: number;     // dot / √d_head, recomputed
  stored: number;    // the score as stored in the trace
  weight: number;
}

export function qkRows(trace: Trace, layer: number, head: number, query: number): { q: Float32Array; rows: QKRow[]; scale: number } {
  const p = `L${layer}.attn.H${head}`;
  const Q = trace.byId.get(`${p}.q`)!.output;
  const K = trace.byId.get(`${p}.k`)!.output;
  const S = trace.byId.get(`${p}.scores`)!.output;
  const W = trace.byId.get(`${p}.weights`)!.output;
  const q = row(Q, query);
  const scale = Math.sqrt(q.length);
  const rows: QKRow[] = [];
  for (let j = 0; j <= query; j++) {
    const k = row(K, j);
    const products = Array.from(q, (x, i) => x * k[i]);
    const dot = products.reduce((a, b) => a + b, 0);
    rows.push({ key: j, k, products, dot, score: dot / scale, stored: at(S, query, j), weight: at(W, query, j) });
  }
  return { q, rows, scale };
}

/** Parse an attention stage id: L2.attn.H1.weights → {layer 2, head 1}. */
export function headOf(stageId: string): { layer: number; head: number } | null {
  const m = /^L(\d+)\.attn\.H(\d+)\./.exec(stageId);
  return m ? { layer: +m[1], head: +m[2] } : null;
}
