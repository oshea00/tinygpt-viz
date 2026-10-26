// Maps of the model's learned tables and of the residual stream: 2D projections of the position
// and token embeddings, each position's path through the snapshots, and which residual
// dimensions carry the most weight. Projections are PCA (lib/concepts.ts), so a map's axes are
// real directions in the model's space.

import type { Tensor, Trace } from "../types";
import { pca, project } from "./concepts";
import { snapshots } from "./residual";
import { dims2, row } from "./tensor";

/** Stages whose columns are the residual stream's own dimensions (d_model wide, same basis):
 * the embeddings, every residual snapshot, the layer norms (which scale each dimension
 * separately) and the two sublayer writes. Q, K, V, the heads, the concatenated heads (one band
 * per head), the MLP's hidden layer and the logits use other coordinates. */
export function inResidualBasis(stageId: string): boolean {
  return /^(embed\.(tok|pos|sum)|ln_f|L\d+\.(ln1|ln2|attn\.out|resid_attn|mlp\.down|resid_mlp))$/.test(stageId);
}

export interface Map2D {
  points: [number, number][];
  explained: [number, number];   // fraction of the variance along each axis
}

function centred(rows: Float64Array[]): Float64Array[] {
  const d = rows[0]?.length ?? 0;
  const mean = new Float64Array(d);
  for (const r of rows) for (let k = 0; k < d; k++) mean[k] += r[k] / rows.length;
  return rows.map((r) => r.map((x, k) => x - mean[k]));
}

function map2d(rows: Float64Array[]): Map2D {
  const { components, variance, total } = pca(rows, 2);
  return {
    points: project(rows, components).map(([x, y]) => [x, y]),
    explained: [variance[0] / (total || 1), variance[1] / (total || 1)],
  };
}

/** Points scaled into a w × h box with one scale for both axes, so distances and shapes are true. */
export function fitBox(points: [number, number][], w: number, h: number, pad: number): [number, number][] {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const k = Math.min((w - 2 * pad) / (x1 - x0 || 1), (h - 2 * pad) / (y1 - y0 || 1));
  const ox = (w - (x1 - x0) * k) / 2, oy = (h - (y1 - y0) * k) / 2;
  return points.map(([x, y]) => [ox + (x - x0) * k, oy + (y - y0) * k]);
}

/** A table's rows (one per token, or per position), centred and projected onto its top two
 * principal components. */
export function tableMap(t: Tensor): Map2D {
  const [n] = dims2(t);
  return map2d(centred(Array.from({ length: n }, (_, i) => Float64Array.from(row(t, i)))));
}

/** Every position's vector at every snapshot, projected into one shared 2D view:
 * points[t][s] is position t at snapshot s. Normalising centres each snapshot across positions
 * and scales each vector to unit length, so the later layers' larger vectors do not dominate. */
export function trajectories(trace: Trace, normalise = true): Map2D & { paths: [number, number][][]; labels: string[] } {
  const snaps = snapshots(trace);
  const T = trace.tokens.length;
  const perSnap = snaps.map((s) => {
    const X = trace.byId.get(s.id)!.output;
    const rows = Array.from({ length: T }, (_, t) => Float64Array.from(row(X, t)));
    if (!normalise) return rows;
    return centred(rows).map((r) => {
      const n = Math.hypot(...r) || 1;
      return r.map((x) => x / n);
    });
  });
  const all = normalise ? perSnap.flat() : centred(perSnap.flat());
  const m = map2d(all);
  // all is snapshot-major: snapshot s, position t is row s * T + t.
  const paths = Array.from({ length: T }, (_, t) => snaps.map((_, s) => m.points[s * T + t]));
  return { ...m, paths, labels: snaps.map((s) => s.label) };
}

/** Mean |x| of each residual dimension over the positions at one stage: a few dimensions are
 * typically far larger than the rest. */
export function dimSizes(trace: Trace, stageId: string): Float64Array {
  const X = trace.byId.get(stageId)!.output;
  const [T, d] = dims2(X);
  const out = new Float64Array(d);
  for (let t = 0; t < T; t++) {
    const r = row(X, t);
    for (let k = 0; k < d; k++) out[k] += Math.abs(r[k]) / T;
  }
  return out;
}

export type TokenKind = "word" | "Capitalised" | "word piece" | "digits" | "punctuation" | "byte" | "end of story";
export const TOKEN_KINDS: TokenKind[] = ["word", "Capitalised", "word piece", "digits", "punctuation", "byte", "end of story"];

/** What a token looks like, from its display text (␣ for a space, ⏎ for a newline, ␄ for the
 * end-of-story marker). */
export function tokenKind(display: string): TokenKind {
  if (display === "␄") return "end of story";
  const s = display.replace(/^␣/, "");
  if (/^[a-z]+$/.test(s)) return display.startsWith("␣") ? "word" : "word piece";
  if (/^[A-Z][a-zA-Z]*$/.test(s)) return "Capitalised";
  if (/^\d+$/.test(s)) return "digits";
  if (/^[\x20-\x7e␣⏎]+$/.test(display)) return "punctuation";
  return "byte";
}
