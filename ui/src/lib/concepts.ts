// Concept-explorer analyses, run in the browser on vectors captured by the server
// (core/concepts.py): similarity, projections, probes, directions and drift.
//
// Raw residual vectors share a large common component, which makes everything look similar,
// so every analysis here centres each snapshot (subtracts the mean over captured items) first.

export interface CaptureItem {
  word: string;
  category: string;
  prompt: string;
  tokens: string[];
  position: number;
}

export interface Capture {
  vectors: Float32Array;   // [N, S, d]
  N: number;
  S: number;
  d: number;
  items: CaptureItem[];
  snapshots: string[];
  model: "current" | "untrained";
  step: number;
}

/** Item vectors at one snapshot, centred: rows are items. */
export function layer(c: Capture, s: number, centre = true): Float64Array[] {
  const rows = Array.from({ length: c.N }, (_, i) => Float64Array.from(c.vectors.subarray((i * c.S + s) * c.d, (i * c.S + s + 1) * c.d)));
  if (!centre || !rows.length) return rows;
  const mean = new Float64Array(c.d);
  for (const r of rows) for (let k = 0; k < c.d; k++) mean[k] += r[k] / rows.length;
  for (const r of rows) for (let k = 0; k < c.d; k++) r[k] -= mean[k];
  return rows;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
const len = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));
export const cosine = (a: ArrayLike<number>, b: ArrayLike<number>) => dot(a, b) / (len(a) * len(b) || 1);

/** One vector per distinct word (the average of its occurrences), ordered by category. */
export function wordMeans(c: Capture, s: number): { word: string; category: string; v: Float64Array }[] {
  const rows = layer(c, s);
  const groups = new Map<string, { category: string; sum: Float64Array; n: number }>();
  c.items.forEach((it, i) => {
    const g = groups.get(it.word) ?? { category: it.category, sum: new Float64Array(c.d), n: 0 };
    for (let k = 0; k < c.d; k++) g.sum[k] += rows[i][k];
    g.n++;
    groups.set(it.word, g);
  });
  return [...groups].map(([word, g]) => ({ word, category: g.category, v: g.sum.map((x) => x / g.n) }))
    .sort((a, b) => a.category.localeCompare(b.category) || a.word.localeCompare(b.word));
}

export function similarityMatrix(vs: ArrayLike<number>[]): Float32Array {
  const n = vs.length;
  const m = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m[i * n + j] = cosine(vs[i], vs[j]);
  return m;
}

/** Top principal directions by power iteration with deflation (rows already centred). */
export function pca(rows: Float64Array[], k = 2, iters = 200): { components: Float64Array[]; variance: number[]; total: number } {
  const d = rows[0]?.length ?? 0;
  const cov = Array.from({ length: d }, () => new Float64Array(d));
  // Symmetric: fill the upper triangle, then mirror it (halves the work for large tables).
  for (const r of rows) for (let i = 0; i < d; i++) {
    const ri = r[i] / rows.length, ci = cov[i];
    for (let j = i; j < d; j++) ci[j] += ri * r[j];
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) cov[i][j] = cov[j][i];
  const total = cov.reduce((s, r, i) => s + r[i], 0);
  const components: Float64Array[] = [], variance: number[] = [];
  for (let c = 0; c < k; c++) {
    let v = Float64Array.from({ length: d }, (_, i) => Math.sin(i * 12.9898 + c * 78.233));   // fixed start: same picture every time
    let lambda = 0;
    for (let t = 0; t < iters; t++) {
      const w = cov.map((r) => dot(r, v));
      lambda = len(w);
      if (lambda === 0) break;
      v = Float64Array.from(w, (x) => x / lambda);
    }
    components.push(v);
    variance.push(lambda);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] -= lambda * v[i] * v[j];
  }
  return { components, variance, total };
}

export function project(rows: ArrayLike<number>[], dirs: ArrayLike<number>[]): number[][] {
  return rows.map((r) => dirs.map((d) => dot(r, d)));
}

/** Multinomial logistic regression, trained on some rows and scored on the rest. */
export function probe(rows: Float64Array[], labels: number[], train: boolean[], classes: number,
                      iters = 300, lr = 0.5, l2 = 1e-3): { trainAcc: number; testAcc: number } {
  const d = rows[0].length;
  // Standardise features using training rows only.
  const mu = new Float64Array(d), sd = new Float64Array(d);
  const tr = rows.filter((_, i) => train[i]);
  for (const r of tr) for (let k = 0; k < d; k++) mu[k] += r[k] / tr.length;
  for (const r of tr) for (let k = 0; k < d; k++) sd[k] += (r[k] - mu[k]) ** 2 / tr.length;
  const X = rows.map((r) => Float64Array.from(r, (x, k) => (x - mu[k]) / (Math.sqrt(sd[k]) || 1)));
  const W = Array.from({ length: classes }, () => new Float64Array(d + 1));   // last entry: bias
  const scores = (x: Float64Array) => W.map((w) => dot(x, w) + w[d]);   // dot runs over x's d entries; w[d] is the bias
  const softmax = (z: number[]) => {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  for (let t = 0; t < iters; t++) {
    const G = Array.from({ length: classes }, () => new Float64Array(d + 1));
    X.forEach((x, i) => {
      if (!train[i]) return;
      const p = softmax(scores(x));
      for (let c = 0; c < classes; c++) {
        const g = p[c] - (labels[i] === c ? 1 : 0);
        for (let k = 0; k < d; k++) G[c][k] += g * x[k];
        G[c][d] += g;
      }
    });
    for (let c = 0; c < classes; c++) for (let k = 0; k <= d; k++) W[c][k] -= lr * (G[c][k] / tr.length + (k < d ? l2 * W[c][k] : 0));
  }
  const acc = (want: boolean) => {
    const idx = X.map((_, i) => i).filter((i) => train[i] === want);
    if (!idx.length) return NaN;
    return idx.filter((i) => { const s = scores(X[i]); return s.indexOf(Math.max(...s)) === labels[i]; }).length / idx.length;
  };
  return { trainAcc: acc(true), testAcc: acc(false) };
}

/** Words held out of probe training: every `every`-th distinct word of each category. */
export function heldOutWords(items: CaptureItem[], every = 4): Set<string> {
  const byCat = new Map<string, string[]>();
  for (const it of items) {
    const ws = byCat.get(it.category) ?? [];
    if (!ws.includes(it.word)) ws.push(it.word);
    byCat.set(it.category, ws);
  }
  const out = new Set<string>();
  for (const ws of byCat.values()) ws.sort().forEach((w, i) => { if (i % every === every - 1) out.add(w); });
  return out;
}

/** Probe accuracy on held-out words at every snapshot; `shuffle` permutes labels as a baseline. */
export function probeByLayer(c: Capture, shuffle = false): { layer: string; testAcc: number; trainAcc: number }[] {
  const cats = [...new Set(c.items.map((i) => i.category))].sort();
  let labels = c.items.map((i) => cats.indexOf(i.category));
  if (shuffle) {
    // Deterministic shuffle (by word, so a word keeps one label): structure here is chance.
    const words = [...new Set(c.items.map((i) => i.word))].sort();
    const fake = new Map(words.map((w, i) => [w, (i * 7 + 3) % cats.length]));
    labels = c.items.map((i) => fake.get(i.word)!);
  }
  const held = heldOutWords(c.items);
  const train = c.items.map((i) => !held.has(i.word));
  return c.snapshots.map((name, s) => ({ layer: name, ...probe(layer(c, s), labels, train, cats.length) }));
}

/** Mean of group A minus mean of group B: a candidate direction for a concept. */
export function direction(rows: ArrayLike<number>[], a: number[], b: number[]): Float64Array {
  const d = rows[0].length;
  const out = new Float64Array(d);
  for (const i of a) for (let k = 0; k < d; k++) out[k] += rows[i][k] / a.length;
  for (const i of b) for (let k = 0; k < d; k++) out[k] -= rows[i][k] / b.length;
  return out;
}

/** How much one word's representation varies across its contexts: mean pairwise cosine per snapshot. */
export function drift(c: Capture, word: string): { layer: string; similarity: number; n: number }[] {
  const idx = c.items.map((it, i) => (it.word === word ? i : -1)).filter((i) => i >= 0);
  return c.snapshots.map((name, s) => {
    const rows = layer(c, s);
    let sum = 0, n = 0;
    for (let a = 0; a < idx.length; a++) for (let b = a + 1; b < idx.length; b++) {
      sum += cosine(rows[idx[a]], rows[idx[b]]);
      n++;
    }
    return { layer: name, similarity: n ? sum / n : NaN, n: idx.length };
  });
}

/** Nearest rows of a table (e.g. the token embeddings) to row `i` by cosine. */
export function nearestRows(table: Float32Array, rows: number, d: number, i: number, k = 12): { i: number; cos: number }[] {
  const q = table.subarray(i * d, (i + 1) * d);
  const out: { i: number; cos: number }[] = [];
  for (let r = 0; r < rows; r++) if (r !== i) out.push({ i: r, cos: cosine(q, table.subarray(r * d, (r + 1) * d)) });
  return out.sort((a, b) => b.cos - a.cos).slice(0, k);
}

/** Attention rollout: average heads, add the residual (identity), renormalise, multiply through
 * layers. Returns how much each position flows into position `row` (default: the last). */
export function attentionRollout(weights: Float32Array[][], T: number, row = T - 1): Float64Array {
  let R = Array.from({ length: T }, (_, i) => Float64Array.from({ length: T }, (_, j) => (i === j ? 1 : 0)));
  for (const heads of weights) {
    const A = Array.from({ length: T }, (_, i) => Float64Array.from({ length: T }, (_, j) => {
      const mean = heads.reduce((s, h) => s + h[i * T + j], 0) / heads.length;
      return 0.5 * mean + (i === j ? 0.5 : 0);
    }));
    R = A.map((row) => Float64Array.from({ length: T }, (_, j) => row.reduce((s, a, k) => s + a * R[k][j], 0)));
  }
  return R[row];
}
