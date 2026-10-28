// Belief trajectories, run in the browser on what core/beliefs.py captures: after each sentence of
// each story, the model's belief (the cloze readout over a word set) and the residual vectors at
// the prefix's last token. After Bigelow et al., "Stories in Space" (arXiv 2605.12412): ridge
// probes from activations to belief, concept centroids and their geometry in both spaces, and
// directions for steering.

export interface BeliefPrefix { story: number; sentence: number; start: number; end: number; }

export interface Beliefs {
  vectors: Float32Array;   // [N, S, d]
  readout: Float32Array;   // [N, K], each row sums to 1
  mass: Float32Array;      // [N]: the words' raw total probability
  N: number;
  S: number;
  d: number;
  K: number;
  stories: string[];
  prefixes: BeliefPrefix[];
  snapshots: string[];
  words: string[];
  cloze: string;
  model: "current" | "untrained";
  step: number;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/** Prefix vectors at one snapshot (not centred): rows are prefixes. */
export function snapshotRows(b: Beliefs, s: number): Float64Array[] {
  return Array.from({ length: b.N }, (_, i) => Float64Array.from(b.vectors.subarray((i * b.S + s) * b.d, (i * b.S + s + 1) * b.d)));
}

export function readoutRows(b: Beliefs): Float64Array[] {
  return Array.from({ length: b.N }, (_, i) => Float64Array.from(b.readout.subarray(i * b.K, (i + 1) * b.K)));
}

/** Rows minus their mean. */
export function centre(rows: ArrayLike<number>[]): Float64Array[] {
  const d = rows[0]?.length ?? 0;
  const mean = new Float64Array(d);
  for (const r of rows) for (let k = 0; k < d; k++) mean[k] += r[k] / rows.length;
  return rows.map((r) => Float64Array.from({ length: d }, (_, k) => r[k] - mean[k]));
}

/** Every fifth story is held out, so no story is in both the training and the test set. */
export const isTestStory = (story: number) => story % 5 === 4;
export const trainMask = (b: Beliefs) => b.prefixes.map((p) => !isTestStory(p.story));

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i] / n; mb += b[i] / n; }
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb || 1);
}

/** Solve A x = B for symmetric positive definite A (d × d, row-major) and several right-hand sides. */
function choleskySolve(A: Float64Array, d: number, B: Float64Array[]): Float64Array[] {
  const L = new Float64Array(d * d);
  for (let i = 0; i < d; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * d + j];
    for (let k = 0; k < j; k++) s -= L[i * d + k] * L[j * d + k];
    L[i * d + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * d + j];
  }
  return B.map((b) => {
    const y = new Float64Array(d), x = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i * d + k] * y[k];
      y[i] = s / L[i * d + i];
    }
    for (let i = d - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < d; k++) s -= L[k * d + i] * x[k];
      x[i] = s / L[i * d + i];
    }
    return x;
  });
}

export interface RidgeFit {
  pred: Float64Array[];      // [N][K], for every row (train and test)
  rmse: number;              // on test rows, over all outputs
  r: number[];               // per output, on test rows
  dirs: Float64Array[];      // [K][d]: each output's weights in residual-stream units (θ / sd)
}

/** Linear regression with an L2 penalty from rows X to targets Y, fitted on the training rows.
 * Features are standardised with training statistics; lambda applies to standardised weights. */
export function ridge(X: ArrayLike<number>[], Y: ArrayLike<number>[], train: boolean[], lambda = 10): RidgeFit {
  const d = X[0].length, K = Y[0].length;
  const tr = X.map((_, i) => i).filter((i) => train[i]), te = X.map((_, i) => i).filter((i) => !train[i]);
  const mu = new Float64Array(d), sd = new Float64Array(d), ym = new Float64Array(K);
  for (const i of tr) {
    for (let k = 0; k < d; k++) mu[k] += X[i][k] / tr.length;
    for (let k = 0; k < K; k++) ym[k] += Y[i][k] / tr.length;
  }
  for (const i of tr) for (let k = 0; k < d; k++) sd[k] += (X[i][k] - mu[k]) ** 2 / tr.length;
  for (let k = 0; k < d; k++) sd[k] = Math.sqrt(sd[k]) || 1;
  const Z = X.map((x) => Float64Array.from({ length: d }, (_, k) => (x[k] - mu[k]) / sd[k]));
  // Normal equations, the upper triangle then mirrored.
  const A = new Float64Array(d * d), B = Array.from({ length: K }, () => new Float64Array(d));
  for (const i of tr) {
    const z = Z[i];
    for (let a = 0; a < d; a++) {
      const za = z[a], off = a * d;
      for (let c = a; c < d; c++) A[off + c] += za * z[c];
      for (let k = 0; k < K; k++) B[k][a] += za * (Y[i][k] - ym[k]);
    }
  }
  for (let a = 0; a < d; a++) {
    A[a * d + a] += lambda;
    for (let c = 0; c < a; c++) A[a * d + c] = A[c * d + a];
  }
  const W = choleskySolve(A, d, B);
  const pred = Z.map((z) => Float64Array.from({ length: K }, (_, k) => ym[k] + dot(z, W[k])));
  let se = 0;
  for (const i of te) for (let k = 0; k < K; k++) se += (pred[i][k] - Y[i][k]) ** 2;
  const r = Array.from({ length: K }, (_, k) => pearson(te.map((i) => pred[i][k]), te.map((i) => Y[i][k])));
  return { pred, rmse: Math.sqrt(se / (te.length * K || 1)), r, dirs: W.map((w) => w.map((x, k) => x / sd[k])) };
}

/** RMSE on test rows of always predicting the training mean: what a probe has to beat. */
export function meanBaseline(Y: ArrayLike<number>[], train: boolean[]): number {
  const K = Y[0].length, ym = new Float64Array(K);
  const tr = Y.filter((_, i) => train[i]), te = Y.filter((_, i) => !train[i]);
  for (const y of tr) for (let k = 0; k < K; k++) ym[k] += y[k] / tr.length;
  let se = 0;
  for (const y of te) for (let k = 0; k < K; k++) se += (y[k] - ym[k]) ** 2;
  return Math.sqrt(se / (te.length * K || 1));
}

export interface ProbeLayer { layer: string; rmse: number; r: number[]; fit: RidgeFit; }

/** A ridge probe at every snapshot of `source`, predicting the readout in `target` (by default
 * the same capture). The untrained baseline probes the untrained model's vectors for the trained
 * model's belief: how much of it is readable without any learning. */
export function probeByLayer(source: Beliefs, target: Beliefs = source, lambda = 10): ProbeLayer[] {
  const Y = readoutRows(target), train = trainMask(target);
  return source.snapshots.map((layer, s) => {
    const fit = ridge(snapshotRows(source, s), Y, train, lambda);
    return { layer, rmse: fit.rmse, r: fit.r, fit };
  });
}

/** Indices of the prefixes that express each concept most, at most `perStory` from any one story
 * (so a single long story cannot dominate a centroid). */
export function topPrefixes(b: Beliefs, k: number, topK: number, perStory = 3): number[] {
  const order = b.prefixes.map((_, i) => i).sort((i, j) => b.readout[j * b.K + k] - b.readout[i * b.K + k]);
  const used = new Map<number, number>(), out: number[] = [];
  for (const i of order) {
    const st = b.prefixes[i].story, n = used.get(st) ?? 0;
    if (n >= perStory) continue;
    used.set(st, n + 1);
    out.push(i);
    if (out.length === topK) break;
  }
  return out;
}

export const defaultTopK = (b: Beliefs) => Math.max(5, Math.round(b.N / 20));

/** One centroid per concept: the mean of `rows` over that concept's top prefixes. */
export function centroids(b: Beliefs, rows: ArrayLike<number>[], topK = defaultTopK(b), perStory = 3): Float64Array[] {
  return b.words.map((_, k) => {
    const idx = topPrefixes(b, k, topK, perStory), d = rows[0].length;
    const c = new Float64Array(d);
    for (const i of idx) for (let j = 0; j < d; j++) c[j] += rows[i][j] / idx.length;
    return c;
  });
}

/** Euclidean distances between every pair of points, K × K row-major. */
export function distanceMatrix(points: ArrayLike<number>[]): Float64Array {
  const K = points.length, m = new Float64Array(K * K);
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    let s = 0;
    for (let k = 0; k < points[i].length; k++) s += (points[i][k] - points[j][k]) ** 2;
    m[i * K + j] = Math.sqrt(s);
  }
  return m;
}

/** Pearson r between the off-diagonal upper triangles of two K × K matrices. */
export function upperTriangleR(A: ArrayLike<number>, B: ArrayLike<number>, K: number): number {
  const a: number[] = [], b: number[] = [];
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { a.push(A[i * K + j]); b.push(B[i * K + j]); }
  return pearson(a, b);
}

/** Behaviour-space and activation-space centroid distances, and how well they agree, at every
 * snapshot. Prefixes are chosen by b's belief; `source` gives the vectors (the untrained baseline
 * passes its own, for the same stories). */
export function geometryByLayer(b: Beliefs, source: Beliefs = b): { behaviour: Float64Array; layers: { layer: string; activation: Float64Array; r: number }[] } {
  const behaviour = distanceMatrix(centroids(b, readoutRows(b)));
  const layers = source.snapshots.map((layer, s) => {
    const activation = distanceMatrix(centroids(b, snapshotRows(source, s)));
    return { layer, activation, r: upperTriangleR(behaviour, activation, b.K) };
  });
  return { behaviour, layers };
}

/** Mean over prefixes whose strongest belief is concept k, minus the mean over the rest. */
export function meanDiffDirection(b: Beliefs, rows: ArrayLike<number>[], k: number): Float64Array {
  const d = rows[0].length, pos = new Float64Array(d), neg = new Float64Array(d);
  let np = 0, nn = 0;
  for (let i = 0; i < b.N; i++) {
    const row = b.readout.subarray(i * b.K, (i + 1) * b.K);
    const top = row.indexOf(Math.max(...row)) === k;
    const acc = top ? pos : neg;
    for (let j = 0; j < d; j++) acc[j] += rows[i][j];
    if (top) np++; else nn++;
  }
  return Float64Array.from({ length: d }, (_, j) => (np ? pos[j] / np : 0) - (nn ? neg[j] / nn : 0));
}

/** The typical length of a residual vector at one snapshot: steering strength is a multiple of it. */
export function meanNorm(rows: ArrayLike<number>[]): number {
  return rows.reduce((s, r) => s + Math.sqrt(dot(r, r)), 0) / (rows.length || 1);
}

/** A unit direction scaled to `alpha` times the snapshot's typical residual length. */
export function scaledDirection(v: ArrayLike<number>, alpha: number, norm: number): number[] {
  const len = Math.sqrt(dot(v, v)) || 1;
  return Array.from(v, (x) => (x / len) * alpha * norm);
}

/** Mean change in log p(word) per word, steered minus plain, over matching prefixes.
 * Log probabilities are not forced to sum to anything, unlike the renormalised shares. */
export function meanLogDelta(plain: number[][], steered: number[][], plainMass: number[], steeredMass: number[]): number[] {
  const K = plain[0]?.length ?? 0, out = new Array(K).fill(0);
  for (let i = 0; i < plain.length; i++) for (let k = 0; k < K; k++) {
    const a = plain[i][k] * plainMass[i], b = steered[i][k] * steeredMass[i];
    out[k] += (Math.log(b + 1e-12) - Math.log(a + 1e-12)) / plain.length;
  }
  return out;
}

/** Mean of each column. */
export function columnMeans(rows: ArrayLike<number>[]): number[] {
  const K = rows[0]?.length ?? 0, out = new Array(K).fill(0);
  for (const r of rows) for (let k = 0; k < K; k++) out[k] += r[k] / rows.length;
  return out;
}
