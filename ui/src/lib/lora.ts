// What the fine-tune tab shows about an adapter (core/lora.py): the change ΔW = A·B it makes to a
// weight, its size, and its singular values (at most r of them: ΔW has rank r at most).

import type { Tensor } from "../types";

/** ΔW = A·B, [d_in, d_out], times the strength. */
export function deltaW(A: Tensor, B: Tensor, scale = 1): Tensor {
  const [n, r] = A.shape, m = B.shape[1];
  const out = new Float32Array(n * m);
  for (let i = 0; i < n; i++) for (let k = 0; k < r; k++) {
    const a = A.data[i * r + k] * scale;
    if (a) for (let j = 0; j < m; j++) out[i * m + j] += a * B.data[k * m + j];
  }
  return { shape: [n, m], data: out };
}

/** The Frobenius norm: the square root of the sum of squares. */
export function frob(t: Tensor | Float32Array): number {
  let s = 0;
  for (const v of "data" in t ? t.data : t) s += v * v;
  return Math.sqrt(s);
}

/** Gram matrix XᵀX of the columns of a [rows, k] matrix (transpose: of the rows of a [k, cols] one). */
function gram(x: Tensor, rowsOf = false): number[][] {
  const [a, b] = x.shape, k = rowsOf ? a : b;
  const g = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  for (let p = 0; p < k; p++) for (let q = p; q < k; q++) {
    let s = 0;
    if (rowsOf) for (let j = 0; j < b; j++) s += x.data[p * b + j] * x.data[q * b + j];
    else for (let i = 0; i < a; i++) s += x.data[i * b + p] * x.data[i * b + q];
    g[p][q] = g[q][p] = s;
  }
  return g;
}

/** Eigenvalues of a small symmetric matrix (cyclic Jacobi rotations). */
export function symEigenvalues(m: number[][]): number[] {
  const a = m.map((r) => [...r]), n = a.length;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
    }
  }
  return a.map((r, i) => r[i]);
}

/** The singular values of A·B, largest first, from r × r matrices only: they are the square roots
 * of the eigenvalues of Lᵀ(B Bᵀ)L, where AᵀA = L Lᵀ. */
export function singularValues(A: Tensor, B: Tensor, scale = 1): number[] {
  const G = gram(A), H = gram(B, true), r = G.length;
  // Cholesky of G (with a whisker added, in case A's columns are dependent).
  const L = Array.from({ length: r }, () => new Array<number>(r).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j <= i; j++) {
    let s = G[i][j] + (i === j ? 1e-12 : 0);
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 0)) : s / (L[j][j] || 1);
  }
  const M = Array.from({ length: r }, (_, p) => Array.from({ length: r }, (_, q) => {
    let s = 0;
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) s += L[i][p] * H[i][j] * L[j][q];
    return s;
  }));
  return symEigenvalues(M).map((e) => Math.abs(scale) * Math.sqrt(Math.max(e, 0))).sort((a, b) => b - a);
}

/** The weight a target adapts ("L3.q" → "L3.W_q"), as core/stages.py's LORA_TARGETS. */
export function baseWeight(target: string): string {
  const i = target.indexOf(".");
  const W = { q: "W_q", k: "W_k", v: "W_v", "attn.out": "W_o", "mlp.up": "W_up", "mlp.down": "W_down" }[target.slice(i + 1)];
  return `${target.slice(0, i)}.${W}`;
}

/** Does generated text start with the refusal (core/dpo.py's REFUSAL_START)? */
export const refuses = (text: string, refusal: string) => text.trimStart().startsWith(refusal.trim().split(".")[0]);
