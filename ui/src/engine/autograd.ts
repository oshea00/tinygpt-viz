// A small reverse-mode autograd over the handful of operations the GPT uses, on Float32Arrays.
// Tensors are 2-D [rows, cols] (traces are batch size 1), 1-D for biases and gains, or [1] for
// the loss (as the trace frame stores it). Each op records a backward closure on the tape;
// backward() walks it in reverse.

export class Tensor implements Node {
  grad: Float32Array | null = null;

  constructor(
    public data: Float32Array,
    public shape: number[],
    public requiresGrad = false,
  ) {}

  get rows(): number {
    return this.shape[0];
  }

  get cols(): number {
    return this.shape[this.shape.length - 1];
  }

  /** The gradient buffer, allocated (as zeros) on first use. */
  g(): Float32Array {
    return (this.grad ??= new Float32Array(this.data.length));
  }
}

/** What the tape needs of a tensor, on the CPU or the GPU. */
export interface Node {
  requiresGrad: boolean;
  grad: unknown;
}

/** Records backward closures in execution order. */
export class Tape {
  private steps: (() => void)[] = [];
  enabled = true;

  needs(inputs: Node[]): boolean {
    return this.enabled && inputs.some((t) => t.requiresGrad);
  }

  /** An op's backward step; it runs only if something downstream gave `o` a gradient. */
  push(o: Node, backward: () => void): void {
    if (o.requiresGrad) this.steps.push(() => o.grad && backward());
  }

  /** Run every backward step, last op first. The caller seeds the root's gradient. */
  backward(): void {
    for (let i = this.steps.length - 1; i >= 0; i--) this.steps[i]();
    this.steps = [];
  }
}

/** The operations the GPT is written in, on CPU tensors (Ops) or GPU ones (GpuOps in ./gpu). */
export interface OpSet<T extends Node> {
  embed(W: T, ids: number[]): T;
  rows(W: T, start: number, n: number): T;
  add(a: T, b: T): T;
  linear(x: T, W: T, b: T | null): T;
  linearT(x: T, W: T): T;
  layerNorm(x: T, gamma: T, beta: T): T;
  gelu(x: T): T;
  cols(x: T, c0: number, c1: number): T;
  concatCols(parts: T[]): T;
  scores(q: T, k: T, scale: number): T;
  causalMask(s: T): T;
  softmax(x: T): T;
  mix(A: T, V: T): T;
  crossEntropy(logits: T, targets: number[]): T;
  probs(logits: T): T;        // softmax of the logits, not on the loss path
}

// -- matrix products ------------------------------------------------------

/** C[m,n] += A[m,k] B[k,n] */
function mm(A: Float32Array, B: Float32Array, C: Float32Array, m: number, k: number, n: number): void {
  for (let i = 0; i < m; i++) {
    const ci = i * n;
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const bp = p * n;
      for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
    }
  }
}

/** C[m,n] += A[m,k] B[n,k]ᵀ */
function mmNT(A: Float32Array, B: Float32Array, C: Float32Array, m: number, k: number, n: number): void {
  for (let i = 0; i < m; i++) {
    const ai = i * k;
    for (let j = 0; j < n; j++) {
      const bj = j * k;
      let s = 0;
      for (let p = 0; p < k; p++) s += A[ai + p] * B[bj + p];
      C[i * n + j] += s;
    }
  }
}

/** C[m,n] += A[k,m]ᵀ B[k,n] */
function mmTN(A: Float32Array, B: Float32Array, C: Float32Array, k: number, m: number, n: number): void {
  for (let p = 0; p < k; p++) {
    const ap = p * m, bp = p * n;
    for (let i = 0; i < m; i++) {
      const a = A[ap + i];
      if (a === 0) continue;
      const ci = i * n;
      for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
    }
  }
}

// -- ops --------------------------------------------------------------------

export class Ops implements OpSet<Tensor> {
  constructor(public tape: Tape) {}

  /** An output tensor that needs a gradient if any input does, with its backward step. */
  private out(data: Float32Array, shape: number[], inputs: Tensor[], backward: (o: Tensor) => void): Tensor {
    const o = new Tensor(data, shape, this.tape.needs(inputs));
    this.tape.push(o, () => backward(o));
    return o;
  }

  /** The loss's own gradient, dL/dL = 1, to start backward from. */
  seed(root: Tensor): void {
    root.g().fill(1);
  }

  probs(logits: Tensor): Tensor {
    return new Tensor(softmaxRows(logits.data, logits.rows, logits.cols), logits.shape);
  }

  /** Rows of W picked by ids: the token embedding lookup. */
  embed(W: Tensor, ids: number[]): Tensor {
    const d = W.cols;
    const out = new Float32Array(ids.length * d);
    ids.forEach((id, t) => out.set(W.data.subarray(id * d, (id + 1) * d), t * d));
    return this.out(out, [ids.length, d], [W], (o) => {
      const g = W.g();
      ids.forEach((id, t) => {
        for (let j = 0; j < d; j++) g[id * d + j] += o.grad![t * d + j];
      });
    });
  }

  /** Rows start..start+n of W: the position embeddings for the prompt's positions. */
  rows(W: Tensor, start: number, n: number): Tensor {
    const d = W.cols;
    return this.out(W.data.slice(start * d, (start + n) * d), [n, d], [W], (o) => {
      const g = W.g();
      for (let i = 0; i < n * d; i++) g[start * d + i] += o.grad![i];
    });
  }

  add(a: Tensor, b: Tensor): Tensor {
    const out = new Float32Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
    return this.out(out, a.shape, [a, b], (o) => {
      for (const t of [a, b]) {
        if (!t.requiresGrad) continue;
        const g = t.g();
        for (let i = 0; i < g.length; i++) g[i] += o.grad![i];
      }
    });
  }

  /** x W + b, with W shaped [d_in, d_out]. */
  linear(x: Tensor, W: Tensor, b: Tensor | null): Tensor {
    const [m, k] = x.shape, n = W.cols;
    const out = new Float32Array(m * n);
    if (b) for (let i = 0; i < m; i++) out.set(b.data, i * n);
    mm(x.data, W.data, out, m, k, n);
    return this.out(out, [m, n], b ? [x, W, b] : [x, W], (o) => {
      const g = o.grad!;
      if (x.requiresGrad) mmNT(g, W.data, x.g(), m, n, k);
      if (W.requiresGrad) mmTN(x.data, g, W.g(), m, k, n);
      if (b?.requiresGrad) {
        const gb = b.g();
        for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) gb[j] += g[i * n + j];
      }
    });
  }

  /** x Wᵀ, with W shaped [n, d_in]: the tied unembedding (W is the token embedding table). */
  linearT(x: Tensor, W: Tensor): Tensor {
    const [m, k] = x.shape, n = W.rows;
    const out = new Float32Array(m * n);
    mmNT(x.data, W.data, out, m, k, n);
    return this.out(out, [m, n], [x, W], (o) => {
      if (x.requiresGrad) mm(o.grad!, W.data, x.g(), m, n, k);
      if (W.requiresGrad) mmTN(o.grad!, x.data, W.g(), m, n, k);
    });
  }

  layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
    const [m, d] = x.shape;
    const out = new Float32Array(m * d);
    const xhat = new Float32Array(m * d);
    const rstd = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      const r = x.data.subarray(i * d, (i + 1) * d);
      let mean = 0;
      for (let j = 0; j < d; j++) mean += r[j];
      mean /= d;
      let v = 0;
      for (let j = 0; j < d; j++) v += (r[j] - mean) ** 2;
      rstd[i] = 1 / Math.sqrt(v / d + eps);
      for (let j = 0; j < d; j++) {
        const h = (r[j] - mean) * rstd[i];
        xhat[i * d + j] = h;
        out[i * d + j] = h * gamma.data[j] + beta.data[j];
      }
    }
    return this.out(out, [m, d], [x, gamma, beta], (o) => {
      const g = o.grad!;
      const gg = gamma.requiresGrad ? gamma.g() : null, gb = beta.requiresGrad ? beta.g() : null;
      const gx = x.requiresGrad ? x.g() : null;
      for (let i = 0; i < m; i++) {
        let mg = 0, mgx = 0;
        for (let j = 0; j < d; j++) {
          const k = i * d + j, gh = g[k] * gamma.data[j];
          mg += gh;
          mgx += gh * xhat[k];
          if (gg) gg[j] += g[k] * xhat[k];
          if (gb) gb[j] += g[k];
        }
        mg /= d;
        mgx /= d;
        if (gx) for (let j = 0; j < d; j++) {
          const k = i * d + j;
          gx[k] += rstd[i] * (g[k] * gamma.data[j] - mg - xhat[k] * mgx);
        }
      }
    });
  }

  /** The tanh approximation used by GPT-2. */
  gelu(x: Tensor): Tensor {
    const c = Math.sqrt(2 / Math.PI);
    const out = new Float32Array(x.data.length);
    for (let i = 0; i < out.length; i++) {
      const u = x.data[i];
      out[i] = 0.5 * u * (1 + Math.tanh(c * (u + 0.044715 * u ** 3)));
    }
    return this.out(out, x.shape, [x], (o) => {
      const gx = x.g();
      for (let i = 0; i < gx.length; i++) {
        const u = x.data[i], th = Math.tanh(c * (u + 0.044715 * u ** 3));
        const dth = (1 - th * th) * c * (1 + 3 * 0.044715 * u * u);
        gx[i] += o.grad![i] * (0.5 * (1 + th) + 0.5 * u * dth);
      }
    });
  }

  /** Columns c0..c1 of x: one head's band. */
  cols(x: Tensor, c0: number, c1: number): Tensor {
    const [m, n] = x.shape, w = c1 - c0;
    const out = new Float32Array(m * w);
    for (let i = 0; i < m; i++) out.set(x.data.subarray(i * n + c0, i * n + c1), i * w);
    return this.out(out, [m, w], [x], (o) => {
      const gx = x.g();
      for (let i = 0; i < m; i++) for (let j = 0; j < w; j++) gx[i * n + c0 + j] += o.grad![i * w + j];
    });
  }

  concatCols(parts: Tensor[]): Tensor {
    const m = parts[0].rows, n = parts.reduce((s, p) => s + p.cols, 0);
    const out = new Float32Array(m * n);
    let c0 = 0;
    const offsets = parts.map((p) => {
      for (let i = 0; i < m; i++) out.set(p.data.subarray(i * p.cols, (i + 1) * p.cols), i * n + c0);
      c0 += p.cols;
      return c0 - p.cols;
    });
    return this.out(out, [m, n], parts, (o) => {
      parts.forEach((p, k) => {
        if (!p.requiresGrad) return;
        const gp = p.g(), w = p.cols;
        for (let i = 0; i < m; i++) for (let j = 0; j < w; j++) gp[i * w + j] += o.grad![i * n + offsets[k] + j];
      });
    });
  }

  /** q kᵀ · scale: every query against every key. */
  scores(q: Tensor, k: Tensor, scale: number): Tensor {
    const [m, d] = q.shape, n = k.rows;
    const out = new Float32Array(m * n);
    mmNT(q.data, k.data, out, m, d, n);
    for (let i = 0; i < out.length; i++) out[i] *= scale;
    return this.out(out, [m, n], [q, k], (o) => {
      const g = o.grad!.map((v) => v * scale);
      if (q.requiresGrad) mm(g, k.data, q.g(), m, n, d);
      if (k.requiresGrad) mmTN(g, q.data, k.g(), m, n, d);
    });
  }

  /** -∞ above the diagonal. `offset` is the first query's position (the KV cache's length). */
  causalMask(s: Tensor, offset = 0): Tensor {
    const [m, n] = s.shape;
    const out = s.data.slice();
    for (let i = 0; i < m; i++) for (let j = i + offset + 1; j < n; j++) out[i * n + j] = -Infinity;
    return this.out(out, [m, n], [s], (o) => {
      const gs = s.g();
      for (let i = 0; i < m; i++) for (let j = 0; j <= Math.min(i + offset, n - 1); j++) gs[i * n + j] += o.grad![i * n + j];
    });
  }

  /** Softmax over each row. */
  softmax(x: Tensor): Tensor {
    const [m, n] = x.shape;
    const out = softmaxRows(x.data, m, n);
    return this.out(out, [m, n], [x], (o) => {
      const gx = x.g(), g = o.grad!;
      for (let i = 0; i < m; i++) {
        let dot = 0;
        for (let j = 0; j < n; j++) dot += g[i * n + j] * out[i * n + j];
        for (let j = 0; j < n; j++) gx[i * n + j] += out[i * n + j] * (g[i * n + j] - dot);
      }
    });
  }

  /** A V: the weighted average of value rows. */
  mix(A: Tensor, V: Tensor): Tensor {
    const [m, n] = A.shape, d = V.cols;
    const out = new Float32Array(m * d);
    mm(A.data, V.data, out, m, n, d);
    return this.out(out, [m, d], [A, V], (o) => {
      if (A.requiresGrad) mmNT(o.grad!, V.data, A.g(), m, d, n);
      if (V.requiresGrad) mmTN(A.data, o.grad!, V.g(), m, n, d);
    });
  }

  /** Mean cross-entropy over positions whose target is not -1. */
  crossEntropy(logits: Tensor, targets: number[]): Tensor {
    const [m, n] = logits.shape;
    const probs = softmaxRows(logits.data, m, n);
    const kept = targets.filter((t) => t >= 0).length;
    let loss = 0;
    targets.forEach((t, i) => {
      if (t < 0) return;
      // log softmax written out, as the model does: z_t - logsumexp(z)
      const row = logits.data.subarray(i * n, (i + 1) * n);
      let max = -Infinity;
      for (const v of row) max = Math.max(max, v);
      let s = 0;
      for (const v of row) s += Math.exp(v - max);
      loss -= row[t] - max - Math.log(s);
    });
    return this.out(new Float32Array([loss / kept]), [1], [logits], (o) => {
      const gl = logits.g(), g = o.grad![0] / kept;
      targets.forEach((t, i) => {
        if (t < 0) return;
        for (let j = 0; j < n; j++) gl[i * n + j] += g * (probs[i * n + j] - (j === t ? 1 : 0));
      });
    });
  }
}

export function softmaxRows(x: Float32Array, m: number, n: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    let max = -Infinity;
    for (let j = 0; j < n; j++) max = Math.max(max, x[i * n + j]);
    let s = 0;
    for (let j = 0; j < n; j++) s += out[i * n + j] = Math.exp(x[i * n + j] - max);
    for (let j = 0; j < n; j++) out[i * n + j] /= s;
  }
  return out;
}
