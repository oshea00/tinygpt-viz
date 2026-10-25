// The forward pass in plain TypeScript, for the static demo: no Python, inference only.
// It produces the same stages, with the same ids, as core/model.py; a parity test checks
// every one against a trace from the Python core.

import type { ModelConfig, Stage, StageKind, Tensor, Trace } from "../types";

type Mat = { rows: number; cols: number; data: Float32Array };

const mat = (rows: number, cols: number): Mat => ({ rows, cols, data: new Float32Array(rows * cols) });
const fromTensor = (t: Tensor): Mat => ({ rows: t.shape.length > 1 ? t.shape[0] : 1, cols: t.shape[t.shape.length - 1], data: t.data });

/** y = x W + b, with W shaped [d_in, d_out] as in the Python model. */
function linear(x: Mat, W: Mat, b?: Float32Array): Mat {
  const y = mat(x.rows, W.cols);
  for (let t = 0; t < x.rows; t++) {
    for (let j = 0; j < W.cols; j++) {
      let s = b ? b[j] : 0;
      for (let i = 0; i < x.cols; i++) s += x.data[t * x.cols + i] * W.data[i * W.cols + j];
      y.data[t * W.cols + j] = s;
    }
  }
  return y;
}

function layerNorm(x: Mat, g: Float32Array, b: Float32Array, eps = 1e-5): Mat {
  const y = mat(x.rows, x.cols);
  for (let t = 0; t < x.rows; t++) {
    const r = x.data.subarray(t * x.cols, (t + 1) * x.cols);
    let mean = 0;
    for (const v of r) mean += v;
    mean /= x.cols;
    let v2 = 0;
    for (const v of r) v2 += (v - mean) ** 2;
    const sd = Math.sqrt(v2 / x.cols + eps);
    for (let i = 0; i < x.cols; i++) y.data[t * x.cols + i] = ((r[i] - mean) / sd) * g[i] + b[i];
  }
  return y;
}

const gelu = (u: number) => 0.5 * u * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (u + 0.044715 * u ** 3)));

function softmaxRows(x: Mat): Mat {
  const y = mat(x.rows, x.cols);
  for (let t = 0; t < x.rows; t++) {
    let max = -Infinity;
    for (let i = 0; i < x.cols; i++) max = Math.max(max, x.data[t * x.cols + i]);
    let total = 0;
    for (let i = 0; i < x.cols; i++) total += (y.data[t * x.cols + i] = Math.exp(x.data[t * x.cols + i] - max));
    for (let i = 0; i < x.cols; i++) y.data[t * x.cols + i] /= total;
  }
  return y;
}

function map(x: Mat, f: (v: number, r: number, c: number) => number): Mat {
  const y = mat(x.rows, x.cols);
  for (let r = 0; r < x.rows; r++) for (let c = 0; c < x.cols; c++) y.data[r * x.cols + c] = f(x.data[r * x.cols + c], r, c);
  return y;
}

function columns(x: Mat, from: number, to: number): Mat {
  return map(mat(x.rows, to - from), (_, r, c) => x.data[r * x.cols + from + c]);
}

export interface StageInfo {
  kind: StageKind;
  layer?: number;
  head?: number;
  inputs: string[];
  weights: string[];
  formula: string;
  explain: string;
  backward: string;
}

/** Runs the model on `tokens` (at most block_size). `info` supplies each stage's description
 * (taken from a reference trace exported by the Python core). */
export function forward(tokens: number[], weights: Record<string, Tensor>, cfg: ModelConfig,
                        info: Map<string, StageInfo>, text: string[]): Trace {
  const W = (n: string) => fromTensor(weights[n]);
  const v = (n: string) => weights[n].data;
  const T = tokens.length, d = cfg.d_model, H = cfg.n_head, dh = d / H;
  const stages: Stage[] = [];
  const tap = (id: string, m: Mat | number): Mat => {
    const meta = info.get(id) ?? { kind: "add" as StageKind, inputs: [], weights: [], formula: "", explain: "", backward: "" };
    // A scalar arrives as shape [1] from the core (numpy makes 0-d arrays 1-d when encoding).
    const output: Tensor = typeof m === "number" ? { shape: [1], data: new Float32Array([m]) } : { shape: [m.rows, m.cols], data: m.data };
    stages.push({ id, ...meta, output });
    return typeof m === "number" ? mat(1, 1) : m;
  };

  const wte = W("wte"), wpe = W("wpe");
  const tok = tap("embed.tok", map(mat(T, d), (_, t, i) => wte.data[tokens[t] * d + i]));
  const pos = tap("embed.pos", map(mat(T, d), (_, t, i) => wpe.data[t * d + i]));
  let x = tap("embed.sum", map(tok, (a, t, i) => a + pos.data[t * d + i]));

  for (let l = 0; l < cfg.n_layer; l++) {
    const P = `L${l}.`;
    const ln1 = tap(P + "ln1", layerNorm(x, v(P + "ln1_g"), v(P + "ln1_b")));
    const q = tap(P + "q", linear(ln1, W(P + "W_q"), v(P + "b_q")));
    const k = tap(P + "k", linear(ln1, W(P + "W_k"), v(P + "b_k")));
    const vv = tap(P + "v", linear(ln1, W(P + "W_v"), v(P + "b_v")));
    const heads: Mat[] = [];
    for (let h = 0; h < H; h++) {
      const p = `${P}attn.H${h}.`;
      const qh = tap(p + "q", columns(q, h * dh, (h + 1) * dh));
      const kh = tap(p + "k", columns(k, h * dh, (h + 1) * dh));
      const vh = tap(p + "v", columns(vv, h * dh, (h + 1) * dh));
      const scores = tap(p + "scores", map(mat(T, T), (_, i, j) => {
        let s = 0;
        for (let c = 0; c < dh; c++) s += qh.data[i * dh + c] * kh.data[j * dh + c];
        return s / Math.sqrt(dh);
      }));
      const masked = tap(p + "masked", map(scores, (s, i, j) => (j > i ? -Infinity : s)));
      const wts = tap(p + "weights", softmaxRows(masked));
      heads.push(tap(p + "mix", map(mat(T, dh), (_, i, c) => {
        let s = 0;
        for (let j = 0; j <= i; j++) s += wts.data[i * T + j] * vh.data[j * dh + c];
        return s;
      })));
    }
    const concat = tap(P + "attn.concat", map(mat(T, d), (_, t, c) => heads[Math.floor(c / dh)].data[t * dh + (c % dh)]));
    const attn = tap(P + "attn.out", linear(concat, W(P + "W_o"), v(P + "b_o")));
    x = tap(P + "resid_attn", map(x, (a, t, i) => a + attn.data[t * d + i]));
    const ln2 = tap(P + "ln2", layerNorm(x, v(P + "ln2_g"), v(P + "ln2_b")));
    const up = tap(P + "mlp.up", linear(ln2, W(P + "W_up"), v(P + "b_up")));
    const act = tap(P + "mlp.gelu", map(up, gelu));
    const down = tap(P + "mlp.down", linear(act, W(P + "W_down"), v(P + "b_down")));
    x = tap(P + "resid_mlp", map(x, (a, t, i) => a + down.data[t * d + i]));
  }

  const lnf = tap("ln_f", layerNorm(x, v("ln_f_g"), v("ln_f_b")));
  // Untied: x @ W_unembed [d, V]. Tied: x @ wteᵀ.
  const U = weights["W_unembed"] ? W("W_unembed") : map(mat(d, cfg.vocab_size), (_, i, j) => wte.data[j * d + i]);
  const logits = tap("logits", linear(lnf, U));
  const probs = tap("probs", softmaxRows(logits));
  const targets = [...tokens.slice(1), -1];
  const nll = targets.flatMap((y, t) => (y >= 0 ? [-Math.log(probs.data[t * cfg.vocab_size + y])] : []));
  const loss = nll.length ? nll.reduce((a, b) => a + b, 0) / nll.length : null;
  if (loss !== null) tap("loss", loss);

  return { step: 0, loss, tokens, text, targets, stages, byId: new Map(stages.map((s) => [s.id, s])), weightGrads: {} };
}

export function infoFrom(trace: Trace): Map<string, StageInfo> {
  return new Map(trace.stages.map((s) => [s.id, {
    kind: s.kind, layer: s.layer, head: s.head, inputs: s.inputs, weights: s.weights,
    formula: s.formula, explain: s.explain, backward: s.backward,
  }]));
}
