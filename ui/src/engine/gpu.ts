// The GPT's operations as WebGPU compute shaders: the same OpSet as the CPU Ops in ./autograd,
// so ./model's forward() runs unchanged. Every dispatch of a trace (forward and backward) goes
// into one compute pass; stage outputs and gradients live in one arena, read back in one copy.

import { Tape, Tensor, type Node, type OpSet } from "./autograd";
import { frameAround, traceHeader } from "./frames";
import { ADAM_EPS, BETAS, runForward, type Engine, type ForwardOptions, type StepSettings, type WeightSet } from "./engine";
import { forward, inStageOrder, prepare, recorder, type ModelConfig, type TraceOptions, type TraceResult, type Weights } from "./model";

const F = 4;                    // bytes per float
const ALIGN = 256;              // minStorageBufferOffsetAlignment, the most any device asks

/** A tensor on the GPU: a range of a buffer. */
/** A layer's heads, one tensor each, `stride` bytes apart in one buffer: one dispatch serves all. */
export interface Block {
  members: GTensor[];
  stride: number;
}

export class GTensor implements Node {
  grad: GTensor | null = null;
  block?: Block;                // set when this tensor is one head of a block

  constructor(
    public buf: GPUBuffer,
    public offset: number,      // bytes
    public shape: number[],
    public requiresGrad = false,
  ) {}

  get length(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get rows(): number {
    return this.shape[0];
  }

  get cols(): number {
    return this.shape[this.shape.length - 1];
  }

  binding(): GPUBufferBinding {
    return { buffer: this.buf, offset: this.offset, size: Math.max(this.length, 1) * F };
  }
}

/** Bump allocation in one buffer, cleared and reused trace after trace. */
export class Arena {
  used = 0;

  constructor(public buf: GPUBuffer) {}

  alloc(shape: number[]): GTensor {
    const bytes = Math.max(shape.reduce((a, b) => a * b, 1), 1) * F;
    const at = this.used;
    if (at + bytes > this.buf.size) throw new Error("trace does not fit the GPU arena");
    this.used = Math.ceil((at + bytes) / ALIGN) * ALIGN;
    return new GTensor(this.buf, at, shape);
  }
}

// -- shaders ----------------------------------------------------------------

// Every kernel reads its sizes from P: sixteen u32 slots (floats are bit-cast).
const HEADER = /* wgsl */ `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32,
           i: u32, j: u32, k: u32, l: u32, m: u32, n: u32, o: u32, q: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;
const LOWEST = "-3.4028234663852886e38";   // the lowest f32, to start a running max
const NEG_INF_BITS = 0xff800000;            // WGSL has no -∞ constant: the mask gets it at runtime

// A row-wise reduction across a 256-thread workgroup.
const REDUCE = /* wgsl */ `
var<workgroup> red: array<f32, 256>;
fn reduce_sum(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn reduce_max(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) { red[lid] = max(red[lid], red[lid + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

const SHADERS: Record<string, string> = {
  // C[m,n] (=|+=) alpha · op(A) op(B) (+ bias). a=m b=n c=k d=transA e=transB f=accumulate g=bias h=alpha.
  // op(A) is [m,k]: stored [k,m] when transposed. op(B) is [k,n]: stored [n,k] when transposed.
  // Batched over workgroup z: z = head · l + sequence, with element strides per sequence (i, j, k)
  // and per head (m, n, o) for A, B and C. l = sequences per head (0 means 1).
  matmul: `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
fn a_at(o: u32, r: u32, c: u32) -> f32 {
  if (r >= p.a || c >= p.c) { return 0.0; }
  if (p.d == 1u) { return A[o + c * p.a + r]; }
  return A[o + r * p.c + c];
}
fn b_at(o: u32, r: u32, c: u32) -> f32 {
  if (r >= p.c || c >= p.b) { return 0.0; }
  if (p.e == 1u) { return B[o + c * p.c + r]; }
  return B[o + r * p.b + c];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.y * 16u + lid.y;
  let col = wg.x * 16u + lid.x;
  let per = max(p.l, 1u);
  let zh = wg.z / per;
  let zb = wg.z % per;
  let oa = zh * p.m + zb * p.i;
  let ob = zh * p.n + zb * p.j;
  var s = 0.0;
  for (var t = 0u; t < (p.c + 15u) / 16u; t++) {
    As[lid.y][lid.x] = a_at(oa, row, t * 16u + lid.x);
    Bs[lid.y][lid.x] = b_at(ob, t * 16u + lid.y, col);
    workgroupBarrier();
    for (var i = 0u; i < 16u; i++) { s += As[lid.y][i] * Bs[i][lid.x]; }
    workgroupBarrier();
  }
  if (row < p.a && col < p.b) {
    var v = bitcast<f32>(p.h) * s;
    if (p.g == 1u) { v += bias[col]; }
    let i = zh * p.o + zb * p.k + row * p.b + col;
    if (p.f == 1u) { v += C[i]; }
    C[i] = v;
  }
}`,

  // dst[i·dn + d0 + j] (=|+=) src[i·sn + s0 + j], for i < m, j < w. a=m b=w c=sn d=s0 e=dn f=d0 g=accumulate.
  // Head splits, concatenation, position rows, residual adds' backward: all are bands.
  band: `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  let v = src[i * p.c + p.d + j];
  let k = i * p.e + p.f + j;
  if (p.g == 1u) { dst[k] += v; } else { dst[k] = v; }
}`,

  // gW[c + j] += Σ_r g[r·b + j] for j < b, r < a: the backward of rows() repeated per sequence.
  repeat_back: `
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> gW: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.y * 65535u * 64u + id.x;
  if (j >= p.b) { return; }
  var s = 0.0;
  for (var r = 0u; r < p.a; r++) { s += g[r * p.b + j]; }
  gW[p.c + j] += s;
}`,

  // Heads to and from their bands of columns: full [a rows, c cols] and head z [a, b] (d elements
  // apart). e=0: head z (=|+=) its band of full; e=1: the band of full (=|+=) head z. f=accumulate.
  heads: `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  let full = i * p.c + id.z * p.b + j;
  let head = id.z * p.d + i * p.b + j;
  var si = full;
  var di = head;
  if (p.e == 1u) { si = head; di = full; }
  if (p.f == 1u) { dst[di] += src[si]; } else { dst[di] = src[si]; }
}`,

  // out = x + y, n = a.
  add: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.y * 65535u * 256u + id.x;
  if (i < p.a) { out[i] = x[i] + y[i]; }
}`,

  // Token embedding: out[t] = W[ids[t]], a=T b=d.
  embed: `
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let t = id.y;
  if (t >= p.a || j >= p.b) { return; }
  out[t * p.b + j] = W[ids[t] * p.b + j];
}`,

  // Its backward: one thread per column, so repeated tokens never race. a=T b=d.
  embed_back: `
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> gW: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  if (j >= p.b) { return; }
  for (var t = 0u; t < p.a; t++) { gW[ids[t] * p.b + j] += g[t * p.b + j]; }
}`,

  // Σ over rows of g, into a column vector (bias and layer-norm β gradients). a=m b=n.
  colsum: `
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  if (j >= p.b) { return; }
  var s = 0.0;
  for (var i = 0u; i < p.a; i++) { s += g[i * p.b + j]; }
  out[j] += s;
}`,

  // Layer norm, one workgroup per row: out, and x̂ and 1/σ for backward. a=d.
  layernorm: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(6) var<storage, read_write> rstd: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * p.a;
  var s = 0.0;
  for (var j = lid.x; j < p.a; j += 256u) { s += x[r + j]; }
  let mean = reduce_sum(s, lid.x) / f32(p.a);
  var v = 0.0;
  for (var j = lid.x; j < p.a; j += 256u) { let c = x[r + j] - mean; v += c * c; }
  let rs = 1.0 / sqrt(reduce_sum(v, lid.x) / f32(p.a) + 1e-5);
  for (var j = lid.x; j < p.a; j += 256u) {
    let h = (x[r + j] - mean) * rs;
    xhat[r + j] = h;
    out[r + j] = h * gamma[j] + beta[j];
  }
  if (lid.x == 0u) { rstd[wg.x] = rs; }
}`,

  // ∂x of layer norm, one workgroup per row. a=d.
  layernorm_back: `
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> xhat: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> gx: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * p.a;
  var s = 0.0;
  var sx = 0.0;
  for (var j = lid.x; j < p.a; j += 256u) { let gh = g[r + j] * gamma[j]; s += gh; sx += gh * xhat[r + j]; }
  let mg = reduce_sum(s, lid.x) / f32(p.a);
  let mgx = reduce_sum(sx, lid.x) / f32(p.a);
  for (var j = lid.x; j < p.a; j += 256u) {
    gx[r + j] += rstd[wg.x] * (g[r + j] * gamma[j] - mg - xhat[r + j] * mgx);
  }
}`,

  // ∂γ = Σ_rows g ⊙ x̂. a=m b=d.
  layernorm_gamma: `
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> xhat: array<f32>;
@group(0) @binding(3) var<storage, read_write> gg: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  if (j >= p.b) { return; }
  var s = 0.0;
  for (var i = 0u; i < p.a; i++) { s += g[i * p.b + j] * xhat[i * p.b + j]; }
  gg[j] += s;
}`,

  // GELU (tanh form), and with a=n b=backward: gx += g · GELU'(u).
  gelu: `
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.y * 65535u * 256u + id.x;
  if (i >= p.a) { return; }
  let c = 0.7978845608028654;
  let x = u[i];
  let th = tanh(c * (x + 0.044715 * x * x * x));
  if (p.b == 0u) {
    out[i] = 0.5 * x * (1.0 + th);
  } else {
    let dth = (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * x * x);
    out[i] += g[i] * (0.5 * (1.0 + th) + 0.5 * x * dth);
  }
}`,

  // The causal mask: a=m b=n c=offset d=backward e=the bits of -∞ f=rows per sequence; over
  // heads in z, g elements apart. Forward writes -∞ above the diagonal; backward passes the
  // gradient through below it.
  mask: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  let k = id.z * p.g + i * p.b + j;
  let keep = j <= i % p.f + p.c;
  if (p.d == 0u) {
    if (keep) { out[k] = x[k]; } else { out[k] = bitcast<f32>(p.e); }
  } else if (keep) {
    out[k] += x[k];
  }
}`,

  // Softmax per row, one workgroup per row (and per head in y, e elements apart). a=n b=causal
  // c=offset d=rows per sequence: with causal set, columns past the diagonal are left out by
  // index rather than through -∞ arithmetic.
  softmax: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.y * p.e + wg.x * p.a;
  var n = p.a;
  if (p.b == 1u) { n = min(p.a, wg.x % p.d + p.c + 1u); }
  var m = ${LOWEST};
  for (var j = lid.x; j < n; j += 256u) { m = max(m, x[r + j]); }
  m = reduce_max(m, lid.x);
  var s = 0.0;
  for (var j = lid.x; j < n; j += 256u) { s += exp(x[r + j] - m); }
  s = reduce_sum(s, lid.x);
  for (var j = lid.x; j < p.a; j += 256u) {
    if (j < n) { out[r + j] = exp(x[r + j] - m) / s; } else { out[r + j] = 0.0; }
  }
}`,

  // ∂x += A ⊙ (g − Σ_j g_j A_j), one workgroup per row (and per head in y, b elements apart). a=n.
  softmax_back: `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> gx: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.y * p.b + wg.x * p.a;
  var s = 0.0;
  for (var j = lid.x; j < p.a; j += 256u) { s += g[r + j] * A[r + j]; }
  let dot = reduce_sum(s, lid.x);
  for (var j = lid.x; j < p.a; j += 256u) { gx[r + j] += A[r + j] * (g[r + j] - dot); }
}`,

  // Cross-entropy, one workgroup per row: −log softmax(z)[target] into nll (0 where the target is
  // −1). a=V.
  nll: `
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<i32>;
@group(0) @binding(3) var<storage, read_write> nll: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * p.a;
  var m = ${LOWEST};
  for (var j = lid.x; j < p.a; j += 256u) { m = max(m, z[r + j]); }
  m = reduce_max(m, lid.x);
  var s = 0.0;
  for (var j = lid.x; j < p.a; j += 256u) { s += exp(z[r + j] - m); }
  s = reduce_sum(s, lid.x);
  let t = targets[wg.x];
  if (lid.x == 0u) {
    if (t < 0) { nll[wg.x] = 0.0; } else { nll[wg.x] = -(z[r + u32(t)] - m - log(s)); }
  }
}`,

  // The mean of nll over kept positions: a=T h=1/kept.
  mean: `
@group(0) @binding(1) var<storage, read> nll: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var s = 0.0;
  for (var i = lid.x; i < p.a; i += 256u) { s += nll[i]; }
  s = reduce_sum(s, lid.x);
  if (lid.x == 0u) { out[0] = s * bitcast<f32>(p.h); }
}`,

  // ∂z += (p − y) · gL / kept, where p is the softmax of z. a=V h=1/kept.
  nll_back: `
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<i32>;
@group(0) @binding(3) var<storage, read> gL: array<f32>;
@group(0) @binding(4) var<storage, read_write> gz: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (j >= p.a) { return; }
  let t = targets[i];
  if (t < 0) { return; }
  var y = 0.0;
  if (j == u32(t)) { y = 1.0; }
  let k = i * p.a + j;
  gz[k] += (probs[k] - y) * gL[0] * bitcast<f32>(p.h);
}`,

  // Interventions. Columns c..d of x [a,b] set to 0 (e=0) or their mean over rows (e=1); with
  // f=1 this is the backward of the mean: out += the column's gradient sum / rows.
  colfill: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = p.c + id.x;
  if (j >= p.d) { return; }
  var s = 0.0;
  if (p.e == 1u) { for (var i = 0u; i < p.a; i++) { s += x[i * p.b + j]; } }
  for (var i = 0u; i < p.a; i++) {
    if (p.f == 0u) { out[i * p.b + j] = s / f32(p.a); } else { out[i * p.b + j] += s / f32(p.a); }
  }
}`,

  // out[index[k]] = values[k] (or 0 when b=1), k < a.
  scatter: `
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= p.a) { return; }
  if (p.b == 1u) { out[index[k]] = 0.0; } else { out[index[k]] = values[k]; }
}`,

  // out = x + scale · vec on every row of x [a,b]; h = scale.
  addrow: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  out[i * p.b + j] = x[i * p.b + j] + bitcast<f32>(p.h) * vec[j];
}`,

  // Row i of out is row perm[i] of x [a,b]; with c=1 the backward: out[perm[i]] += x[i].
  permute: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  if (p.c == 0u) { out[i * p.b + j] = x[perm[i] * p.b + j]; } else { out[perm[i] * p.b + j] += x[i * p.b + j]; }
}`,

  // Training. Σ x² over x [a] into out[b], one workgroup: a weight's share of the gradient norm.
  sumsq: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var s = 0.0;
  for (var i = lid.x; i < p.a; i += 256u) { s += x[i] * x[i]; }
  s = reduce_sum(s, lid.x);
  if (lid.x == 0u) { out[p.b] = s; }
}`,

  // The clipping coefficient from a weights' sums of squares: out = [min(1, clip / (norm + 1e-6)),
  // norm], as torch.nn.utils.clip_grad_norm_. a=count h=clip.
  clip: `
@group(0) @binding(1) var<storage, read> sq: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var s = 0.0;
  for (var i = lid.x; i < p.a; i += 256u) { s += sq[i]; }
  s = reduce_sum(s, lid.x);
  if (lid.x == 0u) {
    let norm = sqrt(s);
    out[0] = min(1.0, bitcast<f32>(p.h) / (norm + 1e-6));
    out[1] = norm;
  }
}`,

  // AdamW on one weight, as torch.optim.AdamW: the gradient is clipped (and kept clipped), the
  // weight decays, then moves by the bias-corrected moments. a=n b=lr c=decay d=β1 e=β2 f=ε
  // g=lr / (1 − β1^t) h=√(1 − β2^t), floats bit-cast.
  adamw: `
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read> coef: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.y * 65535u * 256u + id.x;
  if (i >= p.a) { return; }
  let b1 = bitcast<f32>(p.d);
  let b2 = bitcast<f32>(p.e);
  let gi = g[i] * coef[0];
  g[i] = gi;
  w[i] *= 1.0 - bitcast<f32>(p.b) * bitcast<f32>(p.c);
  m[i] += (gi - m[i]) * (1.0 - b1);
  v[i] = v[i] * b2 + (1.0 - b2) * gi * gi;
  w[i] -= bitcast<f32>(p.g) * m[i] / (sqrt(v[i]) / bitcast<f32>(p.h) + bitcast<f32>(p.f));
}`,

  // Fill with a constant: a=n h=value.
  fill: `
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < p.a) { out[id.x] = bitcast<f32>(p.h); }
}`,
};

const bits = (f: number) => new Uint32Array(new Float32Array([f]).buffer)[0];

/** A device, its pipelines, and the persistent buffers. */
export class Gpu {
  private pipelines = new Map<string, GPUComputePipeline>();

  readonly dummy: GPUBuffer;     // bound where a kernel's optional input is absent

  private constructor(public device: GPUDevice, public adapterName: string) {
    this.dummy = this.upload(new Float32Array(1));
  }

  static async open(): Promise<Gpu | null> {
    const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const limits = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        maxBufferSize: limits.maxBufferSize,
      },
    });
    const info = adapter.info;
    return new Gpu(device, [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "GPU");
  }

  pipeline(name: string): GPUComputePipeline {
    let p = this.pipelines.get(name);
    if (!p) {
      // Every tensor of a trace is a range of one arena buffer, and WebGPU rejects a dispatch that
      // binds one buffer both read-only and writable. Declaring every binding read_write is
      // allowed, since the ranges never overlap.
      const code = HEADER + SHADERS[name].replaceAll("var<storage, read>", "var<storage, read_write>");
      const module = this.device.createShaderModule({ code, label: name });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" }, label: name });
      this.pipelines.set(name, p);
    }
    return p;
  }

  /** A storage buffer holding `data`, e.g. a weight or the token ids. */
  upload(data: Float32Array | Uint32Array | Int32Array): GPUBuffer {
    const buf = this.device.createBuffer({
      size: Math.max(Math.ceil(data.byteLength / 4) * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }
}

/** The op set, encoding into one compute pass. Outputs and gradients come from the arena. */
export class GpuOps implements OpSet<GTensor> {
  private pass: GPUComputePassEncoder;
  private uniforms: GPUBuffer[] = [];
  private temps: GPUBuffer[] = [];

  /** Stage outputs and gradients go in `arena`, which is read back; intermediates only backward
   * needs (x̂, 1/σ, the loss's softmax) go in `scratch`, which is not. */
  constructor(private gpu: Gpu, public tape: Tape, private arena: Arena, private scratch: Arena, encoder: GPUCommandEncoder) {
    this.pass = encoder.beginComputePass();
  }

  /** End the pass; returns the buffers to destroy once the work is done. */
  finish(): GPUBuffer[] {
    this.pass.end();
    return [...this.uniforms, ...this.temps];
  }

  dispatches = 0;

  run(name: string, params: number[], bindings: (GTensor | GPUBuffer)[], groups: [number, number?, number?]): void {
    this.dispatches++;
    const u = new Uint32Array(16);
    u.set(params);
    const ub = this.gpu.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(ub.getMappedRange()).set(u);
    ub.unmap();
    this.uniforms.push(ub);
    const pipeline = this.gpu.pipeline(name);
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: ub } }];
    bindings.forEach((b, i) => entries.push({ binding: i + 1, resource: b instanceof GTensor ? b.binding() : { buffer: b } }));
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, this.gpu.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
    this.pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1);
  }

  /** Elementwise dispatch over n items in 256-wide groups, split across y past the x limit. */
  flat(n: number): [number, number] {
    const g = Math.ceil(n / 256);
    return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
  }

  /** Dispatch over n items in 64-wide groups, split across y past the x limit. */
  private flat64(n: number): [number, number] {
    const g = Math.ceil(n / 64);
    return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
  }

  private out(shape: number[], inputs: GTensor[]): GTensor {
    const o = this.arena.alloc(shape);
    o.requiresGrad = this.tape.needs(inputs);
    return o;
  }

  /** The gradient tensor of t, allocated (zeroed by the arena clear) on first use. A head's
   * gradient is allocated with its block's, as a block too. */
  g(t: GTensor): GTensor {
    if (!t.grad && t.block && t.block.members.every((x) => !x.grad)) {
      this.allocBlock(t.shape, t.block.members.length).forEach((g, h) => (t.block!.members[h].grad = g));
    }
    return (t.grad ??= this.arena.alloc(t.shape));
  }

  /** H tensors of one shape, a fixed stride apart: one block. */
  private allocBlock(shape: number[], H: number, inputs: GTensor[] = []): GTensor[] {
    const n = shape.reduce((a, b) => a * b, 1), stride = Math.ceil((n * F) / ALIGN) * ALIGN;
    const base = this.arena.alloc([(H * stride) / F]);
    const block: Block = { members: [], stride };
    const needs = this.tape.needs(inputs);
    for (let h = 0; h < H; h++) {
      const t = new GTensor(base.buf, base.offset + h * stride, shape, needs);
      t.block = block;
      block.members.push(t);
    }
    return block.members;
  }

  /** The block `ts` makes up, in order, if it does (not after an intervention replaced one). */
  private blockOf(ts: (GTensor | null)[]): { all: GTensor; stride: number } | null {
    const b = ts[0]?.block;
    if (!b || b.members.length !== ts.length || ts.some((t, h) => t !== b.members[h])) return null;
    const first = b.members[0];
    return { all: new GTensor(first.buf, first.offset, [(ts.length * b.stride) / F]), stride: b.stride / F };
  }

  private temp(data: Uint32Array | Int32Array | Float32Array): GPUBuffer {
    const b = this.gpu.upload(data);
    this.temps.push(b);
    return b;
  }

  seed(root: GTensor): void {
    this.run("fill", [1, 0, 0, 0, 0, 0, 0, bits(1)], [this.g(root)], [1]);
  }

  /** C (=|+=) alpha · op(A) op(B) (+ bias), with op(A) [m,k] and op(B) [k,n]; with `batch`, that
   * many products of consecutive blocks (one per sequence); with `heads`, that many again, the
   * operands' heads `hs` elements apart (A, B, C). */
  private matmul(A: GTensor, B: GTensor, C: GTensor, m: number, n: number, k: number,
                 o: { ta?: boolean; tb?: boolean; acc?: boolean; bias?: GTensor | null; alpha?: number; batch?: number;
                      heads?: number; hs?: [number, number, number] } = {}): void {
    const per = o.batch ?? 1, H = o.heads ?? 1, [ha, hb, hc] = o.hs ?? [0, 0, 0];
    this.run("matmul", [m, n, k, +!!o.ta, +!!o.tb, +!!o.acc, +!!o.bias, bits(o.alpha ?? 1), m * k, k * n, m * n, per, ha, hb, hc],
             [A, B, C, o.bias ?? this.gpu.dummy], [Math.ceil(n / 16), Math.ceil(m / 16), per * H]);
  }

  /** dst (=|+=) a band of src: rows m, width w, at column s0 of src and d0 of dst. */
  private band(src: GTensor, dst: GTensor, m: number, w: number, sn: number, s0: number, dn: number, d0: number, acc: boolean): void {
    this.run("band", [m, w, sn, s0, dn, d0, +acc], [src, dst], [Math.ceil(w / 64), m]);
  }

  /** dst += src, elementwise over all of src. */
  private accumulate(src: GTensor, dst: GTensor): void {
    this.band(src, dst, 1, src.length, 0, 0, 0, 0, true);
  }

  embed(W: GTensor, ids: number[]): GTensor {
    const T = ids.length, d = W.cols;
    const idBuf = this.temp(new Uint32Array(ids));
    const o = this.out([T, d], [W]);
    this.run("embed", [T, d], [W, idBuf, o], [Math.ceil(d / 64), T]);
    this.tape.push(o, () => this.run("embed_back", [T, d], [o.grad!, idBuf, this.g(W)], [Math.ceil(d / 64)]));
    return o;
  }

  rows(W: GTensor, start: number, n: number, repeat = 1): GTensor {
    const d = W.cols;
    const o = this.out([repeat * n, d], [W]);
    this.band(W, o, repeat, n * d, 0, start * d, n * d, 0, false);
    this.tape.push(o, () => this.run("repeat_back", [repeat, n * d, start * d], [o.grad!, this.g(W)], this.flat64(n * d)));
    return o;
  }

  add(a: GTensor, b: GTensor): GTensor {
    const o = this.out(a.shape, [a, b]);
    this.run("add", [a.length], [a, b, o], this.flat(a.length));
    this.tape.push(o, () => {
      for (const t of [a, b]) if (t.requiresGrad) this.accumulate(o.grad!, this.g(t));
    });
    return o;
  }

  linear(x: GTensor, W: GTensor, b: GTensor | null): GTensor {
    const [m, k] = x.shape, n = W.cols;
    const o = this.out([m, n], b ? [x, W, b] : [x, W]);
    this.matmul(x, W, o, m, n, k, { bias: b });
    this.tape.push(o, () => {
      const g = o.grad!;
      if (x.requiresGrad) this.matmul(g, W, this.g(x), m, k, n, { tb: true, acc: true });
      if (W.requiresGrad) this.matmul(x, g, this.g(W), k, n, m, { ta: true, acc: true });
      if (b?.requiresGrad) this.run("colsum", [m, n], [g, this.g(b)], [Math.ceil(n / 64)]);
    });
    return o;
  }

  linearT(x: GTensor, W: GTensor): GTensor {
    const [m, k] = x.shape, n = W.rows;
    const o = this.out([m, n], [x, W]);
    this.matmul(x, W, o, m, n, k, { tb: true });
    this.tape.push(o, () => {
      const g = o.grad!;
      if (x.requiresGrad) this.matmul(g, W, this.g(x), m, k, n, { acc: true });
      if (W.requiresGrad) this.matmul(g, x, this.g(W), n, k, m, { ta: true, acc: true });
    });
    return o;
  }

  layerNorm(x: GTensor, gamma: GTensor, beta: GTensor): GTensor {
    const [m, d] = x.shape;
    const o = this.out([m, d], [x, gamma, beta]);
    const xhat = this.scratch.alloc([m, d]), rstd = this.scratch.alloc([m]);
    this.run("layernorm", [d], [x, gamma, beta, o, xhat, rstd], [m]);
    this.tape.push(o, () => {
      const g = o.grad!;
      if (x.requiresGrad) this.run("layernorm_back", [d], [g, gamma, xhat, rstd, this.g(x)], [m]);
      if (gamma.requiresGrad) this.run("layernorm_gamma", [m, d], [g, xhat, this.g(gamma)], [Math.ceil(d / 64)]);
      if (beta.requiresGrad) this.run("colsum", [m, d], [g, this.g(beta)], [Math.ceil(d / 64)]);
    });
    return o;
  }

  gelu(x: GTensor): GTensor {
    const o = this.out(x.shape, [x]);
    this.run("gelu", [x.length, 0], [x, this.gpu.dummy, o], this.flat(x.length));
    this.tape.push(o, () => this.run("gelu", [x.length, 1], [x, o.grad!, this.g(x)], this.flat(x.length)));
    return o;
  }

  cols(x: GTensor, c0: number, c1: number): GTensor {
    const [m, n] = x.shape, w = c1 - c0;
    const o = this.out([m, w], [x]);
    this.band(x, o, m, w, n, c0, w, 0, false);
    this.tape.push(o, () => this.band(o.grad!, this.g(x), m, w, w, 0, n, c0, true));
    return o;
  }

  /** Concatenation one part at a time. */
  private concatEach(parts: GTensor[]): GTensor {
    const m = parts[0].rows, n = parts.reduce((s, p) => s + p.cols, 0);
    const o = this.out([m, n], parts);
    let c0 = 0;
    const at = parts.map((p) => {
      this.band(p, o, m, p.cols, p.cols, 0, n, c0, false);
      c0 += p.cols;
      return c0 - p.cols;
    });
    this.tape.push(o, () => parts.forEach((p, i) => {
      if (p.requiresGrad) this.band(o.grad!, this.g(p), m, p.cols, n, at[i], p.cols, 0, true);
    }));
    return o;
  }

  scores(q: GTensor, k: GTensor, scale: number, B = 1): GTensor {
    const d = q.cols, m = q.rows / B, n = k.rows / B;      // per sequence
    const o = this.out([B * m, n], [q, k]);
    this.matmul(q, k, o, m, n, d, { tb: true, alpha: scale, batch: B });
    this.tape.push(o, () => {
      const g = o.grad!;
      if (q.requiresGrad) this.matmul(g, k, this.g(q), m, d, n, { acc: true, alpha: scale, batch: B });
      if (k.requiresGrad) this.matmul(g, q, this.g(k), n, d, m, { ta: true, acc: true, alpha: scale, batch: B });
    });
    return o;
  }

  causalMask(s: GTensor, offset = 0, B = 1): GTensor {
    const [m, n] = s.shape, per = m / B;
    const o = this.out([m, n], [s]);
    this.run("mask", [m, n, offset, 0, NEG_INF_BITS, per], [s, o], [Math.ceil(n / 64), m]);
    this.tape.push(o, () => this.run("mask", [m, n, offset, 1, 0, per], [o.grad!, this.g(s)], [Math.ceil(n / 64), m]));
    return o;
  }

  /** Softmax per attention row: causal, so columns past the diagonal (offset by the KV cache's
   * length) get weight 0. */
  softmax(x: GTensor, offset = 0, B = 1): GTensor {
    const [m, n] = x.shape;
    const o = this.out([m, n], [x]);
    this.run("softmax", [n, 1, offset, m / B], [x, o], [m]);
    this.tape.push(o, () => this.run("softmax_back", [n], [o, o.grad!, this.g(x)], [m]));
    return o;
  }

  concatRows(a: GTensor, b: GTensor): GTensor {
    const o = this.out([a.rows + b.rows, a.cols], [a, b]);
    this.band(a, o, 1, a.length, 0, 0, 0, 0, false);
    this.band(b, o, 1, b.length, 0, 0, 0, a.length, false);
    this.tape.push(o, () => {
      if (a.requiresGrad) this.band(o.grad!, this.g(a), 1, a.length, 0, 0, 0, 0, true);
      if (b.requiresGrad) this.band(o.grad!, this.g(b), 1, b.length, 0, a.length, 0, 0, true);
    });
    return o;
  }

  mix(A: GTensor, V: GTensor, B = 1): GTensor {
    const m = A.rows / B, n = A.cols, d = V.cols;        // per sequence: A [m, n], V [n, d]
    const o = this.out([B * m, d], [A, V]);
    this.matmul(A, V, o, m, d, n, { batch: B });
    this.tape.push(o, () => {
      const g = o.grad!;
      if (A.requiresGrad) this.matmul(g, V, this.g(A), m, n, d, { tb: true, acc: true, batch: B });
      if (V.requiresGrad) this.matmul(A, g, this.g(V), n, d, m, { ta: true, acc: true, batch: B });
    });
    return o;
  }

  crossEntropy(logits: GTensor, targets: number[], denom?: number): GTensor {
    const [m, V] = logits.shape;
    const kept = denom ?? targets.filter((t) => t >= 0).length;
    const tBuf = this.temp(new Int32Array(targets));
    const nll = this.scratch.alloc([m]);
    this.run("nll", [V], [logits, tBuf, nll], [m]);
    const o = this.out([1], [logits]);
    this.run("mean", [m, 0, 0, 0, 0, 0, 0, bits(1 / kept)], [nll, o], [1]);
    this.tape.push(o, () => {
      const probs = this.scratch.alloc([m, V]);
      this.run("softmax", [V, 0, 0, 1], [logits, probs], [m]);
      this.run("nll_back", [V, 0, 0, 0, 0, 0, 0, bits(1 / kept)], [probs, tBuf, o.grad!, this.g(logits)], [Math.ceil(V / 256), m]);
    });
    return o;
  }

  // -- a layer's heads at once ----------------------------------------------------------
  // Each falls back to one dispatch per head when its inputs are not a block (an intervention
  // replaced a head, or the KV cache put cached rows in front).

  /** Run `backward` once if any output got a gradient. */
  private pushAll(outs: GTensor[], backward: () => void): void {
    if (outs.some((o) => o.requiresGrad)) this.tape.push({ shape: [], requiresGrad: true, get grad() { return outs.some((o) => o.grad) || null; } }, backward);
  }

  splitHeads(x: GTensor, H: number): GTensor[] {
    const [m, n] = x.shape, dh = n / H;
    const outs = this.allocBlock([m, dh], H, [x]);
    const { all, stride } = this.blockOf(outs)!;
    this.run("heads", [m, dh, n, stride, 0, 0], [x, all], [Math.ceil(dh / 64), m, H]);
    this.pushAll(outs, () => {
      const gs = this.blockOf(outs.map((o) => o.grad));
      if (gs) this.run("heads", [m, dh, n, gs.stride, 1, 1], [gs.all, this.g(x)], [Math.ceil(dh / 64), m, H]);
      else outs.forEach((o, h) => o.grad && this.band(o.grad, this.g(x), m, dh, dh, 0, n, h * dh, true));
    });
    return outs;
  }

  concatCols(parts: GTensor[]): GTensor {
    const blk = this.blockOf(parts);
    if (!blk) return this.concatEach(parts);
    const H = parts.length, [m, dh] = parts[0].shape, n = H * dh;
    const o = this.out([m, n], parts);
    this.run("heads", [m, dh, n, blk.stride, 1, 0], [blk.all, o], [Math.ceil(dh / 64), m, H]);
    this.tape.push(o, () => {
      parts.forEach((p) => this.g(p));
      const gs = this.blockOf(parts.map((p) => p.grad))!;
      this.run("heads", [m, dh, n, gs.stride, 0, 1], [o.grad!, gs.all], [Math.ceil(dh / 64), m, H]);
    });
    return o;
  }

  scoresHeads(qs: GTensor[], ks: GTensor[], scale: number, B = 1): GTensor[] {
    const bq = this.blockOf(qs), bk = this.blockOf(ks);
    if (!bq || !bk) return qs.map((q, h) => this.scores(q, ks[h], scale, B));
    const H = qs.length, d = qs[0].cols, m = qs[0].rows / B, n = ks[0].rows / B;
    const outs = this.allocBlock([B * m, n], H, [...qs, ...ks]);
    const bo = this.blockOf(outs)!;
    this.matmul(bq.all, bk.all, bo.all, m, n, d, { tb: true, alpha: scale, batch: B, heads: H, hs: [bq.stride, bk.stride, bo.stride] });
    this.pushAll(outs, () => {
      outs.forEach((o) => this.g(o));
      const g = this.blockOf(outs.map((o) => o.grad))!;
      if (qs[0].requiresGrad) {
        qs.forEach((q) => this.g(q));
        const gq = this.blockOf(qs.map((q) => q.grad))!;
        this.matmul(g.all, bk.all, gq.all, m, d, n, { acc: true, alpha: scale, batch: B, heads: H, hs: [g.stride, bk.stride, gq.stride] });
      }
      if (ks[0].requiresGrad) {
        ks.forEach((k) => this.g(k));
        const gk = this.blockOf(ks.map((k) => k.grad))!;
        this.matmul(g.all, bq.all, gk.all, n, d, m, { ta: true, acc: true, alpha: scale, batch: B, heads: H, hs: [g.stride, bq.stride, gk.stride] });
      }
    });
    return outs;
  }

  maskHeads(ss: GTensor[], offset = 0, B = 1): GTensor[] {
    const bs = this.blockOf(ss);
    if (!bs) return ss.map((s) => this.causalMask(s, offset, B));
    const H = ss.length, [m, n] = ss[0].shape, per = m / B;
    const outs = this.allocBlock([m, n], H, ss);
    const bo = this.blockOf(outs)!;
    this.run("mask", [m, n, offset, 0, NEG_INF_BITS, per, bs.stride], [bs.all, bo.all], [Math.ceil(n / 64), m, H]);
    this.pushAll(outs, () => {
      outs.forEach((o) => this.g(o));
      ss.forEach((s) => this.g(s));
      const g = this.blockOf(outs.map((o) => o.grad))!, gs = this.blockOf(ss.map((s) => s.grad))!;
      this.run("mask", [m, n, offset, 1, 0, per, g.stride], [g.all, gs.all], [Math.ceil(n / 64), m, H]);
    });
    return outs;
  }

  softmaxHeads(xs: GTensor[], offset = 0, B = 1): GTensor[] {
    const bx = this.blockOf(xs);
    if (!bx) return xs.map((x) => this.softmax(x, offset, B));
    const H = xs.length, [m, n] = xs[0].shape;
    const outs = this.allocBlock([m, n], H, xs);
    const bo = this.blockOf(outs)!;
    this.run("softmax", [n, 1, offset, m / B, bx.stride], [bx.all, bo.all], [m, H]);
    this.pushAll(outs, () => {
      outs.forEach((o) => this.g(o));
      xs.forEach((x) => this.g(x));
      const g = this.blockOf(outs.map((o) => o.grad))!, gx = this.blockOf(xs.map((x) => x.grad))!;
      this.run("softmax_back", [n, bo.stride], [bo.all, g.all, gx.all], [m, H]);
    });
    return outs;
  }

  mixHeads(As: GTensor[], Vs: GTensor[], B = 1): GTensor[] {
    const ba = this.blockOf(As), bv = this.blockOf(Vs);
    if (!ba || !bv) return As.map((A, h) => this.mix(A, Vs[h], B));
    const H = As.length, m = As[0].rows / B, n = As[0].cols, d = Vs[0].cols;
    const outs = this.allocBlock([B * m, d], H, [...As, ...Vs]);
    const bo = this.blockOf(outs)!;
    this.matmul(ba.all, bv.all, bo.all, m, d, n, { batch: B, heads: H, hs: [ba.stride, bv.stride, bo.stride] });
    this.pushAll(outs, () => {
      outs.forEach((o) => this.g(o));
      const g = this.blockOf(outs.map((o) => o.grad))!;
      if (As[0].requiresGrad) {
        As.forEach((A) => this.g(A));
        const gA = this.blockOf(As.map((A) => A.grad))!;
        this.matmul(g.all, bv.all, gA.all, m, n, d, { tb: true, acc: true, batch: B, heads: H, hs: [g.stride, bv.stride, gA.stride] });
      }
      if (Vs[0].requiresGrad) {
        Vs.forEach((V) => this.g(V));
        const gV = this.blockOf(Vs.map((V) => V.grad))!;
        this.matmul(ba.all, g.all, gV.all, n, d, m, { ta: true, acc: true, batch: B, heads: H, hs: [ba.stride, g.stride, gV.stride] });
      }
    });
    return outs;
  }

  // -- interventions (./interventions) --------------------------------------

  replaceCols(x: GTensor, c0: number, c1: number, mode: "zero" | "mean"): GTensor {
    const [m, n] = x.shape, mean = +(mode === "mean");
    const o = this.out(x.shape, [x]);
    this.band(x, o, m, n, n, 0, n, 0, false);
    this.run("colfill", [m, n, c0, c1, mean, 0], [x, o], [Math.ceil((c1 - c0) / 64)]);
    this.tape.push(o, () => {
      const g = o.grad!, gx = this.g(x);
      if (c0 > 0) this.band(g, gx, m, c0, n, 0, n, 0, true);
      if (c1 < n) this.band(g, gx, m, n - c1, n, c1, n, c1, true);
      if (mean) this.run("colfill", [m, n, c0, c1, 1, 1], [g, gx], [Math.ceil((c1 - c0) / 64)]);
    });
    return o;
  }

  constant(data: Float32Array, shape: number[]): GTensor {
    const o = this.arena.alloc(shape);     // no gradient: nothing upstream can affect it
    const src = new GTensor(this.temp(data), 0, shape);
    this.band(src, o, 1, data.length, 0, 0, 0, 0, false);
    return o;
  }

  overwrite(x: GTensor, index: number[], values: number[]): GTensor {
    const o = this.out(x.shape, [x]);
    const idx = this.temp(new Uint32Array(index)), vals = this.temp(new Float32Array(values));
    const groups: [number] = [Math.ceil(index.length / 64)];
    this.band(x, o, 1, x.length, 0, 0, 0, 0, false);
    this.run("scatter", [index.length, 0], [idx, vals, o], groups);
    this.tape.push(o, () => {
      // The overwritten entries pass no gradient back.
      const g = this.scratch.alloc(x.shape);
      this.band(o.grad!, g, 1, x.length, 0, 0, 0, 0, false);
      this.run("scatter", [index.length, 1], [idx, vals, g], groups);
      this.accumulate(g, this.g(x));
    });
    return o;
  }

  addRow(x: GTensor, vec: ArrayLike<number>, scale: number): GTensor {
    const [m, n] = x.shape;
    const o = this.out(x.shape, [x]);
    this.run("addrow", [m, n, 0, 0, 0, 0, 0, bits(scale)], [x, this.temp(Float32Array.from(vec)), o], [Math.ceil(n / 64), m]);
    this.tape.push(o, () => this.accumulate(o.grad!, this.g(x)));
    return o;
  }

  permuteRows(x: GTensor, perm: number[]): GTensor {
    const [m, n] = x.shape;
    const o = this.out(x.shape, [x]);
    const p = this.temp(new Uint32Array(perm));
    this.run("permute", [m, n, 0], [x, p, o], [Math.ceil(n / 64), m]);
    this.tape.push(o, () => this.run("permute", [m, n, 1], [o.grad!, p, this.g(x)], [Math.ceil(n / 64), m]));
    return o;
  }

  probs(logits: GTensor): GTensor {
    const [m, V] = logits.shape;
    const o = this.arena.alloc([m, V]);   // not on the loss path: no gradient
    this.run("softmax", [V, 0, 0, 1], [logits, o], [m]);
    return o;
  }
}

/** A CPU copy of a GPU tensor's values, from the arena's read-back copy. */
export function view(t: GTensor, host: Float32Array, base: number): Tensor {
  const start = (t.offset - base) / F;
  return new Tensor(host.subarray(start, start + t.length), t.shape);
}

/** What a trace frame needs besides the tensors (see traceHeader in ./frames). */
export interface FrameSpec {
  meta: (id: string) => object;
  step: number;
  text: (tokens: number[]) => string[];
  extra: object;
}

/** The model on the GPU: weights uploaded once, then one submit and one read-back per trace. */
export class GpuModel implements Engine {
  private sets = new Map<WeightSet, Map<string, GTensor>>();
  private arena: Arena;
  private scratch: Arena;
  private staging: GPUBuffer;
  /** The last trace's milliseconds: encoding its dispatches (JS), waiting for the GPU work and
   * the transfer, and copying the read-back out. */
  timings = { encode: 0, wait: 0, copy: 0, dispatches: 0, bytes: 0 };

  get name(): string {
    return `WebGPU (${this.gpu.adapterName})`;
  }

  // Training: weight gradients summed over micro-batches, Adam's moments, and the clipping result.
  private gradArena: Arena | null = null;
  private gradOf = new Map<string, GTensor>();
  private moments = new Map<string, [GPUBuffer, GPUBuffer]>();
  private adamStep = 0;
  private chunk = 16;            // sequences per micro-batch: halved until one fits the arena

  get config(): ModelConfig {
    return this.cfg;
  }

  constructor(private gpu: Gpu, private cfg: ModelConfig) {
    const d = gpu.device;
    const size = Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize, 256 * 2 ** 20);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = new Arena(d.createBuffer({ size, usage: storage }));
    this.scratch = new Arena(d.createBuffer({ size: Math.min(size, 64 * 2 ** 20), usage: storage }));
    this.staging = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  private get W(): Map<string, GTensor> {
    return this.sets.get("current") ?? new Map();
  }

  setWeights(W: Weights, set: WeightSet = "current"): void {
    for (const t of this.sets.get(set)?.values() ?? []) t.buf.destroy();
    this.sets.set(set, new Map([...W].map(([n, w]) => [n, new GTensor(this.gpu.upload(w.data), 0, w.shape)])));
  }

  hasWeights(set: WeightSet): boolean {
    return this.sets.has(set);
  }

  /** A trace, and with `frame` its trace frame too, built around the read-back in one copy: the
   * arena is laid out as the frame's data section. */
  private gradients(): Map<string, GTensor> {
    if (!this.gradArena) {
      const total = [...this.W.values()].reduce((n, w) => n + Math.ceil((w.length * F) / ALIGN) * ALIGN, 0);
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      this.gradArena = new Arena(this.gpu.device.createBuffer({ size: total, usage: storage }));
      for (const [n, w] of this.W) this.gradOf.set(n, this.gradArena.alloc(w.shape));
    }
    return this.gradOf;
  }

  async trainStep(x: number[], y: number[], B: number, s: StepSettings) {
    const device = this.gpu.device, T = x.length / B;
    const grads = this.gradients();
    const kept = y.filter((t) => t >= 0).length;
    const losses = device.createBuffer({ size: B * F, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const temps: GPUBuffer[] = [losses];
    const zero = device.createCommandEncoder();
    zero.clearBuffer(this.gradArena!.buf);
    device.queue.submit([zero.finish()]);

    // Micro-batches, each adding its share of the batch's gradient into `grads`.
    let chunks = 0;
    for (let b0 = 0; b0 < B;) {
      const n = Math.min(this.chunk, B - b0);
      const tape = new Tape();
      for (const [name, w] of this.W) {
        w.requiresGrad = true;
        w.grad = grads.get(name)!;
      }
      this.arena.used = this.scratch.used = 0;
      const encoder = device.createCommandEncoder();
      const op = new GpuOps(this.gpu, tape, this.arena, this.scratch, encoder);
      let loss: GTensor;
      try {
        loss = forward(this.cfg, (name) => this.W.get(name)!, op, x.slice(b0 * T, (b0 + n) * T), y.slice(b0 * T, (b0 + n) * T),
                       (_, t) => t, { B: n, lossDenom: kept })!;
        op.seed(loss);
        tape.backward();
      } catch (e) {
        temps.push(...op.finish());
        if (chunks === 0 && this.chunk > 1 && e instanceof Error && e.message.includes("does not fit")) {
          this.chunk = Math.max(1, this.chunk >> 1);    // nothing submitted yet: retry smaller
          continue;
        }
        throw e;
      }
      temps.push(...op.finish());
      encoder.copyBufferToBuffer(loss.buf, loss.offset, losses, chunks * F, F);
      const clear = device.createCommandEncoder();
      clear.clearBuffer(this.arena.buf, 0, this.arena.used);
      device.queue.submit([clear.finish(), encoder.finish()]);
      b0 += n;
      chunks++;
    }

    // Clip the global norm, then AdamW on every weight, all on the GPU.
    const names = [...this.W.keys()];
    const sq = device.createBuffer({ size: names.length * F, usage: GPUBufferUsage.STORAGE });
    const clip = device.createBuffer({ size: 2 * F, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    temps.push(sq, clip);
    const encoder = device.createCommandEncoder();
    const op = new GpuOps(this.gpu, new Tape(), this.arena, this.scratch, encoder);
    names.forEach((name, i) => op.run("sumsq", [grads.get(name)!.length, i], [grads.get(name)!, sq], [1]));
    op.run("clip", [names.length, 0, 0, 0, 0, 0, 0, bits(s.gradClip)], [sq, clip], [1]);
    this.adamStep++;
    const [b1, b2] = BETAS;
    const stepSize = s.lr / (1 - b1 ** this.adamStep), bc2 = Math.sqrt(1 - b2 ** this.adamStep);
    for (const [name, w] of this.W) {
      const [m, v] = this.moments.get(name) ?? this.moments.set(name, [0, 1].map(() =>
        device.createBuffer({ size: w.length * F, usage: GPUBufferUsage.STORAGE })) as [GPUBuffer, GPUBuffer]).get(name)!;
      const decay = w.shape.length >= 2 ? s.weightDecay : 0;     // matrices decay; biases and gains do not
      op.run("adamw", [w.length, bits(s.lr), bits(decay), bits(b1), bits(b2), bits(ADAM_EPS), bits(stepSize), bits(bc2)],
             [w, grads.get(name)!, m, v, clip], op.flat(w.length));
    }
    temps.push(...op.finish());
    encoder.copyBufferToBuffer(losses, 0, this.staging, 0, chunks * F);
    encoder.copyBufferToBuffer(clip, F, this.staging, chunks * F, F);
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, (chunks + 1) * F);
    const r = new Float32Array(this.staging.getMappedRange(0, (chunks + 1) * F).slice(0));
    this.staging.unmap();
    for (const b of temps) b.destroy();
    for (const w of this.W.values()) {
      w.requiresGrad = false;
      w.grad = null;
    }
    return { loss: r.subarray(0, chunks).reduce((a, b) => a + b, 0), gradNorm: r[chunks] };
  }

  async loss(x: number[], y: number[], B: number): Promise<number> {
    const device = this.gpu.device, T = x.length / B, kept = y.filter((t) => t >= 0).length;
    const losses = device.createBuffer({ size: B * F, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const temps: GPUBuffer[] = [losses];
    for (const w of this.W.values()) {
      w.requiresGrad = false;
      w.grad = null;
    }
    let chunks = 0;
    for (let b0 = 0; b0 < B; b0 += this.chunk, chunks++) {
      const n = Math.min(this.chunk, B - b0);
      const tape = new Tape();
      tape.enabled = false;
      this.arena.used = this.scratch.used = 0;
      const encoder = device.createCommandEncoder();
      const op = new GpuOps(this.gpu, tape, this.arena, this.scratch, encoder);
      const loss = forward(this.cfg, (name) => this.W.get(name)!, op, x.slice(b0 * T, (b0 + n) * T), y.slice(b0 * T, (b0 + n) * T),
                           (_, t) => t, { B: n, lossDenom: kept })!;
      temps.push(...op.finish());
      encoder.copyBufferToBuffer(loss.buf, loss.offset, losses, chunks * F, F);
      device.queue.submit([encoder.finish()]);
    }
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(losses, 0, this.staging, 0, chunks * F);
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, chunks * F);
    const r = new Float32Array(this.staging.getMappedRange(0, chunks * F).slice(0));
    this.staging.unmap();
    for (const b of temps) b.destroy();
    return r.reduce((a, b) => a + b, 0);
  }

  /** Tensors copied back to the host, in one transfer. */
  private async readBack(tensors: Map<string, GTensor>): Promise<Weights> {
    const device = this.gpu.device, encoder = device.createCommandEncoder();
    let at = 0;
    const places = [...tensors].map(([n, t]) => {
      encoder.copyBufferToBuffer(t.buf, t.offset, this.staging, at, t.length * F);
      at += t.length * F;
      return [n, at - t.length * F, t] as const;
    });
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ, 0, at);
    const host = new Float32Array(this.staging.getMappedRange(0, at).slice(0));
    this.staging.unmap();
    return new Map(places.map(([n, start, t]) => [n, new Tensor(host.subarray(start / F, start / F + t.length), t.shape)]));
  }

  readWeights(): Promise<Weights> {
    return this.readBack(this.W);
  }

  readGrads(): Promise<Weights> {
    return this.readBack(this.gradients());
  }

  resetOptimizer(): void {
    for (const [m, v] of this.moments.values()) {
      m.destroy();
      v.destroy();
    }
    this.moments.clear();
    this.adamStep = 0;
  }

  /** A forward pass without gradients; only the tensors `read` returns are copied back. */
  async forward(tokens: number[], opts: ForwardOptions): Promise<Record<string, Float32Array>> {
    const device = this.gpu.device;
    const tape = new Tape();
    tape.enabled = false;
    const W = this.sets.get(opts.weights ?? "current")!;
    for (const w of W.values()) {
      w.requiresGrad = false;
      w.grad = null;
    }
    this.arena.used = this.scratch.used = 0;
    const encoder = device.createCommandEncoder();
    const op = new GpuOps(this.gpu, tape, this.arena, this.scratch, encoder);
    const wanted = Object.entries(runForward(this.cfg, W, op, tokens, opts));
    const temps = op.finish();
    // Pack the wanted tensors into the staging buffer, one after another.
    let at = 0;
    const places = wanted.map(([name, t]) => {
      const bytes = t.length * F;
      encoder.copyBufferToBuffer(t.buf, t.offset, this.staging, at, bytes);
      at += bytes;
      return { name, start: at - bytes, length: t.length };
    });
    device.queue.submit([encoder.finish()]);
    const out: Record<string, Float32Array> = {};
    if (at > 0) {
      await this.staging.mapAsync(GPUMapMode.READ, 0, at);
      const host = new Float32Array(this.staging.getMappedRange(0, at).slice(0));
      this.staging.unmap();
      for (const p of places) out[p.name] = host.subarray(p.start / F, p.start / F + p.length);
    }
    for (const b of temps) b.destroy();
    return out;
  }

  async trace(tokens: number[], opts: TraceOptions, frame: FrameSpec): Promise<TraceResult & { frame: ArrayBuffer }>;
  async trace(tokens: number[], opts?: TraceOptions, frame?: FrameSpec): Promise<TraceResult & { frame?: ArrayBuffer }>;
  async trace(tokens: number[], opts: TraceOptions = {}, frame?: FrameSpec): Promise<TraceResult & { frame?: ArrayBuffer }> {
    const t0 = performance.now();
    const { tokens: idx, targets, hasTarget, withGrad } = prepare(this.cfg, tokens, opts);
    const device = this.gpu.device;
    const tape = new Tape();
    tape.enabled = withGrad;
    // Unwanted, the weights' gradients (most of a trace) go to the training gradient buffer, which
    // is not read back; wanted, to the arena like everything else.
    const outside = withGrad && opts.weightGrads === false ? this.gradients() : null;
    for (const [n, w] of this.W) {
      w.requiresGrad = withGrad;
      w.grad = outside?.get(n) ?? null;
    }
    this.arena.used = this.scratch.used = 0;
    const encoder = device.createCommandEncoder();
    const op = new GpuOps(this.gpu, tape, this.arena, this.scratch, encoder);
    const live = new Map<string, GTensor>();
    const loss = forward(this.cfg, (n) => this.W.get(n)!, op, idx, hasTarget ? targets : null, recorder(op, live, opts.interventions));
    if (withGrad && loss) {
      op.seed(loss);
      tape.backward();
      // A stage that did not affect the loss still has a (zero) gradient in the arena. A weight cut
      // off from the loss has none, as in core/tracer.py.
      for (const x of live.values()) if (x.requiresGrad) op.g(x);
    }
    const temps = op.finish();
    const used = this.arena.used;
    encoder.copyBufferToBuffer(this.arena.buf, 0, this.staging, 0, used);
    // Gradients accumulate into the arena, so clear what this trace uses before it runs.
    const clear = device.createCommandEncoder();
    clear.clearBuffer(this.arena.buf, 0, used);
    if (outside) clear.clearBuffer(this.gradArena!.buf);
    device.queue.submit([clear.finish(), encoder.finish()]);
    const t1 = performance.now();
    await this.staging.mapAsync(GPUMapMode.READ, 0, used);
    const t2 = performance.now();
    const mapped = this.staging.getMappedRange(0, used);

    // Views of every tensor over `host`, the read-back's floats.
    const result = (host: Float32Array): TraceResult => {
      const at = (t: GTensor) => view(t, host, 0);
      const weightGrads = new Map<string, { data: Float32Array; shape: number[] }>();
      if (withGrad && !outside) for (const [n, w] of this.W) if (w.grad) weightGrads.set(n, { data: at(w.grad).data, shape: w.shape });
      return {
        tokens: idx, targets, weightGrads, loss: loss ? at(loss).data[0] : null,
        stages: inStageOrder(this.cfg, live).map(([id, x]) => ({ id, output: at(x), grad: withGrad && x.requiresGrad ? at(x.grad!).data : null })),
      };
    };
    let out: TraceResult & { frame?: ArrayBuffer };
    if (frame) {
      // The header needs only offsets (each view's byteOffset is its place in the arena) and the
      // loss, so it is written before the copy, and the copy goes straight into the frame.
      const header = traceHeader(result(new Float32Array(mapped)), frame.meta, frame.step, frame.text(idx), frame.extra,
                                 (a) => a.byteOffset);
      const { buf, data } = frameAround(header, used);
      data.set(new Uint8Array(mapped));
      out = { ...result(new Float32Array(buf, data.byteOffset, used / F)), frame: buf };
    } else {
      out = result(new Float32Array(mapped.slice(0)));
    }
    this.staging.unmap();
    for (const b of temps) b.destroy();
    this.timings = { encode: t1 - t0, wait: t2 - t1, copy: performance.now() - t2, dispatches: op.dispatches, bytes: used };
    return out;
  }
}
