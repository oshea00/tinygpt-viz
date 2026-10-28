// The GPT's operations as WebGPU compute shaders: the same OpSet as the CPU Ops in ./autograd,
// so ./model's forward() runs unchanged. Every dispatch of a trace (forward and backward) goes
// into one compute pass; stage outputs and gradients live in one arena, read back in one copy.

import { Tape, Tensor, type Node, type OpSet } from "./autograd";
import { frameAround, traceHeader } from "./frames";
import { forward, prepare, type ModelConfig, type TraceOptions, type TraceResult, type Weights } from "./model";

const F = 4;                    // bytes per float
const ALIGN = 256;              // minStorageBufferOffsetAlignment, the most any device asks

/** A tensor on the GPU: a range of a buffer. */
export class GTensor implements Node {
  grad: GTensor | null = null;

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

// Every kernel reads its sizes from P: eight u32 slots (floats are bit-cast).
const HEADER = /* wgsl */ `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
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
  matmul: `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;
fn a_at(r: u32, c: u32) -> f32 {
  if (r >= p.a || c >= p.c) { return 0.0; }
  if (p.d == 1u) { return A[c * p.a + r]; }
  return A[r * p.c + c];
}
fn b_at(r: u32, c: u32) -> f32 {
  if (r >= p.c || c >= p.b) { return 0.0; }
  if (p.e == 1u) { return B[c * p.c + r]; }
  return B[r * p.b + c];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.y * 16u + lid.y;
  let col = wg.x * 16u + lid.x;
  var s = 0.0;
  for (var t = 0u; t < (p.c + 15u) / 16u; t++) {
    As[lid.y][lid.x] = a_at(row, t * 16u + lid.x);
    Bs[lid.y][lid.x] = b_at(t * 16u + lid.y, col);
    workgroupBarrier();
    for (var i = 0u; i < 16u; i++) { s += As[lid.y][i] * Bs[i][lid.x]; }
    workgroupBarrier();
  }
  if (row < p.a && col < p.b) {
    var v = bitcast<f32>(p.h) * s;
    if (p.g == 1u) { v += bias[col]; }
    let i = row * p.b + col;
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

  // The causal mask: a=m b=n c=offset d=backward e=the bits of -∞. Forward writes -∞ above the
  // diagonal; backward passes the gradient through below it.
  mask: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let j = id.x;
  let i = id.y;
  if (i >= p.a || j >= p.b) { return; }
  let k = i * p.b + j;
  let keep = j <= i + p.c;
  if (p.d == 0u) {
    if (keep) { out[k] = x[k]; } else { out[k] = bitcast<f32>(p.e); }
  } else if (keep) {
    out[k] += x[k];
  }
}`,

  // Softmax per row, one workgroup per row. a=n b=causal c=offset: with causal set, columns past
  // the diagonal are left out by index rather than through -∞ arithmetic.
  softmax: `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * p.a;
  var n = p.a;
  if (p.b == 1u) { n = min(p.a, wg.x + p.c + 1u); }
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

  // ∂x += A ⊙ (g − Σ_j g_j A_j), one workgroup per row. a=n.
  softmax_back: `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> gx: array<f32>;
${REDUCE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wg.x * p.a;
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

  private run(name: string, params: number[], bindings: (GTensor | GPUBuffer)[], groups: [number, number?, number?]): void {
    this.dispatches++;
    const u = new Uint32Array(8);
    u.set(params);
    const ub = this.gpu.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
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
  private flat(n: number): [number, number] {
    const g = Math.ceil(n / 256);
    return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
  }

  private out(shape: number[], inputs: GTensor[]): GTensor {
    const o = this.arena.alloc(shape);
    o.requiresGrad = this.tape.needs(inputs);
    return o;
  }

  /** The gradient tensor of t, allocated (zeroed by the arena clear) on first use. */
  g(t: GTensor): GTensor {
    return (t.grad ??= this.arena.alloc(t.shape));
  }

  private temp(data: Uint32Array | Int32Array): GPUBuffer {
    const b = this.gpu.upload(data);
    this.temps.push(b);
    return b;
  }

  seed(root: GTensor): void {
    this.run("fill", [1, 0, 0, 0, 0, 0, 0, bits(1)], [this.g(root)], [1]);
  }

  /** C (=|+=) alpha · op(A) op(B) (+ bias), with op(A) [m,k] and op(B) [k,n]. */
  private matmul(A: GTensor, B: GTensor, C: GTensor, m: number, n: number, k: number,
                 o: { ta?: boolean; tb?: boolean; acc?: boolean; bias?: GTensor | null; alpha?: number } = {}): void {
    this.run("matmul", [m, n, k, +!!o.ta, +!!o.tb, +!!o.acc, +!!o.bias, bits(o.alpha ?? 1)],
             [A, B, C, o.bias ?? this.gpu.dummy], [Math.ceil(n / 16), Math.ceil(m / 16)]);
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

  rows(W: GTensor, start: number, n: number): GTensor {
    const d = W.cols;
    const o = this.out([n, d], [W]);
    this.band(W, o, 1, n * d, 0, start * d, 0, 0, false);
    this.tape.push(o, () => this.band(o.grad!, this.g(W), 1, n * d, 0, 0, 0, start * d, true));
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

  concatCols(parts: GTensor[]): GTensor {
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

  scores(q: GTensor, k: GTensor, scale: number): GTensor {
    const [m, d] = q.shape, n = k.rows;
    const o = this.out([m, n], [q, k]);
    this.matmul(q, k, o, m, n, d, { tb: true, alpha: scale });
    this.tape.push(o, () => {
      const g = o.grad!;
      if (q.requiresGrad) this.matmul(g, k, this.g(q), m, d, n, { acc: true, alpha: scale });
      if (k.requiresGrad) this.matmul(g, q, this.g(k), n, d, m, { ta: true, acc: true, alpha: scale });
    });
    return o;
  }

  causalMask(s: GTensor, offset = 0): GTensor {
    const [m, n] = s.shape;
    const o = this.out([m, n], [s]);
    this.run("mask", [m, n, offset, 0, NEG_INF_BITS], [s, o], [Math.ceil(n / 64), m]);
    this.tape.push(o, () => this.run("mask", [m, n, offset, 1], [o.grad!, this.g(s)], [Math.ceil(n / 64), m]));
    return o;
  }

  /** Softmax per row. Attention rows are causal: columns past the diagonal get weight 0. */
  softmax(x: GTensor, causal = true, offset = 0): GTensor {
    const [m, n] = x.shape;
    const o = this.out([m, n], [x]);
    this.run("softmax", [n, +causal, offset], [x, o], [m]);
    this.tape.push(o, () => this.run("softmax_back", [n], [o, o.grad!, this.g(x)], [m]));
    return o;
  }

  mix(A: GTensor, V: GTensor): GTensor {
    const [m, n] = A.shape, d = V.cols;
    const o = this.out([m, d], [A, V]);
    this.matmul(A, V, o, m, d, n);
    this.tape.push(o, () => {
      const g = o.grad!;
      if (A.requiresGrad) this.matmul(g, V, this.g(A), m, n, d, { tb: true, acc: true });
      if (V.requiresGrad) this.matmul(A, g, this.g(V), n, d, m, { ta: true, acc: true });
    });
    return o;
  }

  crossEntropy(logits: GTensor, targets: number[]): GTensor {
    const [m, V] = logits.shape;
    const kept = targets.filter((t) => t >= 0).length;
    const tBuf = this.temp(new Int32Array(targets));
    const nll = this.scratch.alloc([m]);
    this.run("nll", [V], [logits, tBuf, nll], [m]);
    const o = this.out([1], [logits]);
    this.run("mean", [m, 0, 0, 0, 0, 0, 0, bits(1 / kept)], [nll, o], [1]);
    this.tape.push(o, () => {
      const probs = this.scratch.alloc([m, V]);
      this.run("softmax", [V, 0, 0], [logits, probs], [m]);
      this.run("nll_back", [V, 0, 0, 0, 0, 0, 0, bits(1 / kept)], [probs, tBuf, o.grad!, this.g(logits)], [Math.ceil(V / 256), m]);
    });
    return o;
  }

  probs(logits: GTensor): GTensor {
    const [m, V] = logits.shape;
    const o = this.arena.alloc([m, V]);   // not on the loss path: no gradient
    this.run("softmax", [V, 0, 0], [logits, o], [m]);
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
export class GpuModel {
  private W = new Map<string, GTensor>();
  private arena: Arena;
  private scratch: Arena;
  private staging: GPUBuffer;
  /** The last trace's milliseconds: encoding its dispatches (JS), waiting for the GPU work and
   * the transfer, and copying the read-back out. */
  timings = { encode: 0, wait: 0, copy: 0, dispatches: 0, bytes: 0 };

  constructor(private gpu: Gpu, private cfg: ModelConfig) {
    const d = gpu.device;
    const size = Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize, 256 * 2 ** 20);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.arena = new Arena(d.createBuffer({ size, usage: storage }));
    this.scratch = new Arena(d.createBuffer({ size: Math.min(size, 64 * 2 ** 20), usage: storage }));
    this.staging = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  setWeights(W: Weights): void {
    for (const t of this.W.values()) t.buf.destroy();
    this.W = new Map([...W].map(([n, w]) => [n, new GTensor(this.gpu.upload(w.data), 0, w.shape)]));
  }

  /** A trace, and with `frame` its trace frame too, built around the read-back in one copy: the
   * arena is laid out as the frame's data section. */
  async trace(tokens: number[], opts: TraceOptions = {}, frame?: FrameSpec): Promise<TraceResult & { frame?: ArrayBuffer }> {
    const t0 = performance.now();
    const { tokens: idx, targets, hasTarget, withGrad } = prepare(this.cfg, tokens, opts);
    const device = this.gpu.device;
    const tape = new Tape();
    tape.enabled = withGrad;
    for (const w of this.W.values()) {
      w.requiresGrad = withGrad;
      w.grad = null;
    }
    this.arena.used = this.scratch.used = 0;
    const encoder = device.createCommandEncoder();
    const op = new GpuOps(this.gpu, tape, this.arena, this.scratch, encoder);
    const live = new Map<string, GTensor>();
    const loss = forward(this.cfg, (n) => this.W.get(n)!, op, idx, hasTarget ? targets : null, (id, x) => (live.set(id, x), x));
    if (withGrad && loss) {
      op.seed(loss);
      tape.backward();
      // A stage or weight that did not affect the loss still has a (zero) gradient in the arena.
      for (const x of [...live.values(), ...this.W.values()]) if (x.requiresGrad) op.g(x);
    }
    const temps = op.finish();
    const used = this.arena.used;
    encoder.copyBufferToBuffer(this.arena.buf, 0, this.staging, 0, used);
    // Gradients accumulate into the arena, so clear what this trace uses before it runs.
    const clear = device.createCommandEncoder();
    clear.clearBuffer(this.arena.buf, 0, used);
    device.queue.submit([clear.finish(), encoder.finish()]);
    const t1 = performance.now();
    await this.staging.mapAsync(GPUMapMode.READ, 0, used);
    const t2 = performance.now();
    const mapped = this.staging.getMappedRange(0, used);

    // Views of every tensor over `host`, the read-back's floats.
    const result = (host: Float32Array): TraceResult => {
      const at = (t: GTensor) => view(t, host, 0);
      const weightGrads = new Map<string, { data: Float32Array; shape: number[] }>();
      if (withGrad) for (const [n, w] of this.W) weightGrads.set(n, { data: at(w.grad!).data, shape: w.shape });
      return {
        tokens: idx, targets, weightGrads, loss: loss ? at(loss).data[0] : null,
        stages: [...live].map(([id, x]) => ({ id, output: at(x), grad: withGrad && x.requiresGrad ? at(x.grad!).data : null })),
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
