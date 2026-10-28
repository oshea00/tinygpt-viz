// Dev-only page (gpu-check.html, run by scripts/gpu-check.mjs): the WebGPU engine against the
// pytest-written fixtures, as model.test.ts checks the CPU engine. Vitest has no WebGPU.

import { decodeFrame, decodeTrace, decodeWeights } from "../bridge";
import type { Trace } from "../types";
import { Tape } from "./autograd";
import { Arena, Gpu, GpuModel, GpuOps } from "./gpu";
import { checkAnalyses } from "./analyses.cases";
import { emptyCache, traceCached } from "./kvcache";
import { checkTraining, TRAIN_CONFIG, trainFixture } from "./train.cases";
import { fetchAssets } from "./local";
import { BPETokenizer } from "./tokenizer";
import { makeWeights, trace, type TraceResult } from "./model";

// The intervention fixtures tests/test_ui_fixtures.py writes, one per kind.
const INTERVENTION_FIXTURES = ["head_zero", "head_mean", "neuron_zero", "neuron_mean", "block_attn", "block_mlp",
                               "attend", "set", "steer", "patch", "shuffle_pos", "combined"];

/** A decoded fixture trace as a TraceResult, to compare like with like. */
function asResult(ref: Trace): TraceResult {
  return {
    tokens: ref.tokens, targets: ref.targets, loss: ref.loss,
    stages: ref.stages.map((s) => ({ id: s.id, output: s.output as unknown as TraceResult["stages"][number]["output"], grad: s.grad?.data ?? null })),
    weightGrads: new Map(Object.entries(ref.weightGrads)),
  };
}

const fixture = async (name: string) => decodeFrame(await (await fetch(`/src/fixtures/${name}`)).arrayBuffer());

/** Largest |got − want| over the tensor, in units of the tolerance: `rel` of the tensor's largest
 * value, plus a floor for gradients that are zero in exact arithmetic. ≤ 1 passes. */
function worst(got: ArrayLike<number>, want: ArrayLike<number>, rel: number): number {
  let scale = 0, diff = 0;
  for (let i = 0; i < want.length; i++) if (Number.isFinite(want[i])) scale = Math.max(scale, Math.abs(want[i]));
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i])) diff = Math.max(diff, Object.is(got[i], want[i]) ? 0 : Infinity);   // NaN and ±∞ must match exactly
    else diff = Math.max(diff, Math.abs(got[i] - want[i]));
  }
  return diff / (rel * scale + 1e-9);
}

/** Every stage, gradient and weight gradient of `t` against `ref`; failures are pushed to `out`. */
function compare(t: TraceResult, ref: TraceResult, label: string, rel: number, out: string[]): void {
  const expect = (cond: boolean, what: string) => cond || out.push(`${label}: ${what}`);
  expect(JSON.stringify(t.stages.map((s) => s.id)) === JSON.stringify(ref.stages.map((s) => s.id)), "stage order");
  expect(Math.abs(t.loss! - ref.loss!) < 1e-5, `loss ${t.loss} vs ${ref.loss}`);
  ref.stages.forEach((r, i) => {
    const s = t.stages[i];
    const w = worst(s.output.data, r.output.data, rel);
    expect(w <= 1, `${s.id} off by ${w.toFixed(1)}× tolerance`);
    expect((s.grad === null) === (r.grad === null), `${s.id} gradient presence`);
    if (s.grad && r.grad) {
      const wg = worst(s.grad, r.grad, rel);
      expect(wg <= 1, `grad ${s.id} off by ${wg.toFixed(1)}× tolerance`);
    }
  });
  expect(JSON.stringify([...t.weightGrads.keys()]) === JSON.stringify([...ref.weightGrads.keys()]), "which weights have gradients");
  for (const [n, g] of t.weightGrads) {
    const wg = worst(g.data, ref.weightGrads.get(n)!.data, rel);
    expect(wg <= 1, `grad ${n} off by ${wg.toFixed(1)}× tolerance`);
  }
}

/** The exported model (core/web_export.py), if there is one: GPU against the CPU engine. */
let timings: GpuModel["timings"] | undefined;
let profile: Record<string, number> | undefined;

/** Median wall time of `f` over a few runs, in ms, after one warm-up. */
async function median(f: () => Promise<unknown>, runs = 5): Promise<number> {
  await f();
  const t: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await f();
    t.push(performance.now() - t0);
  }
  return t.sort((a, b) => a - b)[runs >> 1];
}

/** Where a trace's time goes: the whole trace, a forward pass, the same number of trivial
 * dispatches, and the read-back's transfer alone; and a training micro-batch. */
async function profileGpu(gpu: Gpu, m: GpuModel, ids: number[], cfg: { block_size: number }) {
  const device = gpu.device;
  const spec = { meta: (id: string) => ({ id }), step: 0, text: (t: number[]) => t.map(String), extra: {} };
  const trace = await median(() => m.trace(ids, {}, spec));
  const dispatches = m.timings.dispatches, bytes = m.timings.bytes;
  const forward = await median(() => m.forward(ids, { read: (live, op) => ({ z: op.rows(live.get("logits")!, ids.length - 1, 1) }) }));
  const size = 4 * 2 ** 20;
  const arena = new Arena(device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
  const scratch = new Arena(device.createBuffer({ size, usage: GPUBufferUsage.STORAGE }));
  const cell = arena.alloc([1]);
  const empty = await median(async () => {
    const encoder = device.createCommandEncoder();
    const op = new GpuOps(gpu, new Tape(), arena, scratch, encoder);
    for (let i = 0; i < dispatches; i++) op.run("fill", [1], [cell], [1]);
    const temps = op.finish();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const b of temps) b.destroy();
  });
  const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC });
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const transfer = await median(async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, dst, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    dst.getMappedRange();
    dst.unmap();
  });
  // A training micro-batch of 16 full-length sequences (the default chunk).
  const seq = Array.from({ length: cfg.block_size }, (_, i) => ids[i % ids.length]);
  const x = Array.from({ length: 16 }, () => seq).flat(), y = x.map((_, i) => x[(i + 1) % x.length]);
  const train16 = await median(() => m.trainStep(x, y, 16, { lr: 0, weightDecay: 0, gradClip: 1 }), 3);
  for (const b of [arena.buf, scratch.buf, src, dst]) b.destroy();
  return { traceMs: trace, forwardMs: forward, emptyDispatchesMs: empty, dispatches, transferMs: transfer, transferMB: bytes / 2 ** 20,
           trainStep16x100Ms: train16 };
}

async function checkExported(gpu: Gpu, failures: string[]): Promise<number | undefined> {
  const assets = await fetchAssets(new URL("/model/", location.href).href).catch(() => null);
  if (!assets) return undefined;
  const cfg = assets.meta.config;
  const f = decodeFrame(assets.checkpoint);
  const W = makeWeights(Object.fromEntries(Object.entries(f.header.weights as Record<string, { shape: number[]; offset: number }>)
    .map(([n, r]) => [n, f.tensor(r)])));
  const ids = assets.tokenizer.encode(assets.meta.probe);
  const m = new GpuModel(gpu, cfg);
  m.setWeights(W);
  // As the app asks for it: with the trace frame. The first trace compiles the pipelines.
  const spec = { meta: (id: string) => ({ id }), step: 0, text: (t: number[]) => t.map(String), extra: {} };
  await m.trace(ids, {}, spec);
  const t0 = performance.now();
  const g = await m.trace(ids, {}, spec);
  const ms = performance.now() - t0;
  // The frame must decode to the same values the trace returned.
  const back = decodeTrace(decodeFrame(g.frame!));
  for (const s of g.stages) {
    const r = back.byId.get(s.id)!;
    if (worst(r.output.data, s.output.data, 0) > 0 || (s.grad && worst(r.grad!.data, s.grad, 0) > 0)) failures.push(`frame: ${s.id} differs`);
  }
  timings = m.timings;
  if (new URLSearchParams(location.search).has("profile")) profile = await profileGpu(gpu, m, ids, cfg);
  // Engine against engine, both float32 with sums in different orders: the plan's 1e-4.
  compare(g, trace(cfg, W, ids), cfg.name ?? "exported", 1e-4, failures);
  return ms;
}

async function check(): Promise<{ ok: boolean; adapter?: string; checked?: string[]; failures: string[]; ms?: number; exportedMs?: number; timings?: GpuModel["timings"]; profile?: Record<string, number> }> {
  const gpu = await Gpu.open();
  if (!gpu) return { ok: false, failures: ["no WebGPU adapter"] };
  const errors: string[] = [];
  gpu.device.onuncapturederror = (e) => errors.push(e.error.message);
  const ref = decodeTrace(await fixture("trace.bin"));
  const m = new GpuModel(gpu, { n_layer: 2, n_head: 2, d_model: 32, block_size: 100, vocab_size: 16, tied: false });
  m.setWeights(makeWeights(decodeWeights(await fixture("weights.bin")).tensors));
  const t0 = performance.now();
  const t = await m.trace(ref.tokens, { nextToken: 9 });
  const ms = performance.now() - t0;
  const failures: string[] = [];
  const checked = ["fixture trace", `${INTERVENTION_FIXTURES.length} interventions`];
  const progress = (what: string) => console.log(`gpu-check: ${what} (${failures.length} failures so far)`);
  progress("fixture trace done");
  const want = asResult(ref);
  compare(t, want, "fixture", 1e-5, failures);      // the CPU test's tolerance against core/tracer.py
  for (const name of INTERVENTION_FIXTURES) {
    const r = decodeTrace(await fixture(`interventions/${name}.bin`));
    compare(await m.trace(r.tokens, { nextToken: 9, interventions: r.interventions }), asResult(r), name, 1e-5, failures);
    progress(`intervention ${name} done`);
  }
  // A generation sequence through the KV cache (core/kvcache.py's traces).
  const cache = emptyCache();
  for (let i = 0; i < 4; i++) {
    const r = decodeTrace(await fixture(`kvcache/step${i}.bin`));
    const kinds = new Map(r.stages.map((s) => [s.id, s.kind]));
    const { result, info } = await traceCached(m, { ...m.config, d_mlp: 128 }, cache, r.tokens, 1, (id) => kinds.get(id) ?? "");
    if (JSON.stringify(info) !== JSON.stringify(r.incremental)) failures.push(`kv cache step ${i}: ${JSON.stringify(info)}`);
    compare(result, asResult(r), `kv cache step ${i}`, 1e-5, failures);
  }
  checked.push("KV cache");
  progress("KV cache done");
  // Two steps of core/train.py's Trainer, in micro-batches small enough to be several.
  const fxTrain = trainFixture(await (await fetch("/src/fixtures/train.bin")).arrayBuffer());
  const trainer = new GpuModel(gpu, TRAIN_CONFIG);
  trainer.setWeights(fxTrain.before());
  (trainer as unknown as { chunk: number }).chunk = 2;
  failures.push(...(await checkTraining(trainer, fxTrain)).map((f) => `training: ${f}`));
  checked.push("training");
  progress("training done");
  // The analyses fixture needs the real tokenizer, which the exported model carries.
  const tokFile = await fetch("/model/tokenizer.json");
  if (tokFile.ok) {
    const fx = await (await fetch("/src/fixtures/analyses.json")).json();
    const a = new GpuModel(gpu, fx.config);
    a.setWeights(makeWeights(decodeWeights(await fixture("analyses_weights.bin")).tensors));
    failures.push(...(await checkAnalyses(a, new BPETokenizer(await tokFile.json()), fx)).map((f) => `analyses: ${f}`));
    checked.push("analyses");
    progress("analyses done");
  }
  const exportedMs = await checkExported(gpu, failures);
  if (exportedMs !== undefined) checked.push("exported model against the CPU engine");
  failures.unshift(...errors);       // device errors first: they explain the rest
  return { ok: failures.length === 0, adapter: gpu.adapterName, checked, failures, ms, exportedMs, timings, profile };
}

check().then((r) => {
  (window as any).__result = r;
  document.getElementById("out")!.textContent = JSON.stringify(r, null, 2);
}, (e) => {
  (window as any).__result = { ok: false, failures: [String(e?.stack ?? e)] };
});
