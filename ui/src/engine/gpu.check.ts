// Dev-only page (gpu-check.html, run by scripts/gpu-check.mjs): the WebGPU engine against the
// pytest-written fixtures, as model.test.ts checks the CPU engine. Vitest has no WebGPU.

import { decodeFrame, decodeTrace, decodeWeights } from "../bridge";
import { Gpu, GpuModel } from "./gpu";
import { fetchAssets } from "./local";
import { makeWeights, trace, type TraceResult } from "./model";

const fixture = async (name: string) => decodeFrame(await (await fetch(`/src/fixtures/${name}`)).arrayBuffer());

/** Largest |got − want| over the tensor, in units of the tolerance: `rel` of the tensor's largest
 * value, plus a floor for gradients that are zero in exact arithmetic. ≤ 1 passes. */
function worst(got: ArrayLike<number>, want: ArrayLike<number>, rel: number): number {
  let scale = 0, diff = 0;
  for (let i = 0; i < want.length; i++) if (Number.isFinite(want[i])) scale = Math.max(scale, Math.abs(want[i]));
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i])) diff = Math.max(diff, got[i] === want[i] ? 0 : Infinity);
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
  for (const [n, g] of t.weightGrads) {
    const wg = worst(g.data, ref.weightGrads.get(n)!.data, rel);
    expect(wg <= 1, `grad ${n} off by ${wg.toFixed(1)}× tolerance`);
  }
}

/** The exported model (core/web_export.py), if there is one: GPU against the CPU engine. */
let timings: GpuModel["timings"] | undefined;

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
  await m.trace(ids);                      // the first trace compiles the pipelines
  const t0 = performance.now();
  const g = await m.trace(ids);
  const ms = performance.now() - t0;
  timings = m.timings;
  // Engine against engine, both float32 with sums in different orders: the plan's 1e-4.
  compare(g, trace(cfg, W, ids), cfg.name ?? "exported", 1e-4, failures);
  return ms;
}

async function check(): Promise<{ ok: boolean; adapter?: string; failures: string[]; ms?: number; exportedMs?: number; timings?: GpuModel["timings"] }> {
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
  // The fixture trace as a TraceResult, to compare like with like.
  const want: TraceResult = {
    tokens: ref.tokens, targets: ref.targets, loss: ref.loss,
    stages: ref.stages.map((s) => ({ id: s.id, output: { ...s.output, rows: 0, cols: 0 } as any, grad: s.grad?.data ?? null })),
    weightGrads: new Map(Object.entries(ref.weightGrads)),
  };
  compare(t, want, "fixture", 1e-5, failures);      // the CPU test's tolerance against core/tracer.py
  const exportedMs = await checkExported(gpu, failures);
  failures.push(...errors);
  return { ok: failures.length === 0, adapter: gpu.adapterName, failures, ms, exportedMs, timings };
}

check().then((r) => {
  (window as any).__result = r;
  document.getElementById("out")!.textContent = JSON.stringify(r, null, 2);
}, (e) => {
  (window as any).__result = { ok: false, failures: [String(e?.stack ?? e)] };
});
