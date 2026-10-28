import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeFrame, decodeTrace, decodeWeights } from "../bridge";
import { encodeTrace } from "./frames";
import { makeWeights, trace } from "./model";

// The pytest-written fixtures: a tiny random GPT's weights and its trace (core/tracer.py).
const load = (name: string) => {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeFrame(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const ref = decodeTrace(load("trace.bin"));
const W = makeWeights(decodeWeights(load("weights.bin")).tensors);
const cfg = { n_layer: 2, n_head: 2, d_model: 32, block_size: 100, vocab_size: 16, tied: false };
const t = trace(cfg, W, [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8], { nextToken: 9 });

/** Close to the reference relative to the tensor's largest value (float32 sums in another order). */
function expectClose(got: ArrayLike<number>, want: ArrayLike<number>, what: string) {
  expect(got.length, what).toBe(want.length);
  let scale = 0;
  for (let i = 0; i < want.length; i++) if (Number.isFinite(want[i])) scale = Math.max(scale, Math.abs(want[i]));
  let worst = 0;
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i])) {
      expect(got[i], `${what}[${i}]`).toBe(want[i]);
      continue;
    }
    worst = Math.max(worst, Math.abs(got[i] - want[i]));
  }
  // A floor for gradients that are zero in exact arithmetic (the key bias: softmax ignores shifts).
  expect(worst, what).toBeLessThan(1e-5 * scale + 1e-9);
}

describe("engine trace matches core/tracer.py", () => {
  it("visits the same stages in the same order", () => {
    expect(t.stages.map((s) => s.id)).toEqual(ref.stages.map((s) => s.id));
    expect(t.targets).toEqual(ref.targets);
  });

  it("computes every stage output", () => {
    for (const s of t.stages) {
      const r = ref.byId.get(s.id)!;
      expect(s.output.shape, s.id).toEqual(r.output.shape);
      expectClose(s.output.data, r.output.data, s.id);
    }
    expect(t.loss!).toBeCloseTo(ref.loss!, 5);
  });

  it("computes every stage gradient", () => {
    for (const s of t.stages) {
      const r = ref.byId.get(s.id)!;
      expect(s.grad === null, s.id).toBe(r.grad === undefined);
      if (s.grad) expectClose(s.grad, r.grad!.data, `grad ${s.id}`);
    }
  });

  it("computes every weight gradient", () => {
    expect([...t.weightGrads.keys()]).toEqual(Object.keys(ref.weightGrads));
    for (const [n, g] of t.weightGrads) expectClose(g.data, ref.weightGrads[n].data, `grad ${n}`);
  });
});

describe("encodeTrace", () => {
  it("writes a frame the UI decodes back to the same trace", () => {
    const meta = (id: string) => {
      const { output, grad, ...m } = ref.byId.get(id)!;
      return m;
    };
    const back = decodeTrace(decodeFrame(encodeTrace(t, meta, ref.step, ref.text, { prompt: "fixture", reason: "run" })));
    expect(back.step).toBe(ref.step);
    expect(back.prompt).toBe("fixture");
    expect(back.tokens).toEqual(ref.tokens);
    for (const s of back.stages) {
      const r = ref.byId.get(s.id)!;
      expect({ ...s, output: null, grad: null }).toEqual({ ...r, output: null, grad: null });
      expectClose(s.output.data, r.output.data, s.id);
    }
    expectClose(back.weightGrads["L1.W_q"].data, ref.weightGrads["L1.W_q"].data, "L1.W_q");
  });
});

describe("interventions match core/interventions.py", () => {
  const dir = new URL("../fixtures/interventions/", import.meta.url);
  for (const file of readdirSync(dir)) {
    it(file.replace(".bin", ""), () => {
      const want = decodeTrace(load(`interventions/${file}`));
      const got = trace(cfg, W, want.tokens, { nextToken: 9, interventions: want.interventions });
      expect(got.stages.map((s) => s.id)).toEqual(want.stages.map((s) => s.id));
      expect(got.loss!).toBeCloseTo(want.loss!, 5);
      for (const s of got.stages) {
        const r = want.byId.get(s.id)!;
        expectClose(s.output.data, r.output.data, s.id);
        expect(s.grad === null, `${s.id} has a gradient`).toBe(r.grad === undefined);
        if (s.grad) expectClose(s.grad, r.grad!.data, `grad ${s.id}`);
      }
      expect([...got.weightGrads.keys()]).toEqual(Object.keys(want.weightGrads));
      for (const [n, g] of got.weightGrads) expectClose(g.data, want.weightGrads[n].data, `grad ${n}`);
    });
  }
});
