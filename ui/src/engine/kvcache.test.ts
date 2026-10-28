import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeFrame, decodeTrace, decodeWeights } from "../bridge";
import { CpuEngine } from "./engine";
import { emptyCache, traceCached } from "./kvcache";
import { makeWeights } from "./model";

const load = (name: string) => {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeFrame(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const cfg = { n_layer: 2, n_head: 2, d_model: 32, d_mlp: 128, block_size: 100, vocab_size: 16, tied: false };

// tests/test_ui_fixtures.py: core/kvcache.py's traces for a first pass, two cached steps, and a
// changed prompt, on the tiny fixture model.
describe("KV cache matches core/kvcache.py", () => {
  it("reuses, extends and rebuilds the cache as the core does", async () => {
    const engine = new CpuEngine(cfg);
    engine.setWeights(makeWeights(decodeWeights(load("weights.bin")).tensors));
    const cache = emptyCache();
    for (let i = 0; i < 4; i++) {
      const want = decodeTrace(load(`kvcache/step${i}.bin`));
      const kinds = new Map(want.stages.map((s) => [s.id, s.kind]));
      const { result, info } = await traceCached(engine, cfg, cache, want.tokens, 1, (id) => kinds.get(id) ?? "");
      expect(info, `step ${i}`).toEqual(want.incremental);
      expect(result.stages.map((s) => s.id)).toEqual(want.stages.map((s) => s.id));
      expect(result.loss!).toBeCloseTo(want.loss!, 5);
      for (const s of result.stages) {
        const r = want.byId.get(s.id)!.output;
        expect(s.output.shape, s.id).toEqual(r.shape);
        s.output.data.forEach((v, j) => {
          if (Number.isFinite(r.data[j])) expect(Math.abs(v - r.data[j]), `${s.id}[${j}] at step ${i}`).toBeLessThan(1e-5);
          else expect(v, `${s.id}[${j}] at step ${i}`).toBe(r.data[j]);
        });
      }
    }
  });
});
