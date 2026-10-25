// Parity: the TypeScript forward pass must reproduce the Python core's trace, stage by stage.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { ModelConfig, Trace, Weights } from "../types";
import { forward, infoFrom } from "./forward";

function load(name: string) {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}
const ref = (load("trace.bin") as { trace: Trace }).trace;
const weights = (load("weights.bin") as { weights: Weights }).weights;
const cfg: ModelConfig = { name: "tiny", n_layer: 2, n_head: 2, d_model: 32, d_mlp: 128, block_size: 100, vocab_size: 16, tied: false };

describe("TypeScript forward pass", () => {
  const ours = forward(ref.tokens, weights.tensors, cfg, infoFrom(ref), ref.text);

  it("produces the same stages in the same order", () => {
    expect(ours.stages.map((s) => s.id)).toEqual(ref.stages.map((s) => s.id));
  });

  for (const s of ref.stages) {
    it(`matches ${s.id}`, () => {
      const mine = ours.byId.get(s.id)!.output;
      expect(mine.shape).toEqual(s.output.shape);
      const scale = Math.max(1, ...Array.from(s.output.data, (v) => (Number.isFinite(v) ? Math.abs(v) : 0)));
      for (let i = 0; i < s.output.data.length; i++) {
        const a = s.output.data[i], b = mine.data[i];
        if (a === -Infinity) expect(b).toBe(-Infinity);
        else if (s.id !== "loss") expect(Math.abs(a - b)).toBeLessThan(1e-5 * scale);
      }
    });
  }

  it("computes the loss on the prompt's own next tokens", () => {
    // The fixture trace also had a target at the last position; recompute ours on the same targets.
    const probs = ours.byId.get("probs")!.output;
    const V = cfg.vocab_size;
    const nll = ref.targets.map((y, t) => -Math.log(probs.data[t * V + y]));
    expect(nll.reduce((a, b) => a + b, 0) / nll.length).toBeCloseTo(ref.loss!, 5);
  });
});
