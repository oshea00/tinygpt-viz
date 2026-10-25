import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Trace, Weights } from "../types";
import { gradientHighway, lensRow, logitLens, snapshots, writes } from "./residual";
import { at, dims2, norm, row } from "./tensor";

function load(name: string) {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}
const trace = (load("trace.bin") as { trace: Trace }).trace;
const weights = (load("weights.bin") as { weights: Weights }).weights;

describe("residual stream", () => {
  it("has 1 + 2·layers snapshots and one write per sublayer", () => {
    expect(snapshots(trace).map((s) => s.id)).toEqual(["embed.sum", "L0.resid_attn", "L0.resid_mlp", "L1.resid_attn", "L1.resid_mlp"]);
    expect(writes(trace).map((w) => w.delta)).toEqual(["L0.attn.out", "L0.mlp.down", "L1.attn.out", "L1.mlp.down"]);
  });

  it("final stream = x₀ + every sublayer's write", () => {
    const [T, d] = dims2(trace.byId.get("embed.sum")!.output);
    for (const [t, i] of [[0, 0], [T - 1, d - 1], [5, 7]]) {
      const sum = at(trace.byId.get("embed.sum")!.output, t, i) + writes(trace).reduce((s, w) => s + at(trace.byId.get(w.delta)!.output, t, i), 0);
      expect(sum).toBeCloseTo(at(trace.byId.get("L1.resid_mlp")!.output, t, i), 5);
    }
  });

  it("splits each gradient into the straight-through part and the branch", () => {
    const flow = gradientHighway(trace, 3)!;
    expect(flow.map((f) => f.label)).toEqual(["L1 mlp", "L1 attn", "L0 mlp", "L0 attn"]);
    expect(flow[0].through).toBeCloseTo(norm(row(trace.byId.get("L1.resid_mlp")!.grad!, 3)), 6);
    expect(flow[3].total).toBeCloseTo(norm(row(trace.byId.get("embed.sum")!.grad!, 3)), 6);
  });
});

describe("logit lens", () => {
  const lens = logitLens(trace, weights);

  it("at the last snapshot reproduces the model's own probabilities", () => {
    const probs = trace.byId.get("probs")!.output;
    const last = lens.snaps.length - 1;
    for (let t = 0; t < lens.T; t++) {
      const r = lensRow(lens, last, t);
      for (let k = 0; k < lens.V; k++) expect(r[k]).toBeCloseTo(at(probs, t, k), 5);
    }
  });

  it("gives a distribution at every snapshot", () => {
    for (let s = 0; s < lens.snaps.length; s++) {
      expect(lensRow(lens, s, 4).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    }
  });
});
