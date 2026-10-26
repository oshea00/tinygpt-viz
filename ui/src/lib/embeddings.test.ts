import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Trace, Weights } from "../types";
import { dimSizes, inResidualBasis, tableMap, tokenKind, trajectories } from "./embeddings";
import { dims2 } from "./tensor";

function load(name: string) {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}
const trace = (load("trace.bin") as { trace: Trace }).trace;
const weights = (load("weights.bin") as { weights: Weights }).weights;

describe("residual basis", () => {
  it("holds for embeddings, snapshots, layer norms and sublayer writes only", () => {
    for (const id of ["embed.tok", "embed.pos", "embed.sum", "L0.ln1", "L1.ln2", "L0.attn.out", "L1.resid_attn", "L0.mlp.down", "L1.resid_mlp", "ln_f"]) {
      expect(inResidualBasis(id), id).toBe(true);
    }
    for (const id of ["L0.q", "L1.k", "L0.v", "L0.attn.H1.weights", "L1.attn.concat", "L0.mlp.up", "L1.mlp.gelu", "logits", "probs", "loss"]) {
      expect(inResidualBasis(id), id).toBe(false);
    }
  });

  it("matches every d_model-wide stage in the fixture except the concatenated heads", () => {
    const d = dims2(trace.byId.get("embed.sum")!.output)[1];
    for (const s of trace.stages) {
      if (s.output.shape.length !== 2 || s.output.shape[1] !== d || s.id.includes(".attn.H")) continue;
      expect(inResidualBasis(s.id), s.id).toBe(!/\.(q|k|v|attn\.concat)$/.test(s.id));
    }
  });
});

describe("table maps", () => {
  it("project every row, with the explained variance in order", () => {
    for (const name of ["wte", "wpe"]) {
      const t = weights.tensors[name];
      const m = tableMap(t);
      expect(m.points.length).toBe(t.shape[0]);
      expect(m.explained[0]).toBeGreaterThan(0);
      expect(m.explained[0]).toBeGreaterThanOrEqual(m.explained[1] - 1e-9);
      expect(m.explained[0] + m.explained[1]).toBeLessThanOrEqual(1 + 1e-9);
    }
  });
});

describe("trajectories", () => {
  it("give one path per position, one point per snapshot", () => {
    const tr = trajectories(trace);
    expect(tr.paths.length).toBe(trace.tokens.length);
    expect(tr.labels).toEqual(["x₀ embed", "L0 +attn", "L0 +mlp", "L1 +attn", "L1 +mlp"]);
    for (const p of tr.paths) expect(p.length).toBe(5);
  });

  it("keep normalised points within the unit circle", () => {
    // Unit vectors projected onto unit directions cannot leave the unit disc.
    for (const p of trajectories(trace, true).paths.flat()) expect(Math.hypot(...p)).toBeLessThanOrEqual(1 + 1e-6);
  });

  it("never stretch a raw step: a projection onto orthonormal axes only shortens distances", () => {
    const raw = trajectories(trace, false);
    const X0 = trace.byId.get("embed.sum")!.output, X1 = trace.byId.get("L1.resid_mlp")!.output;
    const d = dims2(X0)[1];
    for (let t = 0; t < trace.tokens.length; t++) {
      let full = 0;
      for (let k = 0; k < d; k++) full += (X1.data[t * d + k] - X0.data[t * d + k]) ** 2;
      const [a, b] = [raw.paths[t][0], raw.paths[t][4]];
      expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeLessThanOrEqual(Math.sqrt(full) + 1e-6);
    }
  });
});

describe("dimension sizes and token kinds", () => {
  it("average |x| per dimension", () => {
    const s = dimSizes(trace, "L1.resid_mlp");
    const X = trace.byId.get("L1.resid_mlp")!.output;
    const [T, d] = dims2(X);
    expect(s.length).toBe(d);
    let first = 0;
    for (let t = 0; t < T; t++) first += Math.abs(X.data[t * d]) / T;
    expect(s[0]).toBeCloseTo(first, 6);
  });

  it("classify tokens by their display text", () => {
    expect(tokenKind("␣dog")).toBe("word");
    expect(tokenKind("ing")).toBe("word piece");
    expect(tokenKind("␣Lily")).toBe("Capitalised");
    expect(tokenKind("The")).toBe("Capitalised");
    expect(tokenKind("␣12")).toBe("digits");
    expect(tokenKind(".")).toBe("punctuation");
    expect(tokenKind("⏎⏎")).toBe("punctuation");
    expect(tokenKind("�")).toBe("byte");
    expect(tokenKind("␄")).toBe("end of story");
  });
});
