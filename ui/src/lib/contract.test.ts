// Decodes frames written by tests/test_ui_fixtures.py (run `uv run pytest` to refresh them).

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary, decodeFrame } from "../bridge";
import type { Trace, Weights } from "../types";
import { pipelineRows, stepSequence } from "./pipeline";
import { gradProvenance, provenance } from "./provenance";
import { at, dims2 } from "./tensor";

function load(name: string): ArrayBuffer {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

const traceMsg = decodeBinary(load("trace.bin"));
const weightsMsg = decodeBinary(load("weights.bin"));
const trace = (traceMsg as { trace: Trace }).trace;
const weights = (weightsMsg as { weights: Weights }).weights;

describe("frame decoding", () => {
  it("reads the trace header and tensors", () => {
    expect(traceMsg.type).toBe("trace");
    expect(trace.step).toBe(42);
    expect(trace.tokens.length).toBe(12);
    expect(trace.stages[0].id).toBe("embed.tok");
    expect(trace.stages.at(-1)!.id).toBe("loss");
    expect(trace.byId.get("L1.attn.H1.weights")!.output.shape).toEqual([12, 12]);
    expect(trace.byId.get("logits")!.grad!.shape).toEqual([12, 16]);
  });

  it("keeps -inf in masked cells", () => {
    const m = trace.byId.get("L0.attn.H0.masked")!.output;
    expect(at(m, 0, 5)).toBe(-Infinity);
    expect(Number.isFinite(at(m, 5, 0))).toBe(true);
  });

  it("reads weights", () => {
    expect(weights.version).toBe(1);
    expect(weights.tensors["L0.W_up"].shape).toEqual([32, 128]);
  });
});

describe("provenance reproduces stored values", () => {
  for (const stage of trace.stages) {
    it(stage.id, () => {
      const [rows, cols] = dims2(stage.output);
      for (const [r, c] of [[0, 0], [rows - 1, cols - 1], [Math.floor(rows / 2), Math.floor(cols / 3)]]) {
        const p = provenance(trace, weights, stage, r, c)!;
        expect(p).not.toBeNull();
        if (p.stored === -Infinity) expect(p.value).toBe(-Infinity);
        else expect(p.value).toBeCloseTo(p.stored, 4);
      }
    });
  }

  it("residual-stream gradients sum their consumers", () => {
    // Fixture trace has a target at every position, so every stage has a gradient.
    const s = trace.byId.get("L0.resid_attn")!;
    const p = gradProvenance(trace, s, 3, 7)!;
    expect(p.lines.some(([l]) => String(l).includes("L0.resid_mlp"))).toBe(true);
  });
});

describe("pipeline layout", () => {
  it("puts residual adds on the spine and groups heads", () => {
    const rows = pipelineRows(trace);
    expect(rows.find((r) => r.key === "L0.resid_attn")!.lane).toBe("spine");
    expect(rows.find((r) => r.key === "L1.H.weights")!.stages.length).toBe(2);
    expect(rows.filter((r) => r.segStart).length).toBe(1 + 2 * 2);
    expect(stepSequence(trace).at(-1)!.phase).toBe("backward");
  });
});

describe("with an adapter", () => {
  // tests/test_ui_fixtures.py: the trace fixture's model with an adapter at half strength.
  const adapted = (decodeBinary(load("lora/trace.bin")) as { trace: Trace }).trace;
  // The train frame's "before" group holds the A and B the trace used.
  const f = decodeFrame(load("lora/train.bin"));
  const all: Weights = { ...weights, tensors: { ...weights.tensors } };
  for (const [name, ref] of Object.entries(f.header.before as Record<string, { shape: number[]; offset: number }>)) {
    all.tensors[name] = f.tensor(ref);
  }

  it("reads which adapter a trace ran with", () => {
    expect(adapted.adapter).toEqual({ name: "fixture", scale: 0.5 });
  });

  for (const id of ["L0.q.lora", "L0.q", "L1.v.lora", "L1.v", "L0.mlp.down.lora", "L0.mlp.down"]) {
    it(`provenance of ${id}`, () => {
      const stage = adapted.byId.get(id)!;
      const [rows, cols] = dims2(stage.output);
      for (const [r, c] of [[0, 0], [rows - 1, cols - 1], [5, 3]]) {
        const p = provenance(adapted, all, stage, r, c)!;
        expect(p.value).toBeCloseTo(p.stored, 4);
      }
    });
  }

  it("gives the adapter's terms their own pipeline rows", () => {
    const rows = pipelineRows(adapted);
    const qkv = rows.find((r) => r.key === "L0.qkv.lora")!;
    expect(qkv.stages.map((s) => s.id)).toEqual(["L0.q.lora", "L0.v.lora"]);
    expect(rows.indexOf(qkv) + 1).toBe(rows.findIndex((r) => r.key === "L0.qkv"));
    expect(rows.find((r) => r.key === "L1.mlp.down.lora")!.label).toBe("adapter on MLP down");
  });
});
