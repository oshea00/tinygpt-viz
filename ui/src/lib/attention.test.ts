import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Tensor, Trace } from "../types";
import { headOf, headPattern, headStats, qkRows } from "./attention";

function matrix(T: number, f: (r: number, j: number) => number): Tensor {
  const data = new Float32Array(T * T);
  for (let r = 0; r < T; r++) for (let j = 0; j < T; j++) data[r * T + j] = f(r, j);
  return { shape: [T, T], data };
}

describe("headStats", () => {
  it("recognises a self-attending head", () => {
    const s = headStats(matrix(10, (r, j) => (r === j ? 1 : 0)));
    expect(s.self).toBeCloseTo(1);
    expect(s.prev).toBeCloseTo(0);
    expect(s.distance).toBeCloseTo(0);
    expect(headPattern(s)).toBe("self 1.00");
  });

  it("recognises a previous-token head", () => {
    const s = headStats(matrix(10, (r, j) => (j === r - 1 ? 1 : 0)));
    expect(s.prev).toBeCloseTo(1);
    expect(s.distance).toBeCloseTo(1);
    expect(s.entropy).toBeCloseTo(0);
    expect(s.peak).toBeCloseTo(1);
    expect(headPattern(s)).toBe("previous 1.00");
  });

  it("calls a focused head with a moving target sharp", () => {
    // Each query attends to a different distance (1, 2, 3, 1, 2, 3, ...).
    const s = headStats(matrix(12, (r, j) => (j === Math.max(0, r - 1 - (r % 3)) ? 1 : 0)));
    expect(s.prev).toBeLessThan(0.4);
    expect(headPattern(s)).toBe("sharp · 1.9 back");   // early rows cannot reach 3 back
  });

  it("measures a uniform head as broad", () => {
    const s = headStats(matrix(20, (r, j) => (j <= r ? 1 / (r + 1) : 0)));
    expect(s.entropy).toBeGreaterThan(1.5);
    expect(s.peak).toBeLessThan(0.3);
    expect(headPattern(s)).toMatch(/^broad · /);
  });
});

describe("qkRows", () => {
  const b = readFileSync(new URL("../fixtures/trace.bin", import.meta.url));
  const msg = decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
  const trace = (msg as { trace: Trace }).trace;

  it("reproduces stored scores and covers only earlier keys", () => {
    const { rows, scale } = qkRows(trace, 1, 1, 7);
    expect(rows.length).toBe(8);
    expect(scale).toBeCloseTo(4);
    for (const r of rows) expect(r.score).toBeCloseTo(r.stored, 4);
    expect(rows.reduce((s, r) => s + r.weight, 0)).toBeCloseTo(1, 5);
  });

  it("parses head ids", () => {
    expect(headOf("L3.attn.H2.weights")).toEqual({ layer: 3, head: 2 });
    expect(headOf("L3.mlp.up")).toBeNull();
  });
});
