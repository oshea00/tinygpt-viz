import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Trace, Weights } from "../types";
import { buildLayout, hitTest, renderAtlas, tileTensor, type AtlasSources } from "./atlas";
import { at, row } from "./tensor";

function load(name: string) {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}
const trace = (load("trace.bin") as { trace: Trace }).trace;
const weights = (load("weights.bin") as { weights: Weights }).weights;
const layout = buildLayout(trace, weights, 2, 2);   // fixture: tiny config, 2 layers × 2 heads

describe("atlas layout", () => {
  it("places every weight exactly once, one pixel per number", () => {
    const placed = layout.tiles.filter((t) => t.kind === "weight");
    expect(placed.map((t) => t.name).sort()).toEqual(Object.keys(weights.tensors).sort());
    const pixels = placed.reduce((n, t) => n + t.rows * t.cols, 0);
    const params = Object.values(weights.tensors).reduce((n, t) => n + t.data.length, 0);
    expect(pixels).toBe(params);
  });

  it("puts weights above activations, columns left to right", () => {
    expect(layout.columns.map((c) => c.title)).toEqual(["input", "block 0", "block 1", "output"]);
    for (const t of layout.tiles) {
      const top = layout.columns[t.column].actTop;   // each column's own activation start
      if (t.kind === "weight") expect(t.y + t.rows).toBeLessThanOrEqual(top);
      else expect(t.y).toBeGreaterThanOrEqual(top);
    }
  });

  it("has no overlapping tiles and fits the bitmap", () => {
    const ts = layout.tiles;
    for (const t of ts) {
      expect(t.x + t.cols).toBeLessThanOrEqual(layout.width);
      expect(t.y + Math.max(t.rows, 2)).toBeLessThanOrEqual(layout.height);
    }
    for (let i = 0; i < ts.length; i++) for (let j = i + 1; j < ts.length; j++) {
      const a = ts[i], b = ts[j];
      const overlap = a.x < b.x + b.cols && b.x < a.x + a.cols && a.y < b.y + Math.max(b.rows, 2) && b.y < a.y + Math.max(a.rows, 2);
      expect(overlap, `${a.name} / ${b.name}`).toBe(false);
    }
  });

  it("hit-tests back to tensor coordinates", () => {
    const t = layout.tiles.find((t) => t.name === "L1.W_up")!;
    const hit = hitTest(layout, t.x + 7.5, t.y + 3.2)!;
    expect(hit.tile.name).toBe("L1.W_up");
    expect([hit.r, hit.c]).toEqual([3, 7]);
    expect(hitTest(layout, -5, -5)).toBeNull();
  });
});

describe("atlas modes", () => {
  const base: AtlasSources = { trace, weights, mode: "values", position: 4 };

  it("weight × input is x[i] · W[i, j] for the selected position", () => {
    const tile = layout.tiles.find((t) => t.name === "L0.W_q")!;
    const flow = tileTensor(tile, { ...base, mode: "flow" })!;
    const x = row(trace.byId.get("L0.ln1")!.output, 4);
    expect(at(flow, 5, 9)).toBeCloseTo(x[5] * at(weights.tensors["L0.W_q"], 5, 9), 6);
  });

  it("diff against itself is zero, and is missing without a pin", () => {
    const tile = layout.tiles.find((t) => t.name === "L1.mlp.gelu")!;
    const d = tileTensor(tile, { ...base, mode: "diff", pinnedTrace: trace, pinnedWeights: weights })!;
    expect(Math.max(...d.data.map(Math.abs))).toBe(0);
    expect(tileTensor(tile, { ...base, mode: "diff" })).toBeNull();
  });

  it("renders one opaque image per tile, sized to the tile", () => {
    const { images } = renderAtlas(layout, base);
    expect(images.length).toBe(layout.tiles.length);
    for (const im of images) {
      expect(im.width).toBe(im.tile.cols);
      expect(im.height).toBe(Math.max(im.tile.rows, 2));
      expect(im.rgba.length).toBe(im.width * im.height * 4);
      expect(im.rgba[3]).toBe(255);
    }
  });
});
