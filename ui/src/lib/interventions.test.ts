import { readFileSync } from "node:fs";
import { describe as group, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Trace } from "../types";
import { biggestChange, comparable, describe, difference, stageChanges, targetProbs } from "./interventions";
import { at } from "./tensor";

function load(name: string): Trace {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return (decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer) as { trace: Trace }).trace;
}
const base = load("trace.bin");
const ablated = load("trace_ablated.bin");

group("intervention effects", () => {
  it("decodes what was changed", () => {
    expect(ablated.interventions).toEqual([{ kind: "head", layer: 1, head: 0, mode: "zero" }]);
    expect(describe(ablated.interventions![0])).toBe("L1 head 0: zeroed");
    expect(comparable(ablated, base)).toBe(true);
  });

  it("finds where the change enters: nothing before L1 head 0's output", () => {
    const changes = stageChanges(ablated, base);
    const first = changes.findIndex((c) => c.change > 0);
    expect(changes[first].id).toBe("L1.attn.H0.mix");
    expect(changes.slice(0, first).every((c) => c.change === 0)).toBe(true);
    expect(changes.find((c) => c.id === "logits")!.change).toBeGreaterThan(0);
  });

  it("names the position whose prediction changed most, consistent with the loss", () => {
    const b = biggestChange(ablated, base)!;
    const probs = base.byId.get("probs")!.output, after = ablated.byId.get("probs")!.output;
    const changes = ablated.targets.flatMap((t, i) => (t >= 0 ? [Math.log(at(probs, i, t)) - Math.log(at(after, i, t))] : []));
    expect(b.target).toBe(ablated.targets[b.position]);
    expect(Math.abs(b.loss)).toBeCloseTo(Math.max(...changes.map(Math.abs)), 6);
    // The loss on the prompt is the mean of these per-position changes.
    const mean = changes.reduce((s, x) => s + x, 0) / changes.length;
    expect(ablated.loss! - base.loss!).toBeCloseTo(mean, 4);
    expect(biggestChange(base, base)!.loss).toBe(0);
  });

  it("treats matching masked cells as unchanged", () => {
    const d = difference(ablated.byId.get("L0.attn.H0.masked")!.output, base.byId.get("L0.attn.H0.masked")!.output)!;
    expect(d.data.every((v) => v === 0)).toBe(true);
  });

  it("pairs p(actual next) before and after", () => {
    const tp = targetProbs(ablated, base);
    expect(tp.length).toBe(12);
    expect(tp.some((p) => p.before !== p.after)).toBe(true);
  });
});
