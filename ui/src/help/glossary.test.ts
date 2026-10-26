import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import { formatText } from "../lessons/lesson";
import type { Trace, Weights } from "../types";
import { GLOSSARY, glossKey } from "./glossary";

function load(name: string) {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}
const trace = (load("trace.bin") as { trace: Trace }).trace;
const weights = (load("weights.bin") as { weights: Weights }).weights;
const entry = (key: string | null) => GLOSSARY.find((e) => e.key === key);

describe("glossary coverage", () => {
  it("has an entry for every kind of stage, whose example stage exists", () => {
    for (const s of trace.stages) {
      const e = entry(glossKey(s.id));
      expect(e?.section, s.id).toBe("Stages");
      expect(trace.byId.has(e!.stage!), e!.stage).toBe(true);
    }
  });

  it("has an entry for every weight", () => {
    for (const name of Object.keys(weights.tensors)) expect(entry(glossKey(name))?.section, name).toBe("Weights");
  });

  it("uses unique keys that are safe as element ids", () => {
    expect(new Set(GLOSSARY.map((e) => e.key)).size).toBe(GLOSSARY.length);
    for (const e of GLOSSARY) expect(e.key).toMatch(/^[\w.-]+$/);
  });
});

describe("glossary links", () => {
  it("match names for any layer and head, and concepts in any case", () => {
    expect(glossKey("L3.mlp.up")).toBe(glossKey("L0.mlp.up"));
    expect(glossKey("L5.attn.H7.weights")).toBe(glossKey("L0.attn.H0.weights"));
    expect(entry(glossKey("W_Q"))?.title).toBe("Query projection");
    expect(entry(glossKey("Residual Stream"))?.section).toBe("Concepts");
    expect(entry(glossKey("logits"))?.section).toBe("Stages");   // the code name wins over the concept
    expect(glossKey("␣upon")).toBeNull();
  });

  it("turn code spans and [[terms]] into links, and leave the rest alone", () => {
    expect(formatText("row of `wte`")).toContain(`data-gloss="${glossKey("wte")}"><code>wte</code></a>`);
    expect(formatText("the [[residual stream]]")).toContain(`>residual stream</a>`);
    expect(formatText("[[the maps|PCA]]")).toContain(`data-gloss="${glossKey("PCA")}">the maps</a>`);
    expect(formatText("a [[no such term]]")).toBe("a no such term");
    expect(formatText("`␣upon`")).toBe("<code>␣upon</code>");
    expect(formatText("`<b>`")).toBe("<code>&lt;b&gt;</code>");
  });
});
