import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { pick, shape } from "./sampling";

const fx = JSON.parse(readFileSync(new URL("../fixtures/sampling.json", import.meta.url), "utf8"));

describe("shape matches core/sampling.py", () => {
  for (const c of fx.cases) {
    it(`T=${c.temperature} k=${c.topK} p=${c.topP}`, () => {
      const { probs } = shape(fx.logits, { temperature: c.temperature, topK: c.topK, topP: c.topP });
      probs.forEach((p, i) => expect(p).toBeCloseTo(c.probs[i], 5));
    });
  }

  it("keeps the unfiltered distribution for the greyed tail", () => {
    const s = shape(fx.logits, { temperature: 1, topK: 2, topP: null });
    expect(s.probs.filter((p) => p > 0).length).toBe(2);
    expect(s.tempered.filter((p) => p > 0).length).toBe(16);
  });
});

describe("pick", () => {
  it("inverts the CDF and skips excluded tokens", () => {
    expect(pick([0, 0.25, 0, 0.75], 0)).toBe(1);
    expect(pick([0, 0.25, 0, 0.75], 0.3)).toBe(3);
    expect(pick([0, 0.25, 0, 0.75], 0.9999999)).toBe(3);
  });
});
