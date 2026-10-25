import { describe, expect, it } from "vitest";
import { DEFAULT_SET, parseSet } from "../concepts/sets";
import { attentionRollout, cosine, direction, drift, heldOutWords, layer, nearestRows, pca, probeByLayer, similarityMatrix, wordMeans, type Capture } from "./concepts";

// A synthetic capture: two categories that separate along dimension 0 at snapshot 1 only,
// plus a large shared offset that centring must remove.
function synthetic(): Capture {
  const words = { animal: ["cat", "dog", "cow", "pig", "hen", "fox", "owl", "bee"], colour: ["red", "tan", "blue", "pink", "gold", "grey", "teal", "jade"] };
  const items = Object.entries(words).flatMap(([category, ws]) => ws.flatMap((word) => [0, 1].map((k) => ({ word, category, prompt: `${word} ${k}`, tokens: [word], position: 1 }))));
  const N = items.length, S = 2, d = 6;
  const vectors = new Float32Array(N * S * d);
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  items.forEach((it, i) => {
    for (let s = 0; s < S; s++) for (let k = 0; k < d; k++) {
      let v = 10 + rand();                                   // shared component
      if (s === 1 && k === 0) v += it.category === "animal" ? 3 : -3;
      vectors[(i * S + s) * d + k] = v;
    }
  });
  return { vectors, N, S, d, items, snapshots: ["embed.sum", "L0.resid_mlp"], model: "current", step: 1 };
}

describe("concept analyses", () => {
  const c = synthetic();

  it("centres before comparing: raw vectors all look alike", () => {
    expect(cosine(layer(c, 0, false)[0], layer(c, 0, false)[20])).toBeGreaterThan(0.99);
    expect(Math.abs(cosine(layer(c, 0)[0], layer(c, 0)[20]))).toBeLessThan(0.9);
  });

  it("finds category structure where it exists", () => {
    const means = wordMeans(c, 1);
    expect(means.length).toBe(16);
    const sim = similarityMatrix(means.map((m) => m.v));
    expect(sim[0 * 16 + 1]).toBeGreaterThan(0.5);        // two animals
    expect(sim[0 * 16 + 15]).toBeLessThan(-0.5);         // animal vs colour
  });

  it("puts the category split on the first principal component", () => {
    const { components, variance, total } = pca(layer(c, 1), 2);
    expect(Math.abs(components[0][0])).toBeGreaterThan(0.95);
    expect(variance[0] / total).toBeGreaterThan(0.5);
  });

  it("probes generalise to held-out words only where the concept is encoded", () => {
    expect(heldOutWords(c.items).size).toBe(4);
    const real = probeByLayer(c);
    expect(real[1].testAcc).toBe(1);
    expect(real[0].testAcc).toBeLessThan(0.9);
    const shuffled = probeByLayer(c, true);
    expect(shuffled[1].testAcc).toBeLessThan(0.9);
  });

  it("builds a direction from group means and measures drift", () => {
    const rows = layer(c, 1);
    const animals = c.items.map((it, i) => (it.category === "animal" ? i : -1)).filter((i) => i >= 0);
    const colours = c.items.map((it, i) => (it.category === "colour" ? i : -1)).filter((i) => i >= 0);
    const dir = direction(rows, animals, colours);
    expect(dir[0]).toBeGreaterThan(5);
    expect(drift(c, "cat")[1].n).toBe(2);
  });

  it("finds nearest table rows and rolls out attention", () => {
    const table = new Float32Array([1, 0, 0.9, 0.1, 0, 1]);
    expect(nearestRows(table, 3, 2, 0, 1)[0].i).toBe(1);
    const T = 3, eye = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const r = attentionRollout([[eye]], T);
    expect(Array.from(r)).toEqual([0, 0, 1]);
  });
});

describe("prompt sets", () => {
  it("expand templates and locate the target word", () => {
    const { prompts, categories, errors } = parseSet(DEFAULT_SET);
    expect(errors).toEqual([]);
    expect(Object.keys(categories)).toEqual(["animal", "colour", "emotion", "family", "place"]);
    expect(prompts.length).toBe(4 * 10 + 4 * 8 + 3 * 6 + 3 * 6 + 2 * 6);
    for (const p of prompts) for (const t of p.targets) expect(p.text.slice(t.start, t.end)).toBe(t.word);
  });

  it("reports malformed templates", () => {
    expect(parseSet("@a: x\nno slot here\n{b} missing").errors.length).toBe(2);
  });
});
