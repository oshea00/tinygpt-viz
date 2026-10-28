import { describe, expect, it } from "vitest";
import {
  centroids, distanceMatrix, geometryByLayer, isTestStory, meanBaseline, meanDiffDirection, meanLogDelta, probeByLayer,
  readoutRows, ridge, scaledDirection, topPrefixes, trainMask, upperTriangleR, type Beliefs,
} from "./beliefs";

// A synthetic capture: 20 stories of 5 sentences, beliefs over 3 words. At snapshot 1 the vector
// is a linear function of the belief (plus noise); at snapshot 0 it is noise only.
function synthetic(): Beliefs {
  const words = ["happy", "sad", "scared"], K = 3, S = 2, d = 8;
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  const prefixes = [], readout: number[] = [], vectors: number[] = [];
  for (let story = 0; story < 20; story++) for (let sentence = 0; sentence < 5; sentence++) {
    prefixes.push({ story, sentence, start: 0, end: 1 });
    const z = [rand() * 4, rand() * 4, rand() * 4].map(Math.exp), sum = z.reduce((a, b) => a + b, 0);
    const p = z.map((v) => v / sum);
    readout.push(...p);
    for (let s = 0; s < S; s++) for (let k = 0; k < d; k++) vectors.push(5 + (s === 1 && k < K ? 4 * p[k] : 0) + 0.01 * rand());
  }
  const N = prefixes.length;
  return {
    vectors: Float32Array.from(vectors), readout: Float32Array.from(readout), mass: new Float32Array(N).fill(0.5), N, S, d, K,
    stories: Array.from({ length: 20 }, (_, i) => `story ${i}`), prefixes, snapshots: ["embed.sum", "L0.resid_mlp"],
    words, cloze: " Everyone was very", model: "current", step: 1,
  };
}

describe("belief analyses", () => {
  const b = synthetic();

  it("holds out whole stories", () => {
    const train = trainMask(b);
    const trainStories = new Set(b.prefixes.filter((_, i) => train[i]).map((p) => p.story));
    const testStories = new Set(b.prefixes.filter((_, i) => !train[i]).map((p) => p.story));
    expect([...testStories].every((s) => !trainStories.has(s) && isTestStory(s))).toBe(true);
    expect(testStories.size).toBe(4);
  });

  it("ridge recovers a planted linear map", () => {
    const X = Array.from({ length: 60 }, (_, i) => [Math.sin(i), Math.cos(3 * i), (i % 7) / 7]);
    const Y = X.map((x) => [2 * x[0] - x[1] + 0.5, x[2]]);
    const train = X.map((_, i) => i % 4 !== 0);
    const fit = ridge(X, Y, train, 1e-6);
    expect(fit.rmse).toBeLessThan(1e-4);
    expect(fit.r[0]).toBeGreaterThan(0.9999);
    // Weights in input units: d(pred_0)/dx = (2, -1, 0).
    expect(fit.dirs[0][0]).toBeCloseTo(2, 3);
    expect(fit.dirs[0][1]).toBeCloseTo(-1, 3);
  });

  it("probes find the belief where it is encoded, and beat the mean only there", () => {
    const [x0, l0] = probeByLayer(b);
    const base = meanBaseline(readoutRows(b), trainMask(b));
    expect(l0.rmse).toBeLessThan(0.15 * base);   // the penalty shrinks a little on 80 rows
    expect(x0.rmse).toBeGreaterThan(0.8 * base);
  });

  it("centroids cap the sentences taken from any one story", () => {
    const top = topPrefixes(b, 0, 30, 2);
    const per = new Map<number, number>();
    for (const i of top) per.set(b.prefixes[i].story, (per.get(b.prefixes[i].story) ?? 0) + 1);
    expect(Math.max(...per.values())).toBeLessThanOrEqual(2);
    // Sorted by belief: the first is the happiest sentence of all.
    const happiest = b.prefixes.map((_, i) => i).reduce((a, i) => (b.readout[i * 3] > b.readout[a * 3] ? i : a), 0);
    expect(top[0]).toBe(happiest);
    expect(centroids(b, readoutRows(b), 10)[0][0]).toBeGreaterThan(centroids(b, readoutRows(b), 10)[1][0]);
  });

  it("distance matrices compare by their upper triangle", () => {
    const pts = [[0, 0], [1, 0], [0, 3], [2, 2]];
    const m = distanceMatrix(pts);
    expect(m[0 * 4 + 1]).toBeCloseTo(1);
    expect(m[2 * 4 + 2]).toBe(0);
    expect(upperTriangleR(m, m, 4)).toBeCloseTo(1);
    expect(upperTriangleR(m, m.map((v) => -v), 4)).toBeCloseTo(-1);
    const g = geometryByLayer(b);
    expect(g.layers[1].r).toBeGreaterThan(0.9);   // vectors encode belief linearly at snapshot 1
  });

  it("mean-difference directions point towards the word", () => {
    const rows = Array.from({ length: b.N }, (_, i) => Array.from(b.vectors.subarray((i * 2 + 1) * 8, (i * 2 + 2) * 8)));
    const v = meanDiffDirection(b, rows, 1);
    expect(v[1]).toBeGreaterThan(Math.abs(v[3]) * 10);
    const scaled = scaledDirection(v, 0.5, 10);
    expect(Math.hypot(...scaled)).toBeCloseTo(5);
  });

  it("log deltas use raw probabilities, so words can rise together", () => {
    const d = meanLogDelta([[0.5, 0.5]], [[0.5, 0.5]], [0.2], [0.4]);
    expect(d[0]).toBeCloseTo(Math.log(2));
    expect(d[1]).toBeCloseTo(Math.log(2));
  });
});
