// Every lesson must make sense on the smallest model (2 layers × 2 heads, as in the fixture).
import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { PatchResult, Trace } from "../types";
import { available, unavailable, formatText, isCorrect, lessonVars, type Lesson, type Selected } from "./lesson";

const dir = new URL(".", import.meta.url);
const lessons: Lesson[] = readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(new URL(f, dir), "utf8")));
const b = readFileSync(new URL("../fixtures/trace.bin", import.meta.url));
const trace = (decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer) as { trace: Trace }).trace;
const TABS = ["inspector", "embeddings", "attention", "residual", "lens", "atlas", "interventions", "concepts", "onestep"];

describe("starter lessons", () => {
  it("are numbered 1..n with unique ids", () => {
    expect(lessons.map((l) => l.order).sort((a, b) => a - b)).toEqual(lessons.map((_, i) => i + 1));
    expect(new Set(lessons.map((l) => l.id)).size).toBe(lessons.length);
  });

  for (const l of lessons) {
    it(`${l.order}. ${l.title} refers only to things that exist`, () => {
      expect(l.steps.length).toBeGreaterThan(2);
      for (const s of l.steps) {
        expect(s.text.length).toBeGreaterThan(20);
        const known = Object.keys(lessonVars({ d_model: 1, vocab_size: 1 }));
        for (const t of [s.text, s.ask?.question ?? "", s.ask?.hint ?? "", s.ask?.correct ?? ""]) {
          for (const m of t.matchAll(/\{\{(\w+)\}\}/g)) expect(known, `unknown {{${m[1]}}}`).toContain(m[1]);
        }
        if (s.stage) expect(trace.byId.has(s.stage), s.stage).toBe(true);
        if (s.tab) expect(TABS).toContain(s.tab);
        if (s.head) expect(s.head[0] < 2 && s.head[1] < 2).toBe(true);
        if (s.ask?.check.kind === "stage") for (const id of s.ask.check.ids) expect(trace.byId.has(id), id).toBe(true);
        if (s.ask?.check.kind === "tab") expect(TABS).toContain(s.ask.check.tab);
        if (s.ask?.check.kind === "bright") for (const id of s.ask.check.ids) expect(trace.byId.has(id), id).toBe(true);
      }
    });

    // What "reset this step" leaves (see Player.reset): a question its own step already
    // answers shows ✓ before the learner does anything, and reset cannot clear it.
    it(`${l.order}. ${l.title} asks nothing the step itself answers`, () => {
      const sel: Selected = { stageId: "embed.sum", position: -1, dim: -1, layer: 0, head: 0, tab: "inspector" };
      for (const [i, s] of l.steps.entries()) {
        if (s.tab) sel.tab = s.tab;
        if (s.head) [sel.layer, sel.head] = s.head;
        if (s.stage) sel.stageId = s.stage;
        if (s.position !== undefined) sel.position = s.position === "last" ? -1 : s.position;
        if (s.dim !== undefined) sel.dim = s.dim;
        if (s.ask) expect(isCorrect(s.ask.check, sel, trace, 2, 2), `step ${i + 1}`).toBe(false);
      }
    });
  }
});

describe("answer checks", () => {
  const sel = { stageId: "embed.sum", position: 3, dim: 0, layer: 0, head: 0, tab: "attention" };
  it("checks stage, position and tab", () => {
    expect(isCorrect({ kind: "stage", ids: ["embed.sum"] }, sel, trace, 2, 2)).toBe(true);
    expect(isCorrect({ kind: "position", positions: [4] }, sel, trace, 2, 2)).toBe(false);
    expect(isCorrect({ kind: "tab", tab: "attention" }, sel, trace, 2, 2)).toBe(true);
  });

  it("checks a bright cell, an early lens hit, and the asks that need an action", () => {
    const out = trace.byId.get("L1.mlp.gelu")!.output;
    const max = Math.max(...out.data);
    const peak = out.data.indexOf(max), cols = out.shape[out.shape.length - 1];
    const at = { ...sel, stageId: "L1.mlp.gelu", position: Math.floor(peak / cols), dim: peak % cols };
    expect(isCorrect({ kind: "bright", ids: ["L1.mlp.gelu"] }, at, trace, 2, 2)).toBe(true);
    expect(isCorrect({ kind: "bright", ids: ["L1.mlp.gelu"] }, { ...at, dim: -1 }, trace, 2, 2)).toBe(false);

    const T = trace.tokens.length;
    const p = trace.targets.findIndex((t) => t >= 0);
    const topIds = Array.from({ length: 6 }, () => Array.from({ length: T }, () => [-1]));
    topIds[0][p] = [trace.targets[p]];
    topIds[5][p] = [trace.targets[p]];
    const lens = { type: "lens" as const, prompt: trace.prompt!, step: trace.step, snapshots: [], topIds, topP: [], pActual: [] };
    const lensSel = { ...sel, position: p };
    expect(isCorrect({ kind: "lensEarly" }, lensSel, trace, 2, 2, { lens, lensSnap: 0 })).toBe(true);
    expect(isCorrect({ kind: "lensEarly" }, lensSel, trace, 2, 2, { lens, lensSnap: 5 })).toBe(false);   // a late row
    expect(isCorrect({ kind: "lensEarly" }, lensSel, trace, 2, 2, { lens, lensSnap: -1 })).toBe(false);  // nothing clicked

    expect(isCorrect({ kind: "captured" }, sel, trace, 2, 2)).toBe(false);
    expect(isCorrect({ kind: "captured" }, sel, trace, 2, 2, { captured: true })).toBe(true);
    expect(isCorrect({ kind: "trained" }, sel, trace, 2, 2, { trained: true })).toBe(true);
    expect(isCorrect({ kind: "reveal" }, sel, trace, 2, 2, { revealed: true })).toBe(true);
  });

  it("finds exactly one best head", () => {
    const hits = [0, 1].flatMap((l) => [0, 1].map((h) => isCorrect({ kind: "headBest", stat: "peak" }, { ...sel, layer: l, head: h }, trace, 2, 2)));
    expect(hits.filter(Boolean).length).toBe(1);
  });

  it("finds no induction head when no token repeats", () => {
    const fixed = { ...trace, tokens: trace.tokens.map((_, i) => i) };
    const hits = [0, 1].flatMap((l) => [0, 1].map((h) => isCorrect({ kind: "headBest", stat: "induction" }, { ...sel, layer: l, head: h }, fixed, 2, 2)));
    expect(hits.some(Boolean)).toBe(false);
  });

  it("checks the patching cells applied", () => {
    // resid: 3 snapshots × 4 positions; the last column first reaches 0.5 at snapshot 1.
    const resid = { what: "resid", grid: { shape: [3, 4], data: new Float32Array([1, 0, 0, 0, 0.6, 0, 0, 0.55, 0, 0, 0, 1]) } } as unknown as PatchResult;
    const heads = { what: "heads", grid: { shape: [2, 2], data: new Float32Array([0.1, 0.2, 0.7, 0.3]) } } as unknown as PatchResult;
    const live = (what: "resid" | "heads", row: number, col: number) => ({ patches: { resid, heads }, patchCell: { what, row, col } });
    expect(isCorrect({ kind: "patchHandoff" }, sel, trace, 2, 2, live("resid", 1, 3))).toBe(true);
    expect(isCorrect({ kind: "patchHandoff" }, sel, trace, 2, 2, live("resid", 2, 3))).toBe(false);   // later than needed
    expect(isCorrect({ kind: "patchHandoff" }, sel, trace, 2, 2, live("resid", 1, 0))).toBe(false);   // not the last position
    expect(isCorrect({ kind: "patchTop" }, sel, trace, 2, 2, live("heads", 1, 0))).toBe(true);
    expect(isCorrect({ kind: "patchTop" }, sel, trace, 2, 2, live("heads", 0, 1))).toBe(false);
    expect(isCorrect({ kind: "patchTop" }, sel, trace, 2, 2, { patches: { heads } })).toBe(false);    // nothing applied
  });

  it("escapes narration and renders bold, code and model-specific words", () => {
    expect(formatText("**a** `b` <i>")).toBe("<b>a</b> <code>b</code> &lt;i&gt;");
    const vars = lessonVars({ d_model: 256, vocab_size: 4096 });
    expect(formatText("one token of {{d}} dims, {{V}} tokens", vars)).toBe("one token of 256 dims, 4,096 tokens");
  });

  it("marks the lesson that trains as needing --train", () => {
    const training = lessons.find((l) => l.id === "training")!;
    expect(available(training, true)).toBe(true);
    expect(available(training, false)).toBe(false);
    expect(unavailable(training, false)).toContain("--train");
    expect(lessons.filter((l) => !available(l, false)).map((l) => l.id)).toEqual(["training"]);
  });
});
