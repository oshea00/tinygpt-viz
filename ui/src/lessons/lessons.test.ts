// Every lesson must make sense on the smallest model (2 layers × 2 heads, as in the fixture).
import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeBinary } from "../bridge";
import type { Trace } from "../types";
import { available, unavailable, formatText, isCorrect, lessonVars, stepsFor, type Lesson } from "./lesson";

const dir = new URL(".", import.meta.url);
const lessons: Lesson[] = readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(new URL(f, dir), "utf8")));
const b = readFileSync(new URL("../fixtures/trace.bin", import.meta.url));
const trace = (decodeBinary(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer) as { trace: Trace }).trace;
const TABS = ["inspector", "attention", "residual", "lens", "atlas", "interventions", "concepts", "onestep"];

describe("starter lessons", () => {
  it("are numbered 1..n with unique ids", () => {
    expect(lessons.map((l) => l.order).sort((a, b) => a - b)).toEqual(lessons.map((_, i) => i + 1));
    expect(new Set(lessons.map((l) => l.id)).size).toBe(lessons.length);
  });

  for (const l of lessons) {
    it(`${l.order}. ${l.title} refers only to things that exist`, () => {
      expect(l.steps.length).toBeGreaterThan(2);
      // Check every step as both kinds of model see it.
      const kinds = (["char", "bpe"] as const).filter((k) => available(l, k));
      expect(kinds.length).toBeGreaterThan(0);
      for (const s of kinds.flatMap((k) => stepsFor(l, k))) {
        expect(s.text.length).toBeGreaterThan(20);
        const known = Object.keys(lessonVars({ tokenizer: "bpe", d_model: 1, vocab_size: 1 }));
        for (const t of [s.text, s.ask?.question ?? "", s.ask?.hint ?? "", s.ask?.correct ?? ""]) {
          for (const m of t.matchAll(/\{\{(\w+)\}\}/g)) expect(known, `unknown {{${m[1]}}}`).toContain(m[1]);
        }
        if (s.stage) expect(trace.byId.has(s.stage), s.stage).toBe(true);
        if (s.tab) expect(TABS).toContain(s.tab);
        if (s.head) expect(s.head[0] < 2 && s.head[1] < 2).toBe(true);
        if (s.ask?.check.kind === "stage") for (const id of s.ask.check.ids) expect(trace.byId.has(id), id).toBe(true);
        if (s.ask?.check.kind === "tab") expect(TABS).toContain(s.ask.check.tab);
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

  it("finds exactly one best head", () => {
    const hits = [0, 1].flatMap((l) => [0, 1].map((h) => isCorrect({ kind: "headBest", stat: "peak" }, { ...sel, layer: l, head: h }, trace, 2, 2)));
    expect(hits.filter(Boolean).length).toBe(1);
  });

  it("escapes narration and renders bold, code and model-specific words", () => {
    expect(formatText("**a** `b` <i>")).toBe("<b>a</b> <code>b</code> &lt;i&gt;");
    const bpe = lessonVars({ tokenizer: "bpe", d_model: 256, vocab_size: 4096 });
    expect(formatText("one {{unit}} of {{d}} dims, {{V}} {{units}}", bpe)).toBe("one token of 256 dims, 4,096 tokens");
    expect(formatText("{{unit}}", lessonVars({ tokenizer: "char", d_model: 64, vocab_size: 64 }))).toBe("character");
  });

  it("shows model-specific steps only to that kind of model", () => {
    const tokens = lessons.find((l) => l.id === "tokens")!;
    expect(stepsFor(tokens, "bpe").length).toBe(stepsFor(tokens, "char").length + 1);
    expect(stepsFor(tokens, "bpe").at(-1)!.tab).toBe("atlas");
  });

  it("merges a step's variant for this kind of model, and drops steps it nulls", () => {
    const st = { text: "x".repeat(30), prompt: "a", variants: { bpe: { prompt: "b" } } };
    const lesson = { id: "x", order: 0, title: "", summary: "", steps: [st, { ...st, variants: { char: null } }] };
    expect(stepsFor(lesson, "char").map((x) => x.prompt)).toEqual(["a"]);
    expect(stepsFor(lesson, "bpe").map((x) => x.prompt)).toEqual(["b", "a"]);   // the second step has no bpe variant
  });

  it("marks the lesson that trains as needing --train", () => {
    const training = lessons.find((l) => l.id === "training")!;
    expect(available(training, "char", true)).toBe(true);
    expect(available(training, "char", false)).toBe(false);
    expect(unavailable(training, "char", false)).toContain("--train");
    expect(lessons.filter((l) => !available(l, "char", false)).map((l) => l.id).sort()).toEqual(["concepts", "training"]);
  });

  it("marks lessons that need the other kind of model", () => {
    const concepts = lessons.find((l) => l.id === "concepts")!;
    expect(available(concepts, "bpe")).toBe(true);
    expect(available(concepts, "char")).toBe(false);
  });
});
