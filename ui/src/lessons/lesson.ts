// Lessons are plain JSON (see ./*.json): a sequence of steps, each setting up the views
// and narrating, optionally asking the learner to find something. New lessons need no code.

import type { Intervention, LensMsg, Trace } from "../types";
import { headStats } from "../lib/attention";
import { argmax } from "../lib/residual";
import { at, dims2, row } from "../lib/tensor";
import { glossKey } from "../help/glossary";

export type Tab = "inspector" | "embeddings" | "attention" | "residual" | "lens" | "atlas" | "interventions" | "concepts" | "onestep";

/** What counts as a correct answer. Checks that depend on the trained model are computed
 * from the live trace, so lessons work on any checkpoint. */
export type Check =
  | { kind: "stage"; ids: string[] }                       // select one of these stages
  | { kind: "position"; positions: number[] }              // select one of these positions
  | { kind: "headBest"; stat: "prev" | "self" | "first" | "peak" }   // select the head scoring highest
  | { kind: "attnTop" }                                    // pick the key the selected head weights most for the query
  | { kind: "tab"; tab: Tab }
  | { kind: "bright"; ids: string[] }                      // click a cell of one of these stages at least half its maximum
  | { kind: "lensEarly" }                                  // click a logit-lens cell in the first half of the rows that is correct
  | { kind: "captured" }                                   // capture the concept prompt set (and the untrained model, if ticked)
  | { kind: "trained" }                                    // take a step in the training step tab
  | { kind: "reveal" };                                    // a question to think about: answered by showing the answer

export interface LessonStep {
  text: string;                 // narration; **bold**, `code` and {{placeholders}} (see lessonVars) are rendered
  prompt?: string;
  tab?: Tab;
  stage?: string;
  phase?: "forward" | "backward";
  position?: number | "last";
  dim?: number;
  head?: [number, number];      // [layer, head]
  atlasMode?: "values" | "grads" | "flow" | "diff" | "recent";
  lensMode?: "confidence" | "actual";
  residualMode?: "snapshots" | "writes";
  conceptPanel?: "similarity" | "map" | "neighbours" | "drift" | "probes" | "steer" | "attribution" | "neurons";
  interventions?: Intervention[];   // replaces the active list; [] clears it
  sampling?: { temperature: number; topK?: number | null; topP?: number | null };
  kvCache?: boolean;            // switch KV cache mode on or off
  ask?: { question: string; check: Check; hint?: string; correct?: string };
}

export interface Lesson {
  id: string;
  order: number;
  title: string;
  summary: string;
  requiresTraining?: boolean;   // trains the model: needs a server started with --train
  steps: LessonStep[];
}

export interface Selected {
  stageId: string;
  position: number;
  dim: number;
  layer: number;
  head: number;
  tab: string;
}

/** State outside the selection that some checks read. Only those checks need it. */
export interface Live {
  lensSnap?: number;            // logit-lens row clicked, -1 for none
  lens?: LensMsg | null;
  captured?: boolean;
  trained?: boolean;
  revealed?: boolean;
}

export function isCorrect(check: Check, sel: Selected, trace: Trace | null, nLayer: number, nHead: number, live: Live = {}): boolean {
  switch (check.kind) {
    case "stage": return check.ids.includes(sel.stageId);
    case "bright": {
      const out = trace?.byId.get(sel.stageId)?.output;
      if (!out || !check.ids.includes(sel.stageId) || sel.position < 0 || sel.dim < 0) return false;
      const [rows, cols] = dims2(out);
      if (sel.position >= rows || sel.dim >= cols) return false;
      let max = -Infinity;
      for (const v of out.data) if (v > max) max = v;
      return max > 0 && at(out, sel.position, sel.dim) >= max / 2;
    }
    case "lensEarly": {
      const s = live.lensSnap ?? -1, p = sel.position, lens = live.lens;
      if (!trace || !lens || lens.prompt !== trace.prompt || s < 0 || p < 0) return false;
      if (s >= lens.topIds.length / 2) return false;
      const target = trace.targets[p];
      return target >= 0 && lens.topIds[s]?.[p]?.[0] === target;
    }
    case "captured": return !!live.captured;
    case "trained": return !!live.trained;
    case "reveal": return !!live.revealed;
    case "position": return check.positions.includes(sel.position);
    case "tab": return sel.tab === check.tab;
    case "headBest": {
      if (!trace) return false;
      let best = [-1, -1], bestV = -Infinity;
      for (let l = 0; l < nLayer; l++) for (let h = 0; h < nHead; h++) {
        const w = trace.byId.get(`L${l}.attn.H${h}.weights`);
        if (!w) continue;
        const v = headStats(w.output)[check.stat];
        if (v > bestV) [bestV, best] = [v, [l, h]];
      }
      return sel.layer === best[0] && sel.head === best[1];
    }
    case "attnTop": {
      const w = trace?.byId.get(`L${sel.layer}.attn.H${sel.head}.weights`);
      if (!w || !trace) return false;
      const q = sel.position >= 0 && sel.position < trace.tokens.length ? sel.position : trace.tokens.length - 1;
      return sel.dim === argmax(row(w.output, q).subarray(0, q + 1));
    }
  }
}

/** Words and numbers that depend on the running model, for {{name}} in narration. */
export function lessonVars(cfg: { d_model: number; vocab_size: number }): Record<string, string> {
  return { d: String(cfg.d_model), V: cfg.vocab_size.toLocaleString() };
}

/** Why a lesson cannot run here (and what to start the server with), or null if it can. */
export function unavailable(lesson: Lesson, training: boolean): string | null {
  if (lesson.requiresTraining && !training) return "trains the model: start the server with --train";
  return null;
}

export const available = (lesson: Lesson, training = true) => unavailable(lesson, training) === null;

const unescape = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
const glossLink = (key: string, inner: string) => `<a class="gloss" href="#${key}" data-gloss="${key}">${inner}</a>`;

/** Minimal inline formatting for narration: **bold**, `code`, [[term]] or [[shown text|term]],
 * and {{vars}}, everything else escaped. A `code` span that names a glossary entry, and every
 * [[term]] that does, links to it (help/glossary.ts). */
export function formatText(text: string, vars: Record<string, string> = {}): string {
  const esc = text.replace(/\{\{(\w+)\}\}/g, (m, k) => vars[k] ?? m).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/`(.+?)`/g, (_, c) => {
      const key = glossKey(unescape(c));
      return key ? glossLink(key, `<code>${c}</code>`) : `<code>${c}</code>`;
    })
    .replace(/\[\[(.+?)\]\]/g, (_, c) => {
      const [shown, term] = c.split("|");
      const key = glossKey(unescape(term ?? shown));
      return key ? glossLink(key, shown) : shown;
    });
}

export function loadLessons(): Lesson[] {
  const files = import.meta.glob<Lesson>("./*.json", { eager: true, import: "default" });
  return Object.values(files).sort((a, b) => a.order - b.order);
}
