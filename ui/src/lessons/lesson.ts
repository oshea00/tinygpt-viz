// Lessons are plain JSON (see ./*.json): a sequence of steps, each setting up the views
// and narrating, optionally asking the learner to find something. New lessons need no code.

import type { Intervention, Trace } from "../types";
import { headStats } from "../lib/attention";
import { argmax } from "../lib/residual";
import { row } from "../lib/tensor";

export type Tab = "inspector" | "attention" | "residual" | "lens" | "atlas" | "interventions" | "concepts" | "onestep";

/** What counts as a correct answer. Checks that depend on the trained model are computed
 * from the live trace, so lessons work on any checkpoint. */
export type Check =
  | { kind: "stage"; ids: string[] }                       // select one of these stages
  | { kind: "position"; positions: number[] }              // select one of these positions
  | { kind: "headBest"; stat: "prev" | "self" | "first" | "peak" }   // select the head scoring highest
  | { kind: "attnTop" }                                    // pick the key the selected head weights most for the query
  | { kind: "tab"; tab: Tab };

export interface LessonStep {
  text: string;                 // narration; **bold** and `code` are rendered
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
  ask?: { question: string; check: Check; hint?: string; correct?: string };
}

export interface Lesson {
  id: string;
  order: number;
  title: string;
  summary: string;
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

export function isCorrect(check: Check, sel: Selected, trace: Trace | null, nLayer: number, nHead: number): boolean {
  switch (check.kind) {
    case "stage": return check.ids.includes(sel.stageId);
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

/** Minimal inline formatting for narration: **bold** and `code`, everything else escaped. */
export function formatText(text: string): string {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
}

export function loadLessons(): Lesson[] {
  const files = import.meta.glob<Lesson>("./*.json", { eager: true, import: "default" });
  return Object.values(files).sort((a, b) => a.order - b.order);
}
