// State for the concept explorer: the prompt set, captured vectors (trained model and the
// untrained baseline), and results of the server-side tools (including activation patching).

import type { BinaryMsg } from "../bridge";
import type { Capture } from "../lib/concepts";
import { app } from "../state/app.svelte";
import type { Intervention, JsonMsg, NeuronExample, PatchResult, Trace } from "../types";
import { DEFAULT_SET, parseSet } from "./sets";

export type Panel = "similarity" | "map" | "neighbours" | "drift" | "probes" | "steer" | "attribution" | "neurons" | "patching";

export type PatchWhat = "resid" | "heads";
export interface PatchCell { what: PatchWhat; row: number; col: number; }

/** The patching panel's starting example: swapping who gave the ball flips the answer. */
export const PATCH_EXAMPLE = {
  clean: "Lily and Tom went to the park. Tom gave the ball to",
  corrupt: "Lily and Tom went to the park. Lily gave the ball to",
  answer: " Lily",
  wrong: " Tom",
};

class ConceptStore {
  source = $state(DEFAULT_SET);
  pool = $state<"last" | "mean">("last");
  withBaseline = $state(true);
  current = $state.raw<Capture | null>(null);
  untrained = $state.raw<Capture | null>(null);
  pending = $state(0);
  snap = $state(-1);                        // snapshot index; -1: the last
  panel = $state<Panel>("similarity");
  word = $state("");                        // word picked for neighbours / drift
  generated = $state<{ plain?: string; steered?: string }>({});
  occlusion = $state.raw<{ prompt: string; position: number; drops: number[] } | null>(null);
  neurons = $state.raw<{ layer: number; neuron: number; examples: NeuronExample[] } | null>(null);
  attrTrace = $state.raw<Trace | null>(null);
  patchInput = $state({ ...PATCH_EXAMPLE });
  patchWhat = $state<PatchWhat>("resid");
  patches = $state.raw<Partial<Record<PatchWhat, PatchResult>>>({});
  patchPending = $state(false);
  patchCell = $state.raw<PatchCell | null>(null);   // the cell last applied

  readonly parsed = $derived(parseSet(this.source));

  capture(): void {
    const prompts = this.parsed.prompts;
    if (!prompts.length) return;
    this.pending = this.withBaseline ? 2 : 1;
    app.send({ type: "capture", prompts, pool: this.pool, model: "current" });
    if (this.withBaseline) app.send({ type: "capture", prompts, pool: this.pool, model: "untrained" });
    else this.untrained = null;
  }

  runPatch(what: PatchWhat = this.patchWhat): void {
    this.patchWhat = what;
    this.patchPending = true;
    app.send({ type: "patch", ...this.patchInput, what });
  }

  /** Patch one cell of a sweep into the live model: the prompt becomes the corrupted one, and
   * the clean run's activation replaces that position (resid) or that head's whole output (heads). */
  applyPatch(r: PatchResult, row: number, col: number): void {
    const T = r.corruptText.length;
    let iv: Intervention;
    if (r.what === "resid") {
      const d = r.source.shape[2], off = (row * T + col) * d;
      iv = { kind: "patch", stage: r.stages[row], positions: [col], values: [Array.from(r.source.data.subarray(off, off + d))], label: "the clean prompt" };
    } else {
      const H = r.grid.shape[1], dh = r.source.shape[3], off = (row * H + col) * T * dh;
      const values = Array.from({ length: T }, (_, p) => Array.from(r.source.data.subarray(off + p * dh, off + (p + 1) * dh)));
      iv = { kind: "patch", stage: r.stages[row * H + col], positions: values.map((_, p) => p), values, label: "the clean prompt" };
    }
    this.patchCell = { what: r.what, row, col };
    if (app.prompt !== r.corrupt) {
      app.prompt = r.corrupt;
      app.promptEdited();
    }
    app.setInterventions([iv]);
  }

  snapIndex(c: Capture | null): number {
    if (!c) return 0;
    return this.snap >= 0 && this.snap < c.S ? this.snap : c.S - 1;
  }

  handle(m: JsonMsg | BinaryMsg): boolean {
    switch (m.type) {
      case "capture":
        if (m.tag?.startsWith("ft")) return false;       // the fine-tune tab's (finetune/store.svelte.ts)
        if (m.capture.model === "untrained") this.untrained = m.capture;
        else this.current = m.capture;
        this.pending = Math.max(0, this.pending - 1);
        if (!this.word && m.capture.items.length) this.word = m.capture.items[0].word;
        return true;
      case "generated":
        if (m.tag !== "plain" && m.tag !== "steered") return false;
        this.generated = { ...this.generated, [m.tag]: m.text };
        return true;
      case "occlusion":
        this.occlusion = m;
        return true;
      case "neuronTop":
        this.neurons = m;
        return true;
      case "patch":
        this.patches = { ...this.patches, [m.patch.what]: m.patch };
        this.patchPending = false;
        return true;
      case "trace":
        if (m.trace.tag !== "attr") return false;
        this.attrTrace = m.trace;
        return true;
      case "error":
        this.pending = 0;
        this.patchPending = false;
        return false;
    }
    return false;
  }
}

export const concepts = new ConceptStore();
app.handlers.push((m) => concepts.handle(m));
