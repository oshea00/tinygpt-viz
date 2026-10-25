// State for the concept explorer: the prompt set, captured vectors (trained model and the
// untrained baseline), and results of the server-side tools.

import type { BinaryMsg } from "../bridge";
import type { Capture } from "../lib/concepts";
import { app } from "../state/app.svelte";
import type { JsonMsg, NeuronExample, Trace } from "../types";
import { DEFAULT_SET, parseSet } from "./sets";

export type Panel = "similarity" | "map" | "neighbours" | "drift" | "probes" | "steer" | "attribution" | "neurons";

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

  readonly parsed = $derived(parseSet(this.source));

  capture(): void {
    const prompts = this.parsed.prompts;
    if (!prompts.length) return;
    this.pending = this.withBaseline ? 2 : 1;
    app.send({ type: "capture", prompts, pool: this.pool, model: "current" });
    if (this.withBaseline) app.send({ type: "capture", prompts, pool: this.pool, model: "untrained" });
    else this.untrained = null;
  }

  snapIndex(c: Capture | null): number {
    if (!c) return 0;
    return this.snap >= 0 && this.snap < c.S ? this.snap : c.S - 1;
  }

  handle(m: JsonMsg | BinaryMsg): boolean {
    switch (m.type) {
      case "capture":
        if (m.capture.model === "untrained") this.untrained = m.capture;
        else this.current = m.capture;
        this.pending = Math.max(0, this.pending - 1);
        if (!this.word && m.capture.items.length) this.word = m.capture.items[0].word;
        return true;
      case "generated":
        if (m.tag === "plain" || m.tag === "steered") this.generated = { ...this.generated, [m.tag]: m.text };
        return true;
      case "occlusion":
        this.occlusion = m;
        return true;
      case "neuronTop":
        this.neurons = m;
        return true;
      case "trace":
        if (m.trace.tag !== "attr") return false;
        this.attrTrace = m.trace;
        return true;
      case "error":
        this.pending = 0;
        return false;
    }
    return false;
  }
}

export const concepts = new ConceptStore();
app.handlers.push((m) => concepts.handle(m));
