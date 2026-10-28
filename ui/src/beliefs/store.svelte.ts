// State for the beliefs tab (core/beliefs.py): the cloze and word set, per-sentence readouts and
// vectors of whole stories (trained model and the untrained baseline), probe fits, and steered runs.

import type { BinaryMsg } from "../bridge";
import { meanBaseline, probeByLayer, readoutRows, trainMask, type Beliefs, type ProbeLayer } from "../lib/beliefs";
import { app } from "../state/app.svelte";
import type { Intervention, JsonMsg } from "../types";

export type BeliefSection = "trajectory" | "probes" | "geometry" | "steering";
export interface BeliefSteerResult { readout: number[][]; mass: number[]; }

class BeliefStore {
  cloze = $state(" Everyone was very");
  words = $state("happy sad angry scared surprised excited");
  n = $state(100);
  custom = $state("");                      // stories of your own, separated by blank lines
  withBaseline = $state(true);
  section = $state<BeliefSection>("trajectory");
  current = $state.raw<Beliefs | null>(null);
  untrained = $state.raw<Beliefs | null>(null);
  pending = $state(0);
  snap = $state(-1);                        // snapshot index; -1: the last
  probes = $state.raw<{ trained: ProbeLayer[]; untrained: ProbeLayer[] | null; baseline: number } | null>(null);
  story = $state(0);
  sentence = $state(-1);                    // -1: the last
  steer = $state.raw<Record<string, BeliefSteerResult>>({});
  steerPending = $state(0);
  generated = $state<{ plain?: string; steered?: string }>({});

  run(): void {
    const words = this.words.split(/[\s,]+/).filter(Boolean);
    const custom = this.custom.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
    const set = { cloze: this.cloze, words, ...(custom.length ? { stories: custom } : { n: this.n }) };
    this.pending = this.withBaseline ? 2 : 1;
    this.probes = null;
    this.steer = {};
    app.send({ type: "beliefs", ...set, model: "current" });
    if (this.withBaseline) app.send({ type: "beliefs", ...set, model: "untrained" });
    else this.untrained = null;
  }

  /** Ridge probes at every snapshot, for the trained model and the untrained baseline (both read
   * against the trained model's belief). Blocking: call it after the button has repainted. */
  fitProbes(): void {
    const b = this.current;
    if (!b) return;
    const u = this.untrained;
    this.probes = {
      trained: probeByLayer(b),
      untrained: u && u.N === b.N ? probeByLayer(u, b) : null,
      baseline: meanBaseline(readoutRows(b), trainMask(b)),
    };
  }

  /** The readout over some stories with interventions applied; results arrive by tag. */
  steerWith(tag: string, stories: string[], interventions: Intervention[]): void {
    const b = this.current;
    if (!b) return;
    this.steerPending++;
    app.send({ type: "beliefSteer", stories, cloze: b.cloze, words: b.words, interventions, tag });
  }

  /** Continue a prompt with and without the interventions. */
  generate(prompt: string, interventions: Intervention[]): void {
    this.generated = {};
    app.send({ type: "generateWith", prompt, n: 30, tag: "beliefPlain" });
    app.send({ type: "generateWith", prompt, n: 30, interventions, tag: "beliefSteered" });
  }

  snapIndex(b: { S: number } | null): number {
    if (!b) return 0;
    return this.snap >= 0 && this.snap < b.S ? this.snap : b.S - 1;
  }

  handle(m: JsonMsg | BinaryMsg): boolean {
    switch (m.type) {
      case "beliefs":
        if (m.beliefs.model === "untrained") this.untrained = m.beliefs;
        else {
          this.current = m.beliefs;
          this.story = 0;
          this.sentence = -1;
        }
        this.pending = Math.max(0, this.pending - 1);
        return true;
      case "beliefSteer":
        this.steer = { ...this.steer, [m.tag]: { readout: m.readout, mass: m.mass } };
        this.steerPending = Math.max(0, this.steerPending - 1);
        return true;
      case "generated":
        if (m.tag === "beliefPlain") this.generated = { ...this.generated, plain: m.text };
        else if (m.tag === "beliefSteered") this.generated = { ...this.generated, steered: m.text };
        else return false;
        return true;
      case "error":
        this.pending = 0;
        this.steerPending = 0;
        return false;
    }
    return false;
  }
}

export const beliefs = new BeliefStore();
app.handlers.push((m) => beliefs.handle(m));
