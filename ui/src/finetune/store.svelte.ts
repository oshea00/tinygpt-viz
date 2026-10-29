// State for the fine-tune tab: low-rank adapters (core/lora.py) trained live on a story set, and
// the preference adapters trained offline with DPO (core/dpo.py; "refusal" ships with the model),
// with what each section shows.

import type { BinaryMsg } from "../bridge";
import { app } from "../state/app.svelte";
import type { Capture } from "../lib/concepts";
import type { AdapterHeader, Intervention, JsonMsg, LoraPoint, PairScoresMsg, Tensor } from "../types";
import dragon from "./dragon-stories.json";

export type FinetuneSection = "lora" | "preferences";
export const LIVE = "dragon";              // the adapter the LoRA section trains
export const READY = "dragon-ready";       // the same, trained in advance (core/lora.py --train)
export const REFUSAL = "refusal";         // the preference adapter the model ships with, and lesson 15's
export interface Pair { prompt: string; chosen: string; rejected: string; kind: "unsafe" | "safe"; activity: string; split: "train" | "heldOut" }

class FinetuneStore {
  section = $state<FinetuneSection>("lora");
  pref = $state(REFUSAL);                  // the preference adapter the preferences section shows
  adapters = $state.raw<Record<string, AdapterHeader>>({});
  tensors = $state.raw<Record<string, Record<string, Tensor>>>({});
  private asked = new Set<string>();

  // Training an adapter on stories.
  stories = $state(dragon.stories.join("\n\n"));
  topic = $state(dragon.topic);
  rank = $state(4);
  steps = $state(150);
  lr = $state(3e-3);
  targets = $state<string[]>(["q", "v"]);
  running = $state(false);
  finished = $state(false);                // the last training ran to its end (not stopped)
  curve = $state.raw<LoraPoint[]>([]);
  total = $state(0);
  startedAt = 0;

  // Comparing generations, the adapter's anatomy, and preference pairs' scores.
  generated = $state<{ base?: string; adapter?: string }>({});
  ownAnswers = $state<{ base?: string; tuned?: string }>({});      // "ask it yourself", in the preference section
  layer = $state(3);
  target = $state("q");
  scores = $state.raw<PairScoresMsg | null>(null);
  scoring = $state(false);
  // Held-out requests answered by the base model and the tuned one, by index into heldOut().
  tests = $state.raw<Record<number, { base?: string; tuned?: string }>>({});
  // For each held-out request, the probability that the answer starts with the refusal's first token.
  firstToken = $state.raw<{ base: number[]; tuned: number[] } | null>(null);
  // The training requests' last-token vectors with the tuned model on, for the refusal direction.
  direction = $state.raw<Capture | null>(null);
  directionPending = $state(false);
  steered = $state<{ text?: string; label?: string }>({});

  /** The preference adapters (core/dpo.py) the model has, "refusal" first. */
  prefNames(): string[] {
    return Object.values(this.adapters).filter((a) => a.info.kind === "dpo").map((a) => a.name)
      .sort((a, b) => +(b === REFUSAL) - +(a === REFUSAL) || a.localeCompare(b));
  }

  /** Show another preference adapter: what was measured with the last one no longer applies. */
  choose(name: string): void {
    if (name === this.pref) return;
    if (app.adapter?.name === this.pref) app.setAdapter(null);
    this.pref = name;
    this.scores = null;
    this.tests = {};
    this.firstToken = null;
    this.direction = null;
    this.steered = {};
    this.ownAnswers = {};
  }

  storyList(): string[] {
    return this.stories.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  }

  train(): void {
    this.curve = [];
    this.running = true;
    this.finished = false;
    this.total = this.steps;
    this.startedAt = performance.now();
    if (app.adapter?.name === LIVE) app.setAdapter(null);      // its old version is about to vanish
    // A plain copy of the targets: the in-browser backend posts commands to a worker, which can't clone a state proxy.
    app.send({ type: "loraTrain", name: LIVE, texts: this.storyList(), topic: this.topic, rank: this.rank, targets: [...this.targets],
               steps: this.steps, lr: this.lr });
  }

  stop(): void {
    app.send({ type: "loraStop" });
  }

  /** An adapter's A and B tensors, asked for once (training sends them as they change). */
  weightsOf(name: string): Record<string, Tensor> | undefined {
    if (!this.tensors[name] && this.adapters[name] && !this.asked.has(name)) {
      this.asked.add(name);
      app.send({ type: "adapterWeights", name });
    }
    return this.tensors[name];
  }

  /** Continue a prompt with the base model and with the adapter, the same seed for both. */
  generate(prompt: string, name: string, scale: number, n = 60, temperature = 0.8, seed = 0): void {
    this.generated = {};
    app.send({ type: "generateWith", prompt, n, temperature, seed, tag: "ftBase", adapter: null });
    app.send({ type: "generateWith", prompt, n, temperature, seed, tag: "ftAdapter", adapter: { name, scale } });
  }

  /** A request of the learner's own, answered greedily with the preference adapter off and on. */
  ask(prompt: string, scale: number): void {
    this.ownAnswers = {};
    app.send({ type: "generateWith", prompt, n: 16, temperature: 0, tag: "ftAskBase", adapter: null });
    app.send({ type: "generateWith", prompt, n: 16, temperature: 0, tag: "ftAskTuned", adapter: { name: this.pref, scale } });
  }

  pairs(): Pair[] {
    return (this.adapters[this.pref]?.info.pairs as Pair[] | undefined) ?? [];
  }

  score(scale: number): void {
    const h = this.adapters[this.pref];
    if (!h) return;
    this.scoring = true;
    app.send({ type: "scorePairs", name: this.pref, scale, beta: h.info.beta ?? 0.5,
               pairs: this.pairs().map(({ prompt, chosen, rejected }) => ({ prompt, chosen, rejected })) });
  }

  heldOut(): Pair[] {
    return this.pairs().filter((p) => p.split === "heldOut");
  }

  /** Every held-out request, answered greedily with the adapter off and on. */
  test(scale: number): void {
    this.tests = {};
    this.firstToken = null;
    // The refusal's first token (" I" of " I'm sorry"), scored as a one-token answer: scorePairs gives its
    // log-probability with the adapter (policy) and without (reference).
    const first = " " + (this.adapters[this.pref]?.info.refusal ?? " I'm").trim().split(/[\s']/)[0];
    app.send({ type: "scorePairs", name: this.pref, scale, beta: 1, tag: "ftFirst",
               pairs: this.heldOut().map((p) => ({ prompt: p.prompt, chosen: first, rejected: first })) });
    this.heldOut().forEach((p, i) => {
      app.send({ type: "generateWith", prompt: p.prompt, n: 14, temperature: 0, tag: `ftTest:${i}:base`, adapter: null });
      app.send({ type: "generateWith", prompt: p.prompt, n: 14, temperature: 0, tag: `ftTest:${i}:tuned`, adapter: { name: this.pref, scale } });
    });
  }

  /** The residual stream at each training request's last token, with the tuned model. */
  captureDirection(scale: number): void {
    this.directionPending = true;
    const prompts = this.pairs().filter((p) => p.split === "train").map((p) => ({
      text: p.prompt, targets: [{ start: p.prompt.length - 1, end: p.prompt.length, word: p.activity, category: p.kind }],
    }));
    app.send({ type: "capture", prompts, pool: "last", model: "current", tag: "ftDirection", adapter: { name: this.pref, scale } });
  }

  /** A request continued with a steering intervention, on the tuned model or the base one. */
  steer(prompt: string, interventions: Intervention[], tuned: boolean, scale: number, label: string): void {
    this.steered = { label };
    app.send({ type: "generateWith", prompt, n: 14, temperature: 0, interventions, tag: "ftSteer",
               adapter: tuned ? { name: this.pref, scale } : null });
  }

  handle(m: JsonMsg | BinaryMsg): boolean {
    switch (m.type) {
      case "hello":
        this.adapters = Object.fromEntries((m.adapters ?? []).map((a) => [a.name, a]));
        this.tensors = {};
        this.asked.clear();
        if (app.adapter && !this.adapters[app.adapter.name]) app.adapter = null;
        if (!this.adapters[this.pref]) this.pref = this.prefNames()[0] ?? REFUSAL;
        return false;
      case "adapter": {
        const { tensors, ...header } = m.adapter;
        this.adapters = { ...this.adapters, [header.name]: header };
        this.tensors = { ...this.tensors, [header.name]: tensors };
        // A newly trained version: the views showing it with the adapter on are out of date.
        if (app.adapter?.name === header.name && !this.running) app.run();
        return true;
      }
      case "loraProgress":
        if (m.name !== LIVE) return true;
        this.curve = [...this.curve, ...m.points];
        this.total = m.steps;
        this.running = m.running;
        if (!m.running) this.finished = this.curve.some((p) => p.step > 0 && p.step === this.steps);
        return true;
      case "pairScores":
        if (m.tag === "ftFirst") {
          this.firstToken = { base: m.ref_chosen.map(Math.exp), tuned: m.policy_chosen.map(Math.exp) };
          return true;
        }
        this.scores = m;
        this.scoring = false;
        return true;
      case "capture":
        if (m.tag !== "ftDirection") return false;
        this.direction = m.capture;
        this.directionPending = false;
        return true;
      case "generated": {
        const test = m.tag?.match(/^ftTest:(\d+):(base|tuned)$/);
        if (test) this.tests = { ...this.tests, [+test[1]]: { ...this.tests[+test[1]], [test[2]]: m.text } };
        else if (m.tag === "ftBase") this.generated = { ...this.generated, base: m.text };
        else if (m.tag === "ftAdapter") this.generated = { ...this.generated, adapter: m.text };
        else if (m.tag === "ftSteer") this.steered = { ...this.steered, text: m.text };
        else if (m.tag === "ftAskBase") this.ownAnswers = { ...this.ownAnswers, base: m.text };
        else if (m.tag === "ftAskTuned") this.ownAnswers = { ...this.ownAnswers, tuned: m.text };
        else return false;
        return true;
      }
      case "error":
        if (m.message.startsWith("loraTrain")) this.running = false;
        if (m.message.startsWith("scorePairs")) this.scoring = false;
        if (m.message.startsWith("capture")) this.directionPending = false;
        return false;
    }
    return false;
  }
}

export const finetune = new FinetuneStore();
app.handlers.push((m) => finetune.handle(m));
