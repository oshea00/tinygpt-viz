// Shared state. Every view is a function of (trace, selection, settings); clicking in any
// view updates the selection and every other view follows.

import { Bridge, type BinaryMsg } from "../bridge";
import type { Command, Hello, JsonMsg, Stage, Status, StepExplain, Trace, Weights } from "../types";
import { headOf } from "../lib/attention";
import type { ScaleMode } from "../lib/scale";

const MAX_POINTS = 50_000;

class AppState {
  connected = $state(false);
  hello = $state.raw<Hello | null>(null);
  status = $state<Status | null>(null);
  trace = $state.raw<Trace | null>(null);
  weights = $state.raw<Weights | null>(null);
  stepExplain = $state.raw<StepExplain | null>(null);
  lossPoints = $state.raw<[number, number][]>([]);
  valPoints = $state.raw<[number, number][]>([]);
  tokensPerSec = $state(0);
  autoSample = $state<{ step: number; text: string } | null>(null);
  sample = $state.raw<{ text: string; tokens: string[]; chosenProbs: number[] } | null>(null);
  error = $state("");
  prompt = $state("");

  private bridge: Bridge | null = null;

  start(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.bridge = new Bridge(`${proto}://${location.host}/ws`, (m) => this.onMessage(m), (c) => (this.connected = c));
    this.bridge.connect();
  }

  send(cmd: Command): void {
    this.bridge?.send(cmd);
  }

  run(prompt = this.prompt): void {
    this.send({ type: "run", prompt });
  }

  private onMessage(m: JsonMsg | BinaryMsg): void {
    switch (m.type) {
      case "hello": {
        this.hello = m;
        this.status = { step: m.step, playing: m.playing, lr: m.lr, batchSize: m.batchSize, traceEvery: m.traceEvery, device: m.device };
        if (!this.prompt) this.prompt = m.probe;
        break;
      }
      case "status": {
        const { type: _, ...s } = m;
        this.status = s;
        break;
      }
      case "trace": {
        // Training re-traces the probe prompt; if the learner is looking at their own prompt,
        // re-run it so the view stays on what they typed, now with the new weights.
        this.trace = m.trace;
        if (m.trace.reason === "probe" && this.prompt && m.trace.prompt !== this.prompt) this.run();
        if (this.status) this.status.step = m.trace.step;
        break;
      }
      case "weights":
        this.weights = m.weights;
        break;
      case "stepExplain":
        this.stepExplain = m.explain;
        break;
      case "metrics": {
        this.lossPoints = [...this.lossPoints, ...m.points].slice(-MAX_POINTS);
        if (m.valLoss !== undefined) this.valPoints = [...this.valPoints, [m.step, m.valLoss]];
        this.tokensPerSec = m.tokensPerSec;
        if (this.status) this.status.step = m.step;
        break;
      }
      case "sample":
        if (m.auto) this.autoSample = { step: m.step, text: m.text };
        else this.sample = { text: m.text, tokens: m.tokens ?? [], chosenProbs: m.chosenProbs ?? [] };
        break;
      case "reset":
        this.lossPoints = [];
        this.valPoints = [];
        this.autoSample = null;
        break;
      case "error":
        this.error = m.message;
        setTimeout(() => (this.error = ""), 6000);
        break;
    }
  }
}

export const app = new AppState();

class Selection {
  stageId = $state("embed.sum");
  phase = $state<"forward" | "backward">("forward");
  position = $state(-1);   // -1: none selected
  dim = $state(-1);
  layer = $state(0);       // the head the attention views show; follows any attention stage selected
  head = $state(0);

  stage(trace: Trace | null): Stage | undefined {
    return trace?.byId.get(this.stageId);
  }

  select(stageId: string, phase?: "forward" | "backward"): void {
    this.stageId = stageId;
    if (phase) this.phase = phase;
    const h = headOf(stageId);
    if (h) {
      this.layer = h.layer;
      this.head = h.head;
    }
  }

  selectHead(layer: number, head: number): void {
    this.select(`L${layer}.attn.H${head}.weights`);
  }

  /** The selected position, or the last one if none is selected. */
  query(trace: Trace): number {
    return this.position >= 0 && this.position < trace.tokens.length ? this.position : trace.tokens.length - 1;
  }
}

export const selection = new Selection();

class Settings {
  scaleMode = $state<ScaleMode>("tensor");
  clip = $state(99);
  symlog = $state(false);
  cell = $state(0);   // 0: fit to width
  tab = $state<"inspector" | "attention" | "onestep">("inspector");
  arcsAllHeads = $state(false);
  arcsTint = $state(false);
  qkSort = $state<"position" | "weight">("position");
}

export const settings = new Settings();
