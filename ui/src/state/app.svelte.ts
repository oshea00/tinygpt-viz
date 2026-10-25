// Shared state. Every view is a function of (trace, selection, settings); clicking in any
// view updates the selection and every other view follows.

import { Bridge, type BinaryMsg } from "../bridge";
import type { Command, Hello, JsonMsg, Stage, Status, StepExplain, Trace, Weights } from "../types";
import { headOf } from "../lib/attention";
import { pick, shape, type SampleSettings } from "../lib/sampling";
import type { ScaleMode } from "../lib/scale";
import { row } from "../lib/tensor";

const MAX_POINTS = 50_000;
const MAX_GENERATED = 600;

/** Display forms of whitespace and the end-of-story marker, back to the raw character. */
const RAW: Record<string, string> = { "␣": " ", "⏎": "\n", "␄": "\x04" };
export const rawChar = (display: string) => RAW[display] ?? display;

export interface GenToken {
  pos: number;               // index in the prompt string where this character was appended
  token: number;
  p: number;                 // its probability in the distribution it was sampled from
  rank: number;              // 1 = the most likely token
  settings: SampleSettings;
  step: number;              // training step of the weights that produced it
}

class AppState {
  connected = $state(false);
  hello = $state.raw<Hello | null>(null);
  status = $state<Status | null>(null);
  trace = $state.raw<Trace | null>(null);
  weights = $state.raw<Weights | null>(null);
  prevWeights = $state.raw<Weights | null>(null);    // the snapshot before `weights`, for "recent change"
  pinned = $state.raw<{ trace: Trace; weights: Weights } | null>(null);   // reference for atlas diffs
  stepExplain = $state.raw<StepExplain | null>(null);
  lossPoints = $state.raw<[number, number][]>([]);
  valPoints = $state.raw<[number, number][]>([]);
  tokensPerSec = $state(0);
  autoSample = $state<{ step: number; text: string } | null>(null);
  error = $state("");
  prompt = $state("");

  // Generation: sampling happens here, from the logits in the trace, so what the output
  // panel shows is exactly what gets sampled. Each character appended re-runs the model.
  sampling = $state<SampleSettings>({ temperature: 0.8, topK: null, topP: null });
  generated = $state.raw<GenToken[]>([]);
  auto = $state(false);
  autoDelay = $state(120);         // ms between characters in auto mode
  stopAtEnd = $state(true);        // stop auto at the end-of-story marker
  travel = $state<number | null>(null);   // index into `generated` being revisited
  private autoTimer: ReturnType<typeof setTimeout> | undefined;

  private bridge: Bridge | null = null;

  start(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.bridge = new Bridge(`${proto}://${location.host}/ws`, (m) => this.onMessage(m), (c) => (this.connected = c));
    this.bridge.connect();
  }

  send(cmd: Command): void {
    this.bridge?.send(cmd);
  }

  run(prompt = this.viewPrompt()): void {
    this.send({ type: "run", prompt, withGrad: !this.auto });
  }

  /** The prompt whose trace is on screen: the full text, or a prefix when time travelling. */
  viewPrompt(): string {
    return this.travel !== null ? this.prompt.slice(0, this.generated[this.travel].pos) : this.prompt;
  }

  /** The learner edited the prompt: generation so far no longer applies. */
  promptEdited(): void {
    this.stopAuto();
    this.generated = [];
    this.travel = null;
  }

  /** Sample one token from the last row of the current trace and append it. */
  sampleNext(): boolean {
    const t = this.trace;
    if (!t || !this.hello || t.prompt !== this.prompt || this.travel !== null) return false;
    const logits = row(t.byId.get("logits")!.output, t.tokens.length - 1);
    const settings = { ...this.sampling };
    const { probs } = shape(logits, settings);
    const token = pick(probs);
    const rank = 1 + probs.filter((p) => p > probs[token]).length;
    const ch = rawChar(this.hello.vocab[token]);
    this.generated = [...this.generated, { pos: this.prompt.length, token, p: probs[token], rank, settings, step: t.step }];
    this.prompt += ch;
    this.run();
    if (this.auto && ((this.stopAtEnd && ch === "\x04") || this.generated.length >= MAX_GENERATED)) this.stopAuto();
    return true;
  }

  startAuto(): void {
    this.travel = null;
    this.auto = true;
    if (this.trace?.prompt === this.prompt) this.sampleNext();
    else this.run();
  }

  stopAuto(): void {
    clearTimeout(this.autoTimer);
    if (!this.auto) return;
    this.auto = false;
    this.run();   // once more with gradients, so every view is complete again
  }

  /** Show the trace that produced generated token i (null: back to the latest). */
  travelTo(i: number | null): void {
    this.stopAuto();
    this.travel = i;
    this.run();
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
        if (m.trace.reason === "probe" && this.prompt && m.trace.prompt !== this.viewPrompt()) this.run();
        if (this.status) this.status.step = m.trace.step;
        if (this.auto && m.trace.prompt === this.prompt) {
          clearTimeout(this.autoTimer);
          this.autoTimer = setTimeout(() => this.auto && this.sampleNext(), this.autoDelay);
        }
        break;
      }
      case "weights":
        // Each weights message is a fresh copy, so keeping the previous one costs one model's worth of memory.
        if (this.weights && this.weights.version !== m.weights.version) this.prevWeights = this.weights;
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
  tab = $state<"inspector" | "attention" | "residual" | "lens" | "atlas" | "onestep">("inspector");
  atlasMode = $state<"values" | "grads" | "flow" | "diff" | "recent">("values");
  residualMode = $state<"snapshots" | "writes">("snapshots");
  highwayScope = $state<"position" | "all">("all");
  lensMode = $state<"confidence" | "actual">("confidence");
  lensSnap = $state(-1);   // snapshot row picked in the logit lens (-1: last)
  arcsAllHeads = $state(false);
  arcsTint = $state(false);
  qkSort = $state<"position" | "weight">("position");
}

export const settings = new Settings();
