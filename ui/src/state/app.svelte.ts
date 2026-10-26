// Shared state. Every view is a function of (trace, selection, settings); clicking in any
// view updates the selection and every other view follows.

import { Bridge, type BinaryMsg } from "../bridge";
import type { Command, Hello, Intervention, JsonMsg, LensMsg, ManualBackwardMsg, Stage, Status, StepExplain, Trace, Weights } from "../types";
import { headOf } from "../lib/attention";
import { pick, shape, type SampleSettings } from "../lib/sampling";
import type { ScaleMode } from "../lib/scale";
import { row } from "../lib/tensor";

const MAX_POINTS = 50_000;
const MAX_GENERATED = 600;

/** Display forms of whitespace and the end-of-story marker, back to the raw character. */
/** A token's display text back to raw text: "␣dog" → " dog". */
export const rawText = (display: string) => display.replace(/␣/g, " ").replace(/⏎/g, "\n").replace(/␄/g, "\x04");

export interface GenToken {
  pos: number;               // index in the prompt string where this token was appended
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
  // panel shows is exactly what gets sampled. Each token appended re-runs the model.
  sampling = $state<SampleSettings>({ temperature: 0.8, topK: null, topP: null });
  generated = $state.raw<GenToken[]>([]);
  auto = $state(false);
  autoDelay = $state(120);         // ms between tokens in auto mode
  stopAtEnd = $state(true);        // stop auto at the end-of-story marker
  travel = $state<number | null>(null);   // index into `generated` being revisited

  // Interventions apply to every trace requested; with any active, an unmodified baseline
  // of the same prompt is fetched too, so every effect can be shown as before / after.
  interventions = $state.raw<Intervention[]>([]);

  // KV cache mode: traces for generation reuse every earlier position and compute only the
  // new one (no gradients). Totals compare its cost with re-running the whole window.
  kvCache = $state(false);
  kvTotals = $state({ withCache: 0, without: 0, steps: 0, resets: 0 });
  baseline = $state.raw<Trace | null>(null);

  serverLens = $state.raw<LensMsg | null>(null);   // logit lens computed on the server
  private lensAsked = "";

  requestLens(): void {
    const t = this.trace;
    if (!t?.prompt) return;
    const key = `${t.prompt}@${t.step}`;
    if (key === this.lensAsked) return;
    this.lensAsked = key;
    this.send({ type: "lens", prompt: t.prompt });
  }

  manual = $state.raw<ManualBackwardMsg | null>(null);   // hand-written backward vs autograd
  private manualAsked = "";

  /** Ask for the hand-written backward check of the current trace, once per prompt and step. */
  requestManual(): void {
    const t = this.trace;
    if (!t?.prompt || this.interventions.length) return;
    const key = `${t.prompt}@${t.step}`;
    if (key === this.manualAsked) return;
    this.manualAsked = key;
    this.send({ type: "manualBackward", prompt: t.prompt });
  }
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
    const interventions = this.interventions;
    const incremental = this.kvCache && !interventions.length && this.travel === null;
    this.send({ type: "run", prompt, withGrad: !this.auto && !incremental, interventions, incremental });
    if (interventions.length) this.send({ type: "run", prompt, withGrad: !this.auto, tag: "baseline" });
    else this.baseline = null;
  }

  setInterventions(list: Intervention[]): void {
    this.interventions = list;
    this.run();
  }

  addIntervention(iv: Intervention): void {
    // One intervention per target: a new one replaces any on the same head / neuron / cell.
    const key = (x: Intervention) => JSON.stringify({ ...x, mode: undefined, value: undefined, position: undefined, seed: undefined });
    this.setInterventions([...this.interventions.filter((x) => key(x) !== key(iv)), iv]);
  }

  /** The prompt whose trace is on screen: the full text, or a prefix when time travelling. */
  viewPrompt(): string {
    return this.travel !== null ? this.prompt.slice(0, this.generated[this.travel].pos) : this.prompt;
  }

  /** The learner edited the prompt: generation so far no longer applies. */
  promptEdited(): void {
    this.kvTotals = { withCache: 0, without: 0, steps: 0, resets: 0 };
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
    const ch = rawText(this.hello.vocab[token]);
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

  /** Feature modules (e.g. the concept explorer) can claim messages before the core handling. */
  readonly handlers: ((m: JsonMsg | BinaryMsg) => boolean)[] = [];

  private onMessage(m: JsonMsg | BinaryMsg): void {
    if (this.handlers.some((h) => h(m))) return;
    switch (m.type) {
      case "hello": {
        this.hello = m;
        this.status = { step: m.step, playing: m.playing, lr: m.lr, batchSize: m.batchSize, traceEvery: m.traceEvery, device: m.device, checkpoints: m.checkpoints };
        if (!this.prompt) this.prompt = m.probe;
        break;
      }
      case "status": {
        const { type: _, ...s } = m;
        this.status = s;
        break;
      }
      case "trace": {
        if (m.trace.tag === "baseline") {
          this.baseline = m.trace;
          break;
        }
        // Training re-traces the probe prompt, without interventions. If the learner is looking
        // at their own prompt or has interventions active, re-run theirs with the new weights.
        if (m.trace.reason === "probe" && this.interventions.length) {
          this.run();
          break;
        }
        this.trace = m.trace;
        const inc = m.trace.incremental;
        if (inc) {
          const k = this.kvTotals;
          this.kvTotals = { withCache: k.withCache + inc.multiplyAdds, without: k.without + inc.multiplyAddsFull,
                            steps: k.steps + 1, resets: k.resets + (inc.reason && k.steps ? 1 : 0) };
        }
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
      case "manualBackward":
        this.manual = m;
        break;
      case "lens":
        this.serverLens = m;
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
  /** A residual-stream dimension followed through every stage that shares the stream's
   * coordinates (see inResidualBasis in lib/embeddings.ts); -1: none. */
  trackDim = $state(-1);
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

  /** Back to nothing selected, as when the app opens. */
  reset(): void {
    this.select("embed.sum", "forward");
    this.position = -1;
    this.dim = -1;
    this.trackDim = -1;
    this.layer = 0;
    this.head = 0;
  }

  /** The selected position, or the last one if none is selected. */
  query(trace: Trace): number {
    return this.position >= 0 && this.position < trace.tokens.length ? this.position : trace.tokens.length - 1;
  }
}

export const selection = new Selection();

class Settings {
  help = $state(false);                    // the help panel
  helpTopic = $state<string | null>(null);   // a panel's help (see help/help.ts PANELS), "glossary"; null: the open tab
  glossaryKey = $state<string | null>(null); // the glossary entry a link pointed to

  openHelp(topic: string | null): void {
    this.helpTopic = topic;
    this.help = true;
  }

  openGlossary(key: string | null): void {
    this.glossaryKey = key;
    this.openHelp("glossary");
  }
  scaleMode = $state<ScaleMode>("tensor");
  clip = $state(99);
  symlog = $state(false);
  cell = $state(3);   // cell size multiplier; 0: fit to width
  tab = $state<"inspector" | "embeddings" | "attention" | "residual" | "lens" | "atlas" | "interventions" | "concepts" | "onestep">("inspector");
  compare = $state(false);   // inspector shows (with interventions − baseline) instead of values
  atlasMode = $state<"values" | "grads" | "flow" | "diff" | "recent">("values");
  atlasView = $state.raw<{ scale: number; tx: number; ty: number } | null>(null);   // kept across tab switches
  residualMode = $state<"snapshots" | "writes">("snapshots");
  trajNormalise = $state(true);   // residual trajectories: scale each snapshot's vectors to unit length
  highwayScope = $state<"position" | "all">("all");
  lensMode = $state<"confidence" | "actual">("confidence");
  lensSnap = $state(-1);   // snapshot row picked in the logit lens (-1: last)
  arcsAllHeads = $state(false);
  arcsTint = $state(false);
  qkSort = $state<"position" | "weight">("position");
}

export const settings = new Settings();
