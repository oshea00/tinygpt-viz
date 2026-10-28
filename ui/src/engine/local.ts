// The in-browser backend: answers the UI's commands as core/scheduler.py does, from the model
// that core/web_export.py writes. It runs in a Web Worker (./worker.ts); this class has no
// worker APIs in it, so the tests can drive it directly.

import { decodeFrame } from "../bridge";
import { Tensor } from "./autograd";
import * as an from "./analyses";
import { CpuEngine, using, type Engine, type WeightSet } from "./engine";
import { encodeTensors, encodeTrace } from "./frames";
import { emptyCache, traceCached } from "./kvcache";
import { GpuModel, type Gpu } from "./gpu";
import { compare as manualCompare } from "./manualBackward";
import type { ModelConfig, Weights } from "./model";
import { BPETokenizer } from "./tokenizer";
import { mulberry32 } from "./interventions";
import { batch as batchOf, checkpointSchedule, copyWeights, initWeights, lrAt, TRAIN_DEFAULTS } from "./trainer";
import type { Intervention } from "../types";

export type Emit = (msg: object | ArrayBuffer) => void;

export interface ModelMeta {
  config: ModelConfig & { name: string; d_mlp: number };
  params: number;
  paramGroups: Record<string, number>;
  checkpoints: number[];
  probe: string;
  stages: ({ id: string; kind: string; inputs: string[]; weights: string[] } & Record<string, unknown>)[];
}

export interface Assets {
  meta: ModelMeta;
  tokenizer: BPETokenizer;
  checkpoint: ArrayBuffer;     // a "weights" frame
  /** Another file of the export, fetched when first needed (val.bin, other checkpoints). */
  fetch(path: string): Promise<ArrayBuffer>;
}

export const checkpointFile = (step: number) => `ckpt_${String(step).padStart(6, "0")}.bin`;

/** Fetch the exported model from `base` (a URL ending in "/"), latest checkpoint first. */
export async function fetchAssets(base: string): Promise<Assets> {
  const get = async (path: string) => {
    const r = await fetch(new URL(path, base));
    if (!r.ok) throw new Error(`${path}: ${r.status} ${r.statusText}`);
    return r;
  };
  const meta: ModelMeta = await (await get("model.json")).json();
  const [tok, checkpoint] = await Promise.all([
    get("tokenizer.json").then((r) => r.json()),
    get(checkpointFile(meta.checkpoints[meta.checkpoints.length - 1])).then((r) => r.arrayBuffer()),
  ]);
  return { meta, tokenizer: new BPETokenizer(tok), checkpoint, fetch: (path) => get(path).then((r) => r.arrayBuffer()) };
}

/** A weights frame's tensors. */
function weightsOf(frame: ArrayBuffer): { step: number; W: Weights } {
  const f = decodeFrame(frame);
  return {
    step: f.header.step,
    W: new Map(Object.entries(f.header.weights as Record<string, { shape: number[]; offset: number }>)
      .map(([n, ref]) => [n, new Tensor(f.tensor(ref).data, ref.shape)])),
  };
}

type Cmd = { type: string } & Record<string, any>;

// Commands that change the weights or the saved checkpoints: only with training on (?train=1, as
// the server's --train).
const TRAINING_COMMANDS = new Set(["train", "set", "loadCheckpoint", "stepExplain", "save"]);
const METRICS_INTERVAL = 100;    // ms between metrics messages while playing
const SAMPLE_EVERY = 500;        // steps between the training panel's automatic samples
const MAX_SNAPSHOTS = 8;         // checkpoints saved this session, kept in memory (23 MB each)

export class LocalModel {
  private cfg: ModelMeta["config"];
  private W: Weights = new Map();        // the host's copy of the weights, current when !hostStale
  private hostStale = false;             // trained since: the engine's weights are newer
  private step = 0;
  private stageMeta: Map<string, ModelMeta["stages"][number]>;
  private engine: Engine;
  private val: Uint16Array | null = null;
  private kvCache = emptyCache();
  private probe: string;
  private weightGrads = true;            // traces carry the weights' gradients (the "view" command)

  // Training (with `training` on).
  private trainCfg = { ...TRAIN_DEFAULTS };
  private weightsVersion = 0;            // bumps whenever the weights change
  private sentWeightsVersion = -1;
  playing = false;
  private trainData: Uint16Array | null = null;
  private rand = mulberry32(TRAIN_DEFAULTS.seed);
  private schedule = checkpointSchedule();
  private snapshots = new Map<number, Weights>();
  private points: [number, number][] = [];
  private lastMetrics = 0;
  private lastTps = 0;

  /** With `gpu`, the model runs on WebGPU; without, on the CPU. `training`: allow the training
   * commands, as the server's --train. */
  constructor(private assets: Assets, private emit: Emit, gpu: Gpu | null = null, private training = false) {
    this.cfg = assets.meta.config;
    this.probe = assets.meta.probe;
    this.stageMeta = new Map(assets.meta.stages.map((s) => [s.id, s]));
    this.engine = gpu ? new GpuModel(gpu, this.cfg) : new CpuEngine(this.cfg);
    ({ step: this.step, W: this.W } = weightsOf(assets.checkpoint));
    this.engine.setWeights(this.W);
  }

  /** What a new connection gets from the server: hello, then the probe's trace. */
  start(): Promise<void> {
    return this.handle([{ type: "hello" }, { type: "run", prompt: this.probe }]);
  }

  /** Commands in arrival order; of consecutive "run"s (per tag) only the last matters. */
  async handle(cmds: Cmd[]): Promise<void> {
    const out: Cmd[] = [];
    for (const c of cmds) {
      const last = out[out.length - 1];
      if (last && c.type === "run" && last.type === "run" && c.tag === last.tag) out[out.length - 1] = c;
      else out.push(c);
    }
    for (const c of out) {
      try {
        await this.dispatch(c);
      } catch (e) {
        this.emit({ type: "error", message: `${c.type}: ${e instanceof Error ? e.message : e}` });
      }
    }
  }

  private checkpoints(): number[] {
    return [...new Set([...this.assets.meta.checkpoints, ...this.snapshots.keys()])].sort((a, b) => a - b);
  }

  private status() {
    const c = this.trainCfg;
    return { step: this.step, playing: this.playing, lr: c.lr, batchSize: c.batchSize, traceEvery: c.traceEvery,
             device: this.engine.name, checkpoints: this.checkpoints() };
  }

  /** The host's copy of the current weights, read back from the engine after training. */
  private async hostWeights(): Promise<Weights> {
    if (this.hostStale) {
      this.W = copyWeights(await this.engine.readWeights());
      this.hostStale = false;
    }
    return this.W;
  }

  /** Token ids of text, never empty (as the server: an empty prompt runs one token 0). */
  private ids(text: string): number[] {
    const ids = this.assets.tokenizer.encode(text);
    return ids.length ? ids : [0];
  }

  private async validation(): Promise<Uint16Array> {
    return (this.val ??= new Uint16Array(await this.assets.fetch("val.bin")));
  }

  /** The engine's weight set for a command's model ("current" or "untrained": step 0). */
  private async weights(model: string | undefined): Promise<WeightSet> {
    if (model !== "untrained") return "current";
    if (!this.engine.hasWeights("untrained")) {
      if (!this.assets.meta.checkpoints.includes(0)) throw new Error("no step-0 checkpoint to compare with");
      this.engine.setWeights(weightsOf(await this.assets.fetch(checkpointFile(0))).W, "untrained");
    }
    return "untrained";
  }

  private async dispatch(cmd: Cmd): Promise<void> {
    const { meta, tokenizer: tok } = this.assets;
    const cfg = this.cfg, e = this.engine;
    if (TRAINING_COMMANDS.has(cmd.type)) {
      if (!this.training) throw new Error("training is off: open the page with ?train=1 to train, replay or save");
      return this.dispatchTraining(cmd);
    }
    switch (cmd.type) {
      case "hello":
        this.emit({ type: "hello", config: meta.config, vocab: Array.from({ length: tok.vocabSize }, (_, i) => tok.tokenText(i)),
                    params: meta.params, paramGroups: meta.paramGroups, probe: this.probe, training: this.training, ...this.status() });
        return;
      case "run":
        if (cmd.incremental && !cmd.interventions?.length && !cmd.tag) return this.sendCachedTrace(cmd.prompt);
        await this.sendTrace(cmd.prompt, cmd.incremental ? false : cmd.withGrad ?? true, cmd.targetIndex ?? null, cmd.tag,
                             cmd.interventions ?? []);
        return;
      case "view":
        // What the open views need: the weights' gradients only for the atlas's gradient mode.
        this.weightGrads = cmd.weightGrads !== false;
        return;
      case "lens":
        this.emit({ type: "lens", prompt: cmd.prompt, step: this.step, ...(await an.logitLens(e, cfg, this.ids(cmd.prompt))) });
        return;
      case "manualBackward": {
        const prompt = cmd.prompt ?? this.probe;
        const t = await e.trace(this.ids(prompt), {}, this.frameSpec(prompt, []));
        if (!t.stages.some((s) => s.grad)) throw new Error("the prompt needs at least two tokens to have a loss");
        this.emit({ type: "manualBackward", prompt, step: this.step, ...manualCompare(t, this.stageMeta, await this.hostWeights()) });
        return;
      }
      case "sample": {
        const prompt = cmd.prompt ?? this.probe;
        const settings = { temperature: cmd.temperature ?? 1, topK: cmd.topK ?? null, topP: cmd.topP ?? null };
        const r = await an.sample(e, cfg, this.ids(prompt), Number(cmd.n ?? 1), settings, Number(cmd.seed ?? Date.now() % 2 ** 31));
        this.emit({ type: "sample", step: this.step, prompt, text: tok.decode(r.out), tokens: r.out.map((i) => tok.tokenText(i)),
                    chosenProbs: r.chosen, probs: r.probs });
        return;
      }
      case "generateWith":
        this.emit({ type: "generated", tag: cmd.tag, prompt: cmd.prompt,
                    text: await an.generate(e, cfg, tok, cmd.prompt, Number(cmd.n ?? 40), cmd.interventions,
                                            Number(cmd.temperature ?? 0), Number(cmd.seed ?? 0)) });
        return;
      case "occlude": {
        const ids = tok.encode(cmd.prompt).slice(-cfg.block_size), pos = Number(cmd.position);
        if (!(pos >= 0 && pos < ids.length - 1)) throw new Error("pick a position that has a next token");
        this.emit({ type: "occlusion", prompt: cmd.prompt, position: pos, step: this.step, drops: await an.occlusion(e, cfg, ids, pos) });
        return;
      }
      case "patch": {
        const what = cmd.what ?? "resid";
        const first = (text: string) => this.ids(text)[0];     // an answer is read at its first token
        const answer = first(cmd.answer), wrong = first(cmd.wrong);
        const clean = tok.encode(cmd.clean), corrupt = tok.encode(cmd.corrupt);
        const r = await an.patchSweep(e, cfg, clean, corrupt, answer, wrong, what);
        const text = (ids: number[]) => ids.slice(-cfg.block_size).map((i) => tok.tokenText(i));
        this.emit(encodeTensors("patch", { tensors: new Map([["grid", r.grid], ["source", r.source]]) }, {
          clean: cmd.clean, corrupt: cmd.corrupt, what, stages: r.stages, cleanText: text(clean), corruptText: text(corrupt),
          answer: tok.tokenText(answer), wrong: tok.tokenText(wrong), ldClean: r.ldClean, ldCorrupt: r.ldCorrupt, step: this.step,
        }));
        return;
      }
      case "neuronTop":
        this.emit({ type: "neuronTop", layer: cmd.layer, neuron: cmd.neuron, step: this.step,
                    examples: await an.neuronTop(e, cfg, tok, await this.validation(), Number(cmd.layer), Number(cmd.neuron)) });
        return;
      case "capture": {
        const weights = await this.weights(cmd.model);
        const r = await an.capture(using(e, weights), cfg, tok, cmd.prompts, cmd.pool ?? "last");
        this.emit(encodeTensors("capture", { vectors: new Map([["x", r.vectors]]) }, {
          items: r.items, snapshots: r.snapshots, pool: cmd.pool ?? "last", model: cmd.model ?? "current",
          step: weights === "untrained" ? 0 : this.step, tag: cmd.tag,
        }));
        return;
      }
      case "beliefs": {
        const weights = await this.weights(cmd.model);
        const stories: string[] = cmd.stories ?? an.storySet(tok, await this.validation(), Number(cmd.n ?? 100));
        const r = await an.beliefs(using(e, weights), cfg, tok, stories, cmd.cloze, cmd.words);
        this.emit(encodeTensors("beliefs", { tensors: new Map([["vectors", r.vectors], ["readout", r.readout], ["mass", r.mass]]) }, {
          stories, prefixes: r.prefixes, snapshots: r.snapshots, cloze: cmd.cloze, words: cmd.words,
          model: cmd.model ?? "current", step: weights === "untrained" ? 0 : this.step,
        }));
        return;
      }
      case "beliefSteer": {
        const r = await an.beliefs(e, cfg, tok, cmd.stories, cmd.cloze, cmd.words, cmd.interventions, false);
        this.emit({ type: "beliefSteer", tag: cmd.tag, readout: nested(r.readout.data, cmd.words.length), mass: Array.from(r.mass.data) });
        return;
      }
      default:
        throw new Error(`unknown command ${JSON.stringify(cmd.type)}`);
    }
  }

  private frameSpec(prompt: string, interventions: Intervention[], tag?: string, reason = "run") {
    const tok = this.assets.tokenizer;
    return {
      meta: (id: string) => this.stageMeta.get(id)!, step: this.step, text: (t: number[]) => t.map((i) => tok.tokenText(i)),
      extra: { prompt, reason, weightsVersion: this.weightsVersion, interventions, ...(tag ? { tag } : {}) },
    };
  }

  /** Generation with the KV cache: only positions not already cached are computed. */
  private async sendCachedTrace(prompt: string): Promise<void> {
    await this.sendWeightsIfNeeded();
    const tok = this.assets.tokenizer;
    const { result, info } = await traceCached(this.engine, this.cfg, this.kvCache, this.ids(prompt), this.weightsVersion,
                                               (id) => this.stageMeta.get(id)?.kind ?? "");
    this.emit(encodeTrace(result, (id) => this.stageMeta.get(id)!, this.step, result.tokens.map((i) => tok.tokenText(i)),
                          { prompt, reason: "run", weightsVersion: this.weightsVersion, interventions: [], incremental: info }));
  }

  private async sendWeightsIfNeeded(): Promise<void> {
    if (this.sentWeightsVersion === this.weightsVersion) return;
    this.emit(encodeTensors("weights", { weights: await this.hostWeights() }, { step: this.step, version: this.weightsVersion }));
    this.sentWeightsVersion = this.weightsVersion;
  }

  private async sendTrace(prompt: string, withGrad: boolean, targetIndex: number | null, tag?: string,
                          interventions: Intervention[] = [], reason = "run"): Promise<void> {
    await this.sendWeightsIfNeeded();
    const t = await this.engine.trace(this.ids(prompt), { withGrad, targetIndex, interventions, weightGrads: this.weightGrads },
                                      this.frameSpec(prompt, interventions, tag, reason));
    this.emit(t.frame);
  }

  private sendProbe(reason = "probe"): Promise<void> {
    return this.sendTrace(this.probe, true, null, undefined, [], reason);
  }

  // -- training (core/scheduler.py's training commands, core/train.py's Trainer) ----------------

  private async dispatchTraining(cmd: Cmd): Promise<void> {
    const c = this.trainCfg;
    switch (cmd.type) {
      case "train":
        if (cmd.action === "play") {
          await this.trainingData();
          this.playing = true;
        } else if (cmd.action === "pause") {
          this.playing = false;
          this.flushMetrics(true);
        } else if (cmd.action === "step") {
          this.playing = false;
          await this.trainStep();
          this.flushMetrics(true);
          await this.sendProbe();
        } else if (cmd.action === "reset") {
          this.playing = false;
          const seed = cmd.seed ?? c.seed;
          this.rand = mulberry32(seed);
          this.setWeights(initWeights(this.cfg, seed), 0);
          this.points = [];
          this.emit({ type: "reset" });
          await this.sendProbe();
        }
        break;
      case "set":
        if ("lr" in cmd) c.lr = Number(cmd.lr);
        if ("batchSize" in cmd) c.batchSize = Number(cmd.batchSize);
        if ("traceEvery" in cmd) c.traceEvery = Math.max(1, Number(cmd.traceEvery));
        if ("probe" in cmd) this.probe = cmd.probe;
        break;
      case "loadCheckpoint": {
        this.playing = false;
        // Replaying must not lose training: keep the current state as a checkpoint first.
        if (!this.checkpoints().includes(this.step)) await this.snapshot();
        const step = cmd.step ?? this.checkpoints().at(-1)!;
        const W = this.snapshots.get(step) ?? weightsOf(await this.assets.fetch(checkpointFile(step))).W;
        this.setWeights(copyWeights(W), step);
        await this.sendProbe();
        break;
      }
      case "stepExplain":
        this.playing = false;
        await this.stepExplain(cmd.prompt ?? this.probe);
        break;
      case "save":
        await this.snapshot();
        break;
    }
    this.emit({ type: "status", ...this.status() });
  }

  /** New weights at `step` (a reset or a checkpoint), with fresh optimizer moments. */
  private setWeights(W: Weights, step: number): void {
    this.engine.setWeights(W);
    this.engine.resetOptimizer();
    this.W = W;
    this.hostStale = false;
    this.step = step;
    this.weightsVersion++;
  }

  private async trainingData(): Promise<Uint16Array> {
    return (this.trainData ??= new Uint16Array(await this.assets.fetch("train.bin")));
  }

  /** The current weights as a checkpoint of this session, in memory (the oldest go first). */
  private async snapshot(): Promise<void> {
    this.snapshots.set(this.step, copyWeights(await this.hostWeights()));
    while (this.snapshots.size > MAX_SNAPSHOTS) {
      this.snapshots.delete([...this.snapshots.keys()].filter((s) => s !== 0).sort((a, b) => a - b)[0]);
    }
  }

  /** One step on a batch from train.bin (or the given one), as Trainer.step. */
  private async trainStep(batch?: { x: number[]; y: number[]; B: number }) {
    const c = this.trainCfg, T = this.cfg.block_size, t0 = performance.now();
    const b = batch ?? { ...batchOf(await this.trainingData(), c.batchSize, T, this.rand), B: c.batchSize };
    const r = await this.engine.trainStep(b.x, b.y, b.B, { lr: lrAt(c, this.step), weightDecay: c.weightDecay, gradClip: c.gradClip });
    this.step++;
    this.weightsVersion++;
    this.hostStale = true;
    this.points.push([this.step, r.loss]);
    this.lastTps = b.x.length / ((performance.now() - t0) / 1000);
    if (this.schedule.has(this.step)) await this.snapshot();
    return { ...r, lr: lrAt(c, this.step) };
  }

  /** While playing: one step, then what the server sends along with it. */
  async tick(): Promise<void> {
    if (!this.playing) return;
    try {
      const c = this.trainCfg;
      await this.trainStep();
      const s = this.step;
      const val = s % c.evalEvery === 0 ? await this.evalVal() : undefined;
      this.flushMetrics(val !== undefined, val);
      if (s % c.traceEvery === 0) await this.sendProbe();
      if (s % SAMPLE_EVERY === 0) {
        const tok = this.assets.tokenizer;
        const r = await an.sample(this.engine, this.cfg, this.ids(this.probe), 200, { temperature: 0.8, topK: null, topP: null }, s);
        this.emit({ type: "sample", step: s, text: this.probe + tok.decode(r.out), auto: true });
      }
    } catch (e) {
      this.playing = false;
      this.emit({ type: "error", message: `train: ${e instanceof Error ? e.message : e}` });
      this.emit({ type: "status", ...this.status() });
    }
  }

  /** The validation loss on the same batches every time, so the curve is comparable. */
  private async evalVal(): Promise<number> {
    const c = this.trainCfg, val = await this.validation(), rand = mulberry32(0);
    let total = 0;
    for (let i = 0; i < c.evalBatches; i++) {
      const b = batchOf(val, c.batchSize, this.cfg.block_size, rand);
      total += await this.engine.loss(b.x, b.y, c.batchSize);
    }
    return total / c.evalBatches;
  }

  private flushMetrics(force: boolean, valLoss?: number): void {
    const now = performance.now();
    if (!this.points.length && valLoss === undefined) return;
    if (force || now - this.lastMetrics >= METRICS_INTERVAL) {
      this.emit({ type: "metrics", step: this.step, points: this.points, tokensPerSec: this.lastTps,
                  ...(valLoss !== undefined ? { valLoss } : {}) });
      this.points = [];
      this.lastMetrics = now;
    }
  }

  /** One training step on the prompt itself, with what the training-step tab shows: loss and
   * p(correct next token) before and after, and every weight's value, gradient and change. */
  private async stepExplain(prompt: string): Promise<void> {
    const tok = this.assets.tokenizer, V = this.cfg.vocab_size;
    const ids = tok.encode(prompt).slice(-(this.cfg.block_size + 1));
    if (ids.length < 2) throw new Error("prompt needs at least two tokens");
    const x = ids.slice(0, -1), y = ids.slice(1);
    const pCorrect = async () => {
      const { p } = await this.engine.forward(x, { read: (live) => ({ p: live.get("probs")! }) });
      return y.map((t, i) => p[i * V + t]);
    };
    const loss = (ps: number[]) => -ps.reduce((a, p) => a + Math.log(Math.max(p, 1e-12)), 0) / ps.length;
    const pBefore = await pCorrect();
    const before = copyWeights(await this.hostWeights());
    const m = await this.trainStep({ x, y, B: 1 });
    const grad = await this.engine.readGrads();
    const after = await this.hostWeights();
    const delta = new Map([...after].map(([n, w]) => [n, { data: w.data.map((v, i) => v - before.get(n)!.data[i]), shape: w.shape }]));
    const pAfter = await pCorrect();
    this.emit(encodeTensors("stepExplain", { before, grad, delta }, {
      step: this.step, prompt, text: ids.map((i) => tok.tokenText(i)),
      lossBefore: loss(pBefore), lossAfter: loss(pAfter), pBefore, pAfter, lr: m.lr,
    }));
    this.flushMetrics(true);
    await this.sendTrace(prompt, true, null, undefined, [], "stepExplain");
  }
}

/** A flat [rows · k] array as rows of k. */
function nested(a: Float32Array, k: number): number[][] {
  return Array.from({ length: a.length / k }, (_, i) => Array.from(a.subarray(i * k, (i + 1) * k)));
}
