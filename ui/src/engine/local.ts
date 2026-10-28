// The in-browser backend: answers the UI's commands as core/scheduler.py does, from the model
// that core/web_export.py writes. It runs in a Web Worker (./worker.ts); this class has no
// worker APIs in it, so the tests can drive it directly.

import { decodeFrame } from "../bridge";
import { Tensor } from "./autograd";
import * as an from "./analyses";
import { CpuEngine, using, type Engine, type WeightSet } from "./engine";
import { encodeTensors } from "./frames";
import { GpuModel, type Gpu } from "./gpu";
import { compare as manualCompare } from "./manualBackward";
import type { ModelConfig, Weights } from "./model";
import { BPETokenizer } from "./tokenizer";
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

// Commands that change the weights or the saved checkpoints: the server only has them with --train.
const TRAINING_COMMANDS = new Set(["train", "set", "loadCheckpoint", "stepExplain", "save"]);

export class LocalModel {
  private cfg: ModelConfig;
  private W: Weights = new Map();
  private step = 0;
  private stageMeta: Map<string, ModelMeta["stages"][number]>;
  private weightsSent = false;
  private engine: Engine;
  private val: Uint16Array | null = null;

  /** With `gpu`, the model runs on WebGPU; without, on the CPU. */
  constructor(private assets: Assets, private emit: Emit, gpu: Gpu | null = null) {
    this.cfg = assets.meta.config;
    this.stageMeta = new Map(assets.meta.stages.map((s) => [s.id, s]));
    this.engine = gpu ? new GpuModel(gpu, this.cfg) : new CpuEngine(this.cfg);
    ({ step: this.step, W: this.W } = weightsOf(assets.checkpoint));
    this.engine.setWeights(this.W);
  }

  /** What a new connection gets from the server: hello, then the probe's trace. */
  start(): Promise<void> {
    return this.handle([{ type: "hello" }, { type: "run", prompt: this.assets.meta.probe }]);
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

  private status() {
    return { step: this.step, playing: false, lr: 0, batchSize: 0, traceEvery: 0, device: this.engine.name,
             checkpoints: this.assets.meta.checkpoints };
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
      throw new Error("training is off: the in-browser backend runs the model but does not train it yet");
    }
    switch (cmd.type) {
      case "hello":
        this.emit({ type: "hello", config: meta.config, vocab: Array.from({ length: tok.vocabSize }, (_, i) => tok.tokenText(i)),
                    params: meta.params, paramGroups: meta.paramGroups, probe: meta.probe, training: false, ...this.status() });
        return;
      case "run":
        await this.sendTrace(cmd.prompt, cmd.incremental ? false : cmd.withGrad ?? true, cmd.targetIndex ?? null, cmd.tag,
                             cmd.interventions ?? []);
        return;
      case "lens":
        this.emit({ type: "lens", prompt: cmd.prompt, step: this.step, ...(await an.logitLens(e, cfg, this.ids(cmd.prompt))) });
        return;
      case "manualBackward": {
        const prompt = cmd.prompt ?? meta.probe;
        const t = await e.trace(this.ids(prompt), {}, this.frameSpec(prompt, []));
        if (!t.stages.some((s) => s.grad)) throw new Error("the prompt needs at least two tokens to have a loss");
        this.emit({ type: "manualBackward", prompt, step: this.step, ...manualCompare(t, this.stageMeta, this.W) });
        return;
      }
      case "sample": {
        const prompt = cmd.prompt ?? meta.probe;
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

  private frameSpec(prompt: string, interventions: Intervention[], tag?: string) {
    const tok = this.assets.tokenizer;
    return {
      meta: (id: string) => this.stageMeta.get(id)!, step: this.step, text: (t: number[]) => t.map((i) => tok.tokenText(i)),
      extra: { prompt, reason: "run", weightsVersion: this.step, interventions, ...(tag ? { tag } : {}) },
    };
  }

  private async sendTrace(prompt: string, withGrad: boolean, targetIndex: number | null, tag?: string,
                          interventions: Intervention[] = []): Promise<void> {
    if (!this.weightsSent) {
      this.emit(encodeTensors("weights", { weights: this.W }, { step: this.step, version: this.step }));
      this.weightsSent = true;
    }
    const t = await this.engine.trace(this.ids(prompt), { withGrad, targetIndex, interventions },
                                      this.frameSpec(prompt, interventions, tag));
    this.emit(t.frame);
  }
}

/** A flat [rows · k] array as rows of k. */
function nested(a: Float32Array, k: number): number[][] {
  return Array.from({ length: a.length / k }, (_, i) => Array.from(a.subarray(i * k, (i + 1) * k)));
}
