// The in-browser backend: answers the UI's commands as core/scheduler.py does, from the model
// that core/web_export.py writes. It runs in a Web Worker (./worker.ts); this class has no
// worker APIs in it, so the tests can drive it directly.

import { decodeFrame } from "../bridge";
import { Tensor, softmaxRows } from "./autograd";
import { encodeTensors, encodeTrace } from "./frames";
import { trace, type ModelConfig, type Weights } from "./model";
import { BPETokenizer } from "./tokenizer";

export type Emit = (msg: object | ArrayBuffer) => void;

export interface ModelMeta {
  config: ModelConfig & { name: string; d_mlp: number };
  params: number;
  paramGroups: Record<string, number>;
  checkpoints: number[];
  probe: string;
  stages: ({ id: string } & Record<string, unknown>)[];
}

export interface Assets {
  meta: ModelMeta;
  tokenizer: BPETokenizer;
  checkpoint: ArrayBuffer;     // a "weights" frame
}

/** Fetch the exported model from `base` (a URL ending in "/"), latest checkpoint first. */
export async function fetchAssets(base: string): Promise<Assets> {
  const get = async (path: string) => {
    const r = await fetch(new URL(path, base));
    if (!r.ok) throw new Error(`${path}: ${r.status} ${r.statusText}`);
    return r;
  };
  const meta: ModelMeta = await (await get("model.json")).json();
  const step = meta.checkpoints[meta.checkpoints.length - 1];
  const [tok, checkpoint] = await Promise.all([
    get("tokenizer.json").then((r) => r.json()),
    get(`ckpt_${String(step).padStart(6, "0")}.bin`).then((r) => r.arrayBuffer()),
  ]);
  return { meta, tokenizer: new BPETokenizer(tok), checkpoint };
}

type Cmd = { type: string } & Record<string, any>;

export class LocalModel {
  private cfg: ModelConfig;
  private W: Weights = new Map();
  private step = 0;
  private stageMeta: Map<string, object>;
  private weightsSent = false;

  constructor(private assets: Assets, private emit: Emit) {
    this.cfg = assets.meta.config;
    this.stageMeta = new Map(assets.meta.stages.map((s) => [s.id, s]));
    this.loadWeights(assets.checkpoint);
  }

  private loadWeights(frame: ArrayBuffer): void {
    const f = decodeFrame(frame);
    this.step = f.header.step;
    this.W = new Map(Object.entries(f.header.weights as Record<string, { shape: number[]; offset: number }>)
      .map(([n, ref]) => [n, new Tensor(f.tensor(ref).data, ref.shape)]));
    this.weightsSent = false;
  }

  /** What a new connection gets from the server: hello, then the probe's trace. */
  start(): void {
    this.handle([{ type: "hello" }, { type: "run", prompt: this.assets.meta.probe }]);
  }

  /** Commands in arrival order; of consecutive "run"s (per tag) only the last matters. */
  handle(cmds: Cmd[]): void {
    const out: Cmd[] = [];
    for (const c of cmds) {
      const last = out[out.length - 1];
      if (last && c.type === "run" && last.type === "run" && c.tag === last.tag) out[out.length - 1] = c;
      else out.push(c);
    }
    for (const c of out) {
      try {
        this.dispatch(c);
      } catch (e) {
        this.emit({ type: "error", message: `${c.type}: ${e instanceof Error ? e.message : e}` });
      }
    }
  }

  private status() {
    return { step: this.step, playing: false, lr: 0, batchSize: 0, traceEvery: 0, device: "browser (CPU)",
             checkpoints: this.assets.meta.checkpoints };
  }

  private dispatch(cmd: Cmd): void {
    const { meta, tokenizer: tok } = this.assets;
    switch (cmd.type) {
      case "hello":
        this.emit({ type: "hello", config: meta.config, vocab: Array.from({ length: tok.vocabSize }, (_, i) => tok.tokenText(i)),
                    params: meta.params, paramGroups: meta.paramGroups, probe: meta.probe, training: false, ...this.status() });
        return;
      case "run":
        if (cmd.interventions?.length) throw new Error("interventions do not run in the browser yet");
        this.sendTrace(cmd.prompt, cmd.incremental ? false : cmd.withGrad ?? true, cmd.targetIndex ?? null, cmd.tag);
        return;
      case "lens":
        this.emit({ type: "lens", prompt: cmd.prompt, step: this.step, ...this.lens(tok.encode(cmd.prompt)) });
        return;
      default:
        throw new Error("not available in the browser yet");
    }
  }

  private sendTrace(prompt: string, withGrad: boolean, targetIndex: number | null, tag?: string): void {
    if (!this.weightsSent) {
      this.emit(encodeTensors("weights", { weights: this.W }, { step: this.step, version: this.step }));
      this.weightsSent = true;
    }
    const tok = this.assets.tokenizer;
    const ids = tok.encode(prompt);
    const t = trace(this.cfg, this.W, ids.length ? ids : [0], { withGrad, targetIndex });
    this.emit(encodeTrace(t, (id) => this.stageMeta.get(id)!, this.step, t.tokens.map((i) => tok.tokenText(i)), {
      prompt, reason: "run", weightsVersion: this.step, interventions: [], ...(tag ? { tag } : {}),
    }));
  }

  /** The logit lens (core/lens.py): every residual snapshot through ln_f and the unembedding. */
  private lens(ids: number[], k = 8) {
    const t = trace(this.cfg, this.W, ids.length ? ids : [0], { withGrad: false });
    const tokens = t.tokens, T = tokens.length, d = this.cfg.d_model;
    const snaps = t.stages.filter((s) => s.id === "embed.sum" || /resid_(attn|mlp)$/.test(s.id));
    const g = this.W.get("ln_f_g")!.data, b = this.W.get("ln_f_b")!.data;
    const U = this.cfg.tied ? this.W.get("wte")! : this.W.get("W_unembed")!;
    const V = this.cfg.vocab_size;
    const topIds: number[][][] = [], topP: number[][][] = [], pActual: number[][] = [];
    for (const s of snaps) {
      const x = s.output.data, logits = new Float32Array(T * V);
      for (let i = 0; i < T; i++) {
        const r = x.subarray(i * d, (i + 1) * d);
        let mean = 0, v = 0;
        for (const a of r) mean += a;
        mean /= d;
        for (const a of r) v += (a - mean) ** 2;
        const rstd = 1 / Math.sqrt(v / d + 1e-5);
        const h = r.map((a, j) => (a - mean) * rstd * g[j] + b[j]);
        for (let j = 0; j < V; j++) {
          let z = 0;
          if (this.cfg.tied) for (let c = 0; c < d; c++) z += h[c] * U.data[j * d + c];
          else for (let c = 0; c < d; c++) z += h[c] * U.data[c * V + j];
          logits[i * V + j] = z;
        }
      }
      const probs = softmaxRows(logits, T, V);
      const ti: number[][] = [], tp: number[][] = [], pa: number[] = [];
      for (let i = 0; i < T; i++) {
        const row = probs.subarray(i * V, (i + 1) * V);
        const top = [...row.keys()].sort((a, c) => row[c] - row[a]).slice(0, k);
        ti.push(top);
        tp.push(top.map((j) => row[j]));
        pa.push(i < T - 1 ? row[tokens[i + 1]] : -1);
      }
      topIds.push(ti);
      topP.push(tp);
      pActual.push(pa);
    }
    return { snapshots: snaps.map((s) => s.id), topIds, topP, pActual };
  }
}
