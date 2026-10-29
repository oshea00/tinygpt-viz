// Low-rank adapters (core/lora.py) in the browser: a new adapter, the stories cut into windows,
// and the measures the fine-tuning tab charts while one trains.

import { Tensor } from "./autograd";
import type { Engine } from "./engine";
import { nextLogits } from "./analyses";
import { mulberry32 } from "./interventions";
import { LORA_TARGETS, type ModelConfig, type Weights } from "./model";
import type { BPETokenizer } from "./tokenizer";
import { normals } from "./trainer";

export const DEFAULT_TARGETS = ["q", "v"];
export const DEFAULT_RANK = 4;
/** core/lora.py's OPENERS: they end where a noun goes. */
export const OPENERS = [
  "Once upon a time, there was a",
  "One day, Tom went to the park and saw a",
  "Lily looked up at the sky and saw a big",
  "In the forest lived a",
  "Under the old bridge, there was a",
  "Max wanted to be friends with the",
  "The little girl had a toy",
  "Far away, on top of a hill, lived a",
];

/** Short target names ("q", "mlp.down") in every layer, or stage ids already ("L3.q"). */
export function stageIds(cfg: ModelConfig, targets: string[]): string[] {
  return targets.flatMap((t) => {
    if (t in LORA_TARGETS) return Array.from({ length: cfg.n_layer }, (_, l) => `L${l}.${t}`);
    if (t.split(".").slice(1).join(".") in LORA_TARGETS) return [t];
    throw new Error(`an adapter can't attach to ${JSON.stringify(t)}`);
  });
}

/** A ~ N(0, 1/d_in), B = 0 (a no-op until it trains), as core/lora.py's new_adapter. */
export function newAdapter(cfg: ModelConfig, W: Weights, targets: string[], rank: number, seed: number): Weights {
  const randn = normals(mulberry32(seed));
  const out: Weights = new Map();
  for (const id of stageIds(cfg, targets)) {
    const [l, short] = [id.slice(0, id.indexOf(".")), id.slice(id.indexOf(".") + 1)];
    const [dIn, dOut] = W.get(`${l}.${LORA_TARGETS[short]}`)!.shape;
    out.set(`${id}.lora_A`, new Tensor(Float32Array.from({ length: dIn * rank }, () => randn() / Math.sqrt(dIn)), [dIn, rank]));
    out.set(`${id}.lora_B`, new Tensor(new Float32Array(rank * dOut), [rank, dOut]));
  }
  return out;
}

/** The adapter with B multiplied by `scale` (the engine's tensors fold the strength into B). */
export function scaled(A: Weights, scale: number): Weights {
  if (scale === 1) return A;
  return new Map([...A].map(([n, t]) => [n, n.endsWith(".lora_B") ? new Tensor(t.data.map((v) => v * scale), t.shape) : t]));
}

/** Each story as in training (between end-of-text tokens), in windows of block_size + 1. */
export function windows(tok: BPETokenizer, texts: string[], T: number): number[][] {
  const out: number[][] = [];
  for (const text of texts) {
    const ids = [tok.eosId, ...tok.encode(text.trim()), tok.eosId];
    for (let s = 0; s < Math.max(1, ids.length - 1); s += T) {
      const w = ids.slice(s, s + T + 1);
      if (w.length >= 2) out.push(w);
    }
  }
  return out;
}

/** Inputs and targets, flat, padded to the longest window; padded targets are -1 (ignored). */
export function batchOf(ws: number[][]): { x: number[]; y: number[]; B: number } {
  const T = Math.max(...ws.map((w) => w.length)) - 1, x: number[] = [], y: number[] = [];
  for (const w of ws) for (let t = 0; t < T; t++) {
    x.push(t < w.length - 1 ? w[t] : 0);
    y.push(t < w.length - 1 ? w[t + 1] : -1);
  }
  return { x, y, B: ws.length };
}

/** Mean loss per predicted token over windows (in chunks, each the loss's own mean). */
export async function meanLoss(e: Engine, ws: number[][], adapter: string | undefined, chunk = 16): Promise<number> {
  let total = 0, n = 0;
  for (let i = 0; i < ws.length; i += chunk) {
    const b = batchOf(ws.slice(i, i + chunk)), k = b.y.filter((t) => t >= 0).length;
    total += (await e.loss(b.x, b.y, b.B, adapter)) * k;
    n += k;
  }
  return total / Math.max(n, 1);
}

/** The mean probability of a word as the next token after each opener. */
export async function pWord(e: Engine, cfg: ModelConfig, tok: BPETokenizer, openers: string[], word: number): Promise<number> {
  let sum = 0;
  for (const text of openers) {
    const z = await nextLogits(e, cfg, tok.encode(text).slice(-cfg.block_size));
    let max = -Infinity, s = 0;
    for (const v of z) max = Math.max(max, v);
    for (const v of z) s += Math.exp(v - max);
    sum += Math.exp(z[word] - max) / s;
  }
  return sum / openers.length;
}

export const topicId = (tok: BPETokenizer, word: string) => tok.encode(" " + word.trim())[0];

/** Random distinct picks (Python's random.sample is not reproduced: the seeds differ anyway). */
export function pick<T>(xs: T[], k: number, rand: () => number): T[] {
  const a = [...xs];
  for (let i = 0; i < Math.min(k, a.length); i++) {
    const j = i + Math.floor(rand() * (a.length - i));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, k);
}

/** log π(answer | prompt): the answer tokens' log-probabilities, summed (core/dpo.py's log_probs). */
export async function answerLogProb(e: Engine, cfg: ModelConfig, tok: BPETokenizer, prompt: string, answer: string): Promise<number> {
  const b = tok.encode(answer), ids = [...tok.encode(prompt), ...b].slice(-(cfg.block_size + 1));
  const x = ids.slice(0, -1), n = Math.min(b.length, x.length), V = cfg.vocab_size;
  const { z } = await e.forward(x, { read: (live, op) => ({ z: op.rows(live.get("logits")!, x.length - n, n) }) });
  let total = 0;
  for (let i = 0; i < n; i++) {
    const row = z.subarray(i * V, (i + 1) * V);
    let max = -Infinity, s = 0;
    for (const v of row) max = Math.max(max, v);
    for (const v of row) s += Math.exp(v - max);
    total += row[ids[x.length - n + i + 1]] - max - Math.log(s);
  }
  return total;
}

export interface Pair { prompt: string; chosen: string; rejected: string }

/** Each pair's answers scored with the adapter (`policy`, an engine with it on) and without
 * (`ref`), and the implicit reward margins: core/dpo.py's score. */
export async function scorePairs(ref: Engine, policy: Engine, cfg: ModelConfig, tok: BPETokenizer, pairs: Pair[], beta: number) {
  const out = { ref_chosen: [] as number[], ref_rejected: [] as number[], policy_chosen: [] as number[], policy_rejected: [] as number[],
                margin: [] as number[] };
  for (const p of pairs) {
    for (const key of ["chosen", "rejected"] as const) {
      out[`ref_${key}`].push(await answerLogProb(ref, cfg, tok, p.prompt, p[key]));
      out[`policy_${key}`].push(await answerLogProb(policy, cfg, tok, p.prompt, p[key]));
    }
    const i = out.margin.length;
    out.margin.push(beta * ((out.policy_chosen[i] - out.ref_chosen[i]) - (out.policy_rejected[i] - out.ref_rejected[i])));
  }
  return { ...out, beta };
}
