// The commands beyond a single trace, ported from core/scheduler.py, core/concepts.py,
// core/beliefs.py and core/lens.py, written once against an Engine: sampling, generation with
// interventions, occlusion, activation patching, concept capture, belief trajectories, neuron
// search, and the logit lens.
//
// The server batches prompts; here each one is its own forward pass. Random choices (sampling,
// neuron-search windows) use a seeded mulberry32, so they are repeatable but not the same draws
// as torch's.

import { mulberry32 } from "./interventions";
import type { Node, OpSet } from "./autograd";
import type { Engine } from "./engine";
import type { ModelConfig } from "./model";
import type { BPETokenizer } from "./tokenizer";
import type { Intervention } from "../types";
import { pick, shape, type SampleSettings } from "../lib/sampling";

export function snapshotIds(cfg: ModelConfig): string[] {
  const ids = ["embed.sum"];
  for (let l = 0; l < cfg.n_layer; l++) ids.push(`L${l}.resid_attn`, `L${l}.resid_mlp`);
  return ids;
}

/** log softmax of one row, in float64. */
function logSoftmax(z: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) max = Math.max(max, z[i]);
  let s = 0;
  for (let i = 0; i < z.length; i++) s += Math.exp(z[i] - max);
  const lse = max + Math.log(s);
  return Float64Array.from(z, (v) => v - lse);
}

/** The logits at the last position of `ids` (the window's last block_size tokens). */
export async function nextLogits(e: Engine, cfg: ModelConfig, ids: number[], interventions?: Intervention[]): Promise<Float32Array> {
  const T = Math.min(ids.length, cfg.block_size);
  const r = await e.forward(ids, { interventions, read: (live, op) => ({ z: op.rows(live.get("logits")!, T - 1, 1) }) });
  return r.z;
}

/** Sample n tokens (the "sample" command): the chosen tokens, their probabilities, and the last
 * step's distribution. */
export async function sample(e: Engine, cfg: ModelConfig, ids: number[], n: number, settings: SampleSettings, seed: number) {
  const rand = mulberry32(seed);
  const out: number[] = [], chosen: number[] = [];
  let probs: number[] | null = null;
  for (let i = 0; i < n; i++) {
    probs = shape(await nextLogits(e, cfg, [...ids, ...out].slice(-cfg.block_size)), settings).probs;
    const next = pick(probs, rand());
    out.push(next);
    chosen.push(probs[next]);
  }
  return { out, chosen, probs };
}

/** Continue a prompt with interventions (such as steering) at every step (concepts.generate). */
export async function generate(e: Engine, cfg: ModelConfig, tok: BPETokenizer, prompt: string, n: number,
                               interventions: Intervention[] | undefined, temperature: number, seed: number): Promise<string> {
  const ids = tok.encode(prompt);
  if (!ids.length) ids.push(0);
  const rand = mulberry32(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const logits = await nextLogits(e, cfg, [...ids, ...out].slice(-cfg.block_size), interventions);
    out.push(pick(shape(logits, { temperature, topK: null, topP: null }).probs, rand()));
  }
  return tok.decode(out);
}

/** For predicting the token after `t`: how much log p(actual) drops when each earlier token's
 * embedding is zeroed (its position kept). Positive: that token helped (concepts.occlusion). */
export async function occlusion(e: Engine, cfg: ModelConfig, ids: number[], t: number): Promise<number[]> {
  ids = ids.slice(-cfg.block_size);
  const n = t + 1, actual = ids[t + 1];
  const logp = async (zeroed: number | null) => {
    const r = await e.forward(ids.slice(0, n), {
      rewrite: (id, x, op) => (id === "embed.tok" && zeroed !== null
        ? op.overwrite(x, Array.from({ length: x.shape[1] }, (_, j) => zeroed * x.shape[1] + j), new Array(x.shape[1]).fill(0))
        : x),
      read: (live, op) => ({ z: op.rows(live.get("logits")!, t, 1) }),
    });
    return logSoftmax(r.z)[actual];
  };
  const base = await logp(null);
  const drops: number[] = [];
  for (let j = 0; j < n; j++) drops.push(base - (await logp(j)));
  return drops;
}

/** Activation patching (concepts.patch_sweep): copy one clean activation into the corrupted run
 * and measure how much of the clean answer comes back, as recovery of the logit difference. */
export async function patchSweep(e: Engine, cfg: ModelConfig, clean: number[], corrupt: number[], answer: number,
                                 wrong: number, what: "resid" | "heads") {
  const T = cfg.block_size;
  clean = clean.slice(-T);
  corrupt = corrupt.slice(-T);
  if (clean.length !== corrupt.length) {
    throw new Error(`the prompts must have the same number of tokens (${clean.length} vs ${corrupt.length})`);
  }
  const n = corrupt.length;
  const stages = what === "resid" ? snapshotIds(cfg)
    : what === "heads" ? Array.from({ length: cfg.n_layer * cfg.n_head }, (_, i) => `L${Math.floor(i / cfg.n_head)}.attn.H${i % cfg.n_head}.mix`)
    : (() => { throw new Error(`unknown patching target ${JSON.stringify(what)}`); })();
  const ld = (z: Float32Array) => z[answer] - z[wrong];
  const lastLogits = <T extends Node>(live: Map<string, T>, op: OpSet<T>) => op.rows(live.get("logits")!, n - 1, 1);

  const src = await e.forward(clean, { read: (live, op) => ({ ...Object.fromEntries(stages.map((s) => [s, live.get(s)!])), z: lastLogits(live, op) }) });
  const ldClean = ld(src.z);
  const ldCorrupt = ld((await e.forward(corrupt, { read: (live, op) => ({ z: lastLogits(live, op) }) })).z);
  const width = src[stages[0]].length / n;

  const recovered: number[] = [];
  const gap = ldClean - ldCorrupt;
  const cells = what === "resid" ? stages.flatMap((s) => Array.from({ length: n }, (_, p) => [s, p] as const)) : stages.map((s) => [s, null] as const);
  for (const [stage, p] of cells) {
    const rows = p === null ? Array.from({ length: n }, (_, i) => i) : [p];
    const index = rows.flatMap((r) => Array.from({ length: width }, (_, j) => r * width + j));
    const values = index.map((k) => src[stage][k]);
    const r = await e.forward(corrupt, {
      rewrite: (id, x, op) => (id === stage ? op.overwrite(x, index, values) : x),
      read: (live, op) => ({ z: lastLogits(live, op) }),
    });
    recovered.push(Math.abs(gap) > 1e-6 ? (ld(r.z) - ldCorrupt) / gap : 0);
  }
  const grid = what === "resid" ? { data: Float32Array.from(recovered), shape: [stages.length, n] }
    : { data: Float32Array.from(recovered), shape: [cfg.n_layer, cfg.n_head] };
  const source = new Float32Array(stages.length * n * width);
  stages.forEach((s, i) => source.set(src[s], i * n * width));
  return {
    grid, source: { data: source, shape: what === "resid" ? [stages.length, n, width] : [cfg.n_layer, cfg.n_head, n, width] },
    stages, ldClean, ldCorrupt,
  };
}

export interface CapturePrompt {
  text: string;
  targets: { start: number; end: number; word: string; category: string }[];
}

/** Residual vectors of every tagged word at every snapshot (concepts.capture). Spans are string
 * indices; a word split into several tokens is read at its last token, or averaged ("mean"). */
export async function capture(e: Engine, cfg: ModelConfig, tok: BPETokenizer, prompts: CapturePrompt[], pool: "last" | "mean") {
  const snaps = snapshotIds(cfg), d = cfg.d_model, S = snaps.length;
  const vectors: Float32Array[] = [];
  const items: object[] = [];
  for (const p of prompts) {
    let { ids, offsets } = tok.encodeWithOffsets(p.text);
    ids = ids.slice(0, cfg.block_size);
    offsets = offsets.slice(0, cfg.block_size);
    if (!ids.length) continue;
    const r = await e.forward(ids, { read: (live) => Object.fromEntries(snaps.map((s) => [s, live.get(s)!])) });
    for (const target of p.targets) {
      const toks = offsets.flatMap(([s, en], i) => (s < target.end && en > target.start ? [i] : []));
      if (!toks.length) continue;
      const v = new Float32Array(S * d);
      snaps.forEach((s, si) => {
        const use = pool === "last" ? [toks[toks.length - 1]] : toks;
        for (const t of use) for (let j = 0; j < d; j++) v[si * d + j] += r[s][t * d + j] / use.length;
      });
      vectors.push(v);
      items.push({ word: target.word, category: target.category, prompt: p.text,
                   tokens: toks.map((i) => tok.tokenText(ids[i])), position: toks[toks.length - 1] });
    }
  }
  const x = new Float32Array(vectors.length * S * d);
  vectors.forEach((v, i) => x.set(v, i * S * d));
  return { vectors: { data: x, shape: [vectors.length, S, d] }, items, snapshots: snaps };
}

// -- belief trajectories (core/beliefs.py) ------------------------------------------------

const SENTENCE_END = /(?<=[.!?])\s+/g;

/** [start, end) of each sentence, ending after its punctuation. */
export function splitSentences(text: string): [number, number][] {
  const spans: [number, number][] = [];
  let start = 0;
  for (const m of text.matchAll(SENTENCE_END)) {
    spans.push([start, m.index!]);
    start = m.index! + m[0].length;
  }
  if (text.slice(start).trim()) spans.push([start, text.trimEnd().length]);
  return spans;
}

/** The first n stories of the validation stream with at least two sentences. */
export function storySet(tok: BPETokenizer, val: Uint16Array, n: number): string[] {
  const stories: string[] = [];
  let start = 0;
  const limit = Math.min(val.length, 2_000_000);
  for (let i = 0; i < limit && stories.length < n; i++) {
    if (val[i] !== tok.eosId) continue;
    const text = tok.decode(Array.from(val.subarray(start, i))).trim();
    start = i + 1;
    if (splitSentences(text).length >= 2) stories.push(text);
  }
  return stories;
}

function wordIds(tok: BPETokenizer, words: string[]): number[] {
  return words.map((w) => {
    const enc = tok.encode(" " + w.trim());
    if (enc.length !== 1) throw new Error(`belief words must be single tokens: ${JSON.stringify(w)} is ${enc.length} tokens`);
    return enc[0];
  });
}

/** After each sentence of each story: the cloze readout over the words, its raw mass, and the
 * residual vectors at the prefix's last token. */
export async function beliefs(e: Engine, cfg: ModelConfig, tok: BPETokenizer, stories: string[], cloze: string, words: string[],
                              interventions?: Intervention[], withVectors = true) {
  const targets = wordIds(tok, words);
  const clozeIds = tok.encode(cloze);
  const keep = cfg.block_size - clozeIds.length;
  if (keep < 1) throw new Error("the cloze does not fit in the context window");
  const snaps = snapshotIds(cfg), S = snaps.length, d = cfg.d_model, K = targets.length;
  const prefixes: { story: number; sentence: number; start: number; end: number }[] = [];
  const readout: number[] = [], mass: number[] = [], vectors: Float32Array[] = [];
  for (const [si, text] of stories.entries()) {
    for (const [k, [start, end]] of splitSentences(text).entries()) {
      const ids = tok.encode(text.slice(0, end)).slice(-keep);
      const row = [...ids, ...clozeIds];
      const r = await e.forward(row, {
        interventions,
        read: (live, op) => ({
          z: op.rows(live.get("logits")!, row.length - 1, 1),
          ...(withVectors ? Object.fromEntries(snaps.map((s) => [s, op.rows(live.get(s)!, ids.length - 1, 1)])) : {}),
        }),
      });
      const lp = logSoftmax(r.z);
      const p = targets.map((t) => Math.exp(lp[t]));
      const total = p.reduce((a, b) => a + b, 0);
      readout.push(...p.map((v) => v / total));
      mass.push(total);
      if (withVectors) {
        const v = new Float32Array(S * d);
        snaps.forEach((s, i) => v.set(r[s], i * d));
        vectors.push(v);
      }
      prefixes.push({ story: si, sentence: k, start, end });
    }
  }
  const N = prefixes.length;
  const x = new Float32Array(N * S * d);
  vectors.forEach((v, i) => x.set(v, i * S * d));
  return {
    readout: { data: Float32Array.from(readout), shape: [N, K] }, mass: { data: Float32Array.from(mass), shape: [N] },
    vectors: { data: x, shape: [withVectors ? N : 0, S, d] }, prefixes, snapshots: snaps,
  };
}

/** The validation contexts where one MLP neuron fires most strongly (concepts.neuron_top). */
export async function neuronTop(e: Engine, cfg: ModelConfig, tok: BPETokenizer, val: Uint16Array, layer: number, neuron: number,
                                windows = 512, k = 12) {
  const T = cfg.block_size, rand = mulberry32(0);    // the same windows every time
  const stage = `L${layer}.mlp.gelu`;
  const best: { value: number; s: number; p: number }[] = [];
  for (let w = 0; w < windows; w++) {
    const s = Math.floor(rand() * (val.length - T - 1));
    const r = await e.forward(Array.from(val.subarray(s, s + T)), { read: (live, op) => ({ a: op.cols(live.get(stage)!, neuron, neuron + 1) }) });
    let p = 0;
    for (let i = 1; i < r.a.length; i++) if (r.a[i] > r.a[p]) p = i;
    best.push({ value: r.a[p], s, p });
  }
  best.sort((a, b) => b.value - a.value);
  return best.slice(0, k).map(({ value, s, p }) => {
    const ctx = Array.from(val.subarray(s + Math.max(0, p - 12), s + p + 4));
    const hi = Math.min(p, 12);
    return { value, before: tok.decode(ctx.slice(0, hi)), token: tok.tokenText(ctx[hi]), after: tok.decode(ctx.slice(hi + 1)) };
  });
}

/** The logit lens (core/lens.py): every residual snapshot through the final layer norm and the
 * unembedding, on the engine; the top k per cell and p(actual next token). */
export async function logitLens(e: Engine, cfg: ModelConfig, ids: number[], k = 8) {
  ids = ids.slice(-cfg.block_size);
  const snaps = snapshotIds(cfg), T = ids.length, V = cfg.vocab_size;
  const r = await e.forward(ids, {
    read: (live, op, w) => Object.fromEntries(snaps.map((s) => {
      const h = op.layerNorm(live.get(s)!, w("ln_f_g"), w("ln_f_b"));
      return [s, op.probs(cfg.tied ? op.linearT(h, w("wte")) : op.linear(h, w("W_unembed"), null))];
    })),
  });
  const topIds: number[][][] = [], topP: number[][][] = [], pActual: number[][] = [];
  for (const s of snaps) {
    const probs = r[s], ti: number[][] = [], tp: number[][] = [], pa: number[] = [];
    for (let i = 0; i < T; i++) {
      const row = probs.subarray(i * V, (i + 1) * V);
      const top = topK(row, k);
      ti.push(top);
      tp.push(top.map((j) => row[j]));
      pa.push(i < T - 1 ? row[ids[i + 1]] : -1);
    }
    topIds.push(ti);
    topP.push(tp);
    pActual.push(pa);
  }
  return { snapshots: snaps, topIds, topP, pActual };
}

/** Indices of the k largest values, largest first. */
function topK(row: ArrayLike<number>, k: number): number[] {
  const top: number[] = [];
  for (let j = 0; j < row.length; j++) {
    if (top.length === k && row[j] <= row[top[k - 1]]) continue;
    let at = top.length === k ? k - 1 : top.length;
    if (top.length < k) top.push(j);
    while (at > 0 && row[top[at - 1]] < row[j]) {
      top[at] = top[at - 1];
      at--;
    }
    top[at] = j;
  }
  return top;
}
