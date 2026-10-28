// Incremental generation with a KV cache, traced (core/kvcache.py).
//
// A causal model never lets a position see later ones, so appending a token changes nothing at
// earlier positions. Generation can keep each layer's keys and values and run only the new
// position; the trace is assembled from the cached rows plus the new one. Once the text is
// longer than the window, every position shifts and every cached key and value is stale.

import { Tensor } from "./autograd";
import type { Engine } from "./engine";
import type { ModelConfig, TraceResult } from "./model";

const ATTENTION_MATRICES: Record<string, number> = { attn_scores: NaN, mask: -Infinity, softmax: 0 };  // [queries, keys]

export interface KVCache {
  tokens: number[];
  weightsVersion: number;
  past: { k: Float32Array; v: Float32Array }[];
  outputs: Map<string, { data: Float32Array; shape: number[] }>;
}

export const emptyCache = (): KVCache => ({ tokens: [], weightsVersion: -1, past: [], outputs: new Map() });

/** Multiply-adds to compute positions start..end-1 (a full pass: 0..T). */
export function multiplyAdds(c: ModelConfig & { d_mlp: number }, start: number, end: number): number {
  const perPosition = c.n_layer * (4 * c.d_model ** 2 + 2 * c.d_model * c.d_mlp) + c.d_model * c.vocab_size;
  // Attention: position p scores against p + 1 keys and mixes p + 1 values, in every layer.
  let attention = 0;
  for (let p = start; p < end; p++) attention += 2 * (p + 1) * c.d_model * c.n_layer;
  return (end - start) * perPosition + attention;
}

/** null if the cache applies to `tokens`; otherwise the reason it cannot. */
export function whyNotReuse(cache: KVCache, tokens: number[], weightsVersion: number, block: number): string | null {
  const n = cache.tokens.length;
  const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
  if (!n) return "no cache yet: the first pass computes every position";
  if (cache.weightsVersion !== weightsVersion) return "the weights changed, so every cached key and value is stale";
  if (tokens.length === block && n === block && same(tokens.slice(0, -1), cache.tokens.slice(1))) {
    return "the window slid: every position moved, so every cached key and value is stale";
  }
  if (tokens.length <= n || !same(tokens.slice(0, n), cache.tokens)) return "the prompt changed, so the cache no longer matches";
  return null;
}

/** Trace `tokens`, reusing the cache where it applies, and update the cache. `kinds` gives each
 * stage's kind (core/stages.py). */
export async function traceCached(e: Engine, cfg: ModelConfig & { d_mlp: number }, cache: KVCache, tokens: number[],
                                  weightsVersion: number, kinds: (id: string) => string) {
  const block = cfg.block_size;
  tokens = tokens.slice(-block);
  const reason = whyNotReuse(cache, tokens, weightsVersion, block);
  const t0 = reason ? 0 : cache.tokens.length;
  const r = await e.forward(tokens.slice(t0), {
    past: reason ? undefined : cache.past, keepKV: true,
    read: (live) => Object.fromEntries(live),
  });

  const T = tokens.length;
  const outputs = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [id, data] of Object.entries(r)) {
    if (id.startsWith("kv.")) continue;
    const rows = T - t0, cols = data.length / rows;
    const fill = ATTENTION_MATRICES[kinds(id)];
    if (t0 === 0) {
      outputs.set(id, { data: data.slice(), shape: [rows, cols] });
    } else if (fill !== undefined) {
      // Old query rows are unchanged; their entries for the new keys are masked (−∞ before
      // softmax, 0 after), and the raw scores there were never computed (NaN).
      const full = new Float32Array(T * T).fill(fill), old = cache.outputs.get(id)!.data;
      for (let i = 0; i < t0; i++) full.set(old.subarray(i * t0, (i + 1) * t0), i * T);
      full.set(data, t0 * T);
      outputs.set(id, { data: full, shape: [T, T] });
    } else {
      const old = cache.outputs.get(id)!.data, full = new Float32Array(old.length + data.length);
      full.set(old);
      full.set(data, old.length);
      outputs.set(id, { data: full, shape: [T, cols] });
    }
  }

  // The loss over the prompt's own next tokens, from the assembled logits (in float64).
  const targets = [...tokens.slice(1), -1];
  const logits = outputs.get("logits")!, V = logits.shape[1];
  let loss: number | null = null;
  if (T > 1) {
    let total = 0;
    for (let i = 0; i < T - 1; i++) {
      const row = logits.data.subarray(i * V, (i + 1) * V);
      let max = -Infinity, s = 0;
      for (const z of row) max = Math.max(max, z);
      for (const z of row) s += Math.exp(z - max);
      total -= row[targets[i]] - max - Math.log(s);
    }
    loss = total / (T - 1);
    outputs.set("loss", { data: Float32Array.of(loss), shape: [1] });
  }

  cache.tokens = [...tokens];
  cache.weightsVersion = weightsVersion;
  cache.outputs = outputs;
  cache.past = Array.from({ length: cfg.n_layer }, (_, l) => ({ k: r[`kv.L${l}.k`].slice(), v: r[`kv.L${l}.v`].slice() }));

  const result: TraceResult = {
    tokens, targets, loss, weightGrads: new Map(),
    stages: [...outputs].map(([id, o]) => ({ id, output: new Tensor(o.data, o.shape), grad: null })),
  };
  const info = { reused: t0, computed: T - t0, reason, multiplyAdds: multiplyAdds(cfg, t0, T), multiplyAddsFull: multiplyAdds(cfg, 0, T) };
  return { result, info };
}
