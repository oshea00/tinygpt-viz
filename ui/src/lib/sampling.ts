// Logits → the distribution actually sampled from. Mirrors core/sampling.py, which the
// parity test checks, so the bars the learner sees are exactly what gets sampled.

export interface SampleSettings {
  temperature: number;     // 0 means greedy
  topK: number | null;     // null: off
  topP: number | null;     // null: off
}

export interface Shaped {
  tempered: number[];      // softmax(logits / T) before any filtering
  probs: number[];         // after top-k / top-p, renormalised; excluded entries are 0
}

export function shape(logits: ArrayLike<number>, s: SampleSettings): Shaped {
  const n = logits.length;
  if (s.temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    const probs = Array.from({ length: n }, (_, i) => (i === best ? 1 : 0));
    return { tempered: probs, probs };
  }
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logits[i] / s.temperature);
  const e = Array.from(logits, (z) => Math.exp(z / s.temperature - max));
  const total = e.reduce((a, b) => a + b, 0);
  const tempered = e.map((v) => v / total);
  let probs = [...tempered];
  if (s.topK !== null && s.topK < n) {
    const cutoff = [...probs].sort((a, b) => b - a)[s.topK - 1];
    probs = probs.map((p) => (p >= cutoff ? p : 0));
  }
  if (s.topP !== null && s.topP < 1) {
    // Keep the smallest prefix (by probability) whose mass reaches top_p, at least one token.
    const order = probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0]);
    let before = 0;
    const kept = new Array(n).fill(0);
    for (const [p, i] of order) {
      if (before < s.topP) kept[i] = p;
      before += p;
    }
    probs = kept;
  }
  const sum = probs.reduce((a, b) => a + b, 0);
  return { tempered, probs: probs.map((p) => p / sum) };
}

/** Inverse-CDF draw: u in [0, 1). */
export function pick(probs: number[], u = Math.random()): number {
  let acc = 0;
  let last = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] <= 0) continue;
    acc += probs[i];
    last = i;
    if (u < acc) return i;
  }
  return last;
}

export function describeSettings(s: SampleSettings): string {
  if (s.temperature <= 0) return "greedy";
  return [`T ${s.temperature.toFixed(2)}`, s.topK !== null ? `top-k ${s.topK}` : "", s.topP !== null && s.topP < 1 ? `top-p ${s.topP.toFixed(2)}` : ""]
    .filter(Boolean).join(" · ");
}
