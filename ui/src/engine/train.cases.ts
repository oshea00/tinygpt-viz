// core/train.py's Trainer against any engine, from tests/test_ui_fixtures.py's two steps: Vitest
// runs it on the CPU (train.test.ts), gpu-check.html on WebGPU. Returns what failed.

import { decodeFrame } from "../bridge";
import { Tensor } from "./autograd";
import type { Engine } from "./engine";
import type { Weights } from "./model";

export const TRAIN_CONFIG = { n_layer: 2, n_head: 2, d_model: 32, d_mlp: 128, block_size: 100, vocab_size: 16, tied: false };

export function trainFixture(buf: ArrayBuffer) {
  const f = decodeFrame(buf);
  const group = (g: string): Weights => new Map(Object.entries(f.header[g] as Record<string, { shape: number[]; offset: number }>)
    .map(([n, r]) => [n, new Tensor(f.tensor(r).data.slice(), r.shape)]));
  return { header: f.header, before: () => group("before"), after: group("after") };
}

export async function checkTraining(e: Engine, fx: ReturnType<typeof trainFixture>): Promise<string[]> {
  const failures: string[] = [];
  const h = fx.header, before = fx.before();
  e.setWeights(fx.before());
  e.resetOptimizer();
  // core/train.py's warmup: step k (from 1) uses lr · min(1, k / warmup).
  const lrAt = (k: number) => h.config.lr * Math.min(1, k / h.config.warmupSteps);
  for (const [k, batch] of h.batches.entries()) {
    const r = await e.trainStep(batch.x.flat(), batch.y.flat(), batch.x.length,
                                { lr: lrAt(k + 1), weightDecay: h.config.weightDecay, gradClip: h.config.gradClip });
    if (Math.abs(r.loss - h.steps[k].loss) > 1e-5) failures.push(`step ${k + 1} loss ${r.loss} vs ${h.steps[k].loss}`);
    if (Math.abs(r.gradNorm - h.steps[k].gradNorm) > 1e-4 * h.steps[k].gradNorm) {
      failures.push(`step ${k + 1} gradient norm ${r.gradNorm} vs ${h.steps[k].gradNorm}`);
    }
  }
  // AdamW's changes are what is compared: the weights barely move in two steps.
  for (const [n, w] of await e.readWeights()) {
    // The key bias's gradient is zero in exact arithmetic (softmax ignores a shift), so both
    // sides hold rounding noise, which Adam scales up to a step: nothing to compare.
    if (n.endsWith(".b_k")) continue;
    const b = before.get(n)!.data, want = fx.after.get(n)!.data;
    let scale = 0, worst = 0;
    for (let i = 0; i < b.length; i++) scale = Math.max(scale, Math.abs(want[i] - b[i]));
    for (let i = 0; i < b.length; i++) worst = Math.max(worst, Math.abs(w.data[i] - b[i] - (want[i] - b[i])));
    if (worst > 1e-3 * scale + 1e-8) failures.push(`change in ${n} off by ${worst.toExponential(2)} (scale ${scale.toExponential(2)})`);
  }
  return failures;
}
