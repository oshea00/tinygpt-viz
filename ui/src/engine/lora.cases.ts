// core/lora.py's adapters against any engine, from tests/test_ui_fixtures.py's lora fixtures: a
// trace with the adapter on at half strength, then two of its training steps. Vitest runs it on
// the CPU (lora.test.ts), gpu-check.html on WebGPU. Returns what failed.

import { decodeFrame, decodeTrace, type Frame } from "../bridge";
import { Tensor } from "./autograd";
import type { Engine } from "./engine";
import { batchOf, scaled } from "./lora";
import type { Weights } from "./model";
import { copyWeights } from "./trainer";

const group = (f: Frame, g: string): Weights => new Map(Object.entries(f.header[g] as Record<string, { shape: number[]; offset: number }>)
  .map(([n, r]) => [n, new Tensor(f.tensor(r).data.slice(), r.shape)]));

/** Close to the reference relative to its largest value, as model.test.ts's expectClose (with
 * its floor for gradients that are zero in exact arithmetic, such as the key bias's). */
function close(got: ArrayLike<number>, want: ArrayLike<number>, tol: number): boolean {
  let scale = 0, worst = 0;
  for (let i = 0; i < want.length; i++) if (Number.isFinite(want[i])) scale = Math.max(scale, Math.abs(want[i]));
  for (let i = 0; i < want.length; i++) {
    if (!Number.isFinite(want[i])) worst = Math.max(worst, got[i] === want[i] ? 0 : Infinity);
    else worst = Math.max(worst, Math.abs(got[i] - want[i]));
  }
  return worst < tol * scale + 1e-9;
}

/** `base`: the trace fixture's weights; `trace`, `train`: the lora/*.bin frames. */
export async function checkLora(e: Engine, base: Weights, traceBuf: ArrayBuffer, trainBuf: ArrayBuffer, tol = 1e-5): Promise<string[]> {
  const failures: string[] = [];
  const ref = decodeTrace(decodeFrame(traceBuf));
  const fx = decodeFrame(trainBuf), before = group(fx, "before"), after = group(fx, "after");
  e.setWeights(copyWeights(base));
  e.setAdapter("fixture", scaled(before, ref.adapter!.scale ?? 1));
  const meta = (id: string) => {
    const { output, grad, ...m } = ref.byId.get(id)!;
    return m;
  };
  const t = await e.trace(ref.tokens, { nextToken: 9, adapter: "fixture" }, { meta, step: 0, text: () => [], extra: {} });
  const ids = t.stages.map((s) => s.id).join(" ");
  if (ids !== ref.stages.map((s) => s.id).join(" ")) failures.push(`trace stages: ${ids}`);
  if (Math.abs(t.loss! - ref.loss!) > 1e-5) failures.push(`trace loss ${t.loss} vs ${ref.loss}`);
  for (const s of t.stages) {
    const r = ref.byId.get(s.id);
    if (!r) continue;
    if (!close(s.output.data, r.output.data, tol)) failures.push(`stage ${s.id}`);
    if (s.grad && r.grad && !close(s.grad, r.grad.data, tol)) failures.push(`grad ${s.id}`);
  }
  for (const [n, g] of t.weightGrads) {
    if (!close(g.data, ref.weightGrads[n].data, tol)) failures.push(`weight grad ${n}`);
  }

  // The loss and a forward pass without gradients, with the adapter on (what training's measures use).
  const targets = [...ref.tokens.slice(1), 9];
  const loss = await e.loss(ref.tokens, targets, 1, "fixture");
  if (Math.abs(loss - ref.loss!) > 1e-5) failures.push(`loss with the adapter ${loss} vs ${ref.loss}`);
  const { z } = await e.forward(ref.tokens, { adapter: "fixture", read: (live) => ({ z: live.get("logits")! }) });
  if (!close(z, ref.byId.get("logits")!.output.data, tol)) failures.push("forward with the adapter: logits");

  // Training: the weights stay put, A and B move as core/lora.py's train_step moved them.
  e.setAdapter("fixture", copyWeights(before));     // the engine trains its copy in place
  e.resetOptimizer("fixture");
  const h = fx.header;
  for (const [k, ws] of (h.batches as number[][][]).entries()) {
    const b = batchOf(ws);
    const r = await e.trainStep(b.x, b.y, b.B, { lr: h.lr, weightDecay: 0, gradClip: h.gradClip }, "fixture");
    if (Math.abs(r.loss - h.losses[k]) > 1e-5) failures.push(`train step ${k + 1} loss ${r.loss} vs ${h.losses[k]}`);
  }
  const W = await e.readWeights();
  for (const [n, w] of base) if (!close(W.get(n)!.data, w.data, 0)) failures.push(`training an adapter moved ${n}`);
  for (const [n, w] of await e.readAdapter("fixture")) {
    const b = before.get(n)!.data, want = after.get(n)!.data;
    const change = (x: ArrayLike<number>) => Array.from(x, (v, i) => v - b[i]);
    if (!close(change(w.data), change(want), 1e-3)) failures.push(`change in ${n}`);
  }
  return failures;
}
