// The analyses against tests/test_ui_fixtures.py's results, for any engine: Vitest runs them on
// the CPU (analyses.test.ts), gpu-check.html on WebGPU. Returns what failed.

import { beliefs, capture, generate, logitLens, occlusion, patchSweep } from "./analyses";
import { Tensor } from "./autograd";
import { using, type Engine } from "./engine";
import { meanLoss, pWord, scaled, scorePairs, topicId, windows } from "./lora";
import type { BPETokenizer } from "./tokenizer";

export async function checkAnalyses(engine: Engine, tok: BPETokenizer, fx: any): Promise<string[]> {
  const failures: string[] = [];
  const cfg = fx.config;
  const close = (got: ArrayLike<number>, want: ArrayLike<number>, what: string, rel = 1e-4) => {
    if (got.length !== want.length) return failures.push(`${what}: ${got.length} values, expected ${want.length}`);
    let scale = 0, worst = 0;
    for (let i = 0; i < want.length; i++) scale = Math.max(scale, Math.abs(want[i]));
    for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
    if (worst > rel * scale + 1e-7) failures.push(`${what}: off by ${worst.toExponential(2)} (scale ${scale.toExponential(2)})`);
  };
  const same = (got: unknown, want: unknown, what: string) => {
    if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${what} differ`);
  };

  const lens = await logitLens(engine, cfg, fx.ids);
  same(lens.snapshots, fx.lens.snapshots, "lens snapshots");
  close(lens.topP.flat(2), fx.lens.topP.flat(2), "lens topP");
  close(lens.pActual.flat(), fx.lens.pActual.flat(), "lens pActual");
  same(lens.topIds, fx.lens.topIds, "lens topIds");

  close(await occlusion(engine, cfg, fx.ids, fx.occlusion.position), fx.occlusion.drops, "occlusion");

  const p = fx.patch;
  for (const what of ["resid", "heads"] as const) {
    const r = await patchSweep(engine, cfg, p.clean, p.corrupt, p.answer, p.wrong, what);
    close([r.ldClean, r.ldCorrupt], [p[what].ldClean, p[what].ldCorrupt], `patch ${what} logit differences`);
    close(r.grid.data, p[what].grid.flat(), `patch ${what} grid`, 1e-3);
    close([r.source.data.reduce((a, b) => a + b, 0)], [p[what].sourceSum], `patch ${what} source`);
  }

  for (const pool of ["last", "mean"] as const) {
    const r = await capture(engine, cfg, tok, fx.capture.prompts, pool);
    same(r.items, fx.capture.items, "capture items");
    close(r.vectors.data, fx.capture[pool].flat(2), `capture vectors (${pool})`);
  }

  const b = fx.beliefs;
  const r = await beliefs(engine, cfg, tok, b.stories, b.cloze, b.words);
  same(r.prefixes, b.prefixes, "belief prefixes");
  close(r.readout.data, b.readout.flat(), "belief readout");
  close(r.mass.data, b.mass, "belief mass");
  close(r.vectors.data, b.vectors.flat(2), "belief vectors");
  const steered = await beliefs(engine, cfg, tok, b.stories, b.cloze, b.words, b.steer, false);
  close(steered.readout.data, b.steered.flat(), "steered belief readout");

  const g = fx.generate;
  same(await generate(engine, cfg, tok, g.prompt, g.n, g.interventions, 0, 0), g.text, "greedy steered generation");

  // An adapter at half strength: the fine-tuning tab's measures (core/lora.py, core/dpo.py).
  const L = fx.lora;
  const A = new Map(Object.entries(L.weights as Record<string, number[][]>)
    .map(([n, rows]) => [n, new Tensor(Float32Array.from(rows.flat()), [rows.length, rows[0].length])]));
  engine.setAdapter("fixture", scaled(A, L.scale));
  const adapted = using(engine, "current", "fixture");
  same(windows(tok, L.stories, cfg.block_size), L.windows, "adapter training windows");
  close([await pWord(adapted, cfg, tok, L.openers, topicId(tok, L.topic))], [L.pTopic], "p(topic) with an adapter");
  close([await meanLoss(adapted, L.windows, "fixture")], [L.meanLoss], "mean loss with an adapter");
  const sc = await scorePairs(engine, adapted, cfg, tok, L.pairs, L.scores.beta);
  for (const k of ["ref_chosen", "ref_rejected", "policy_chosen", "policy_rejected", "margin"] as const) {
    close(sc[k], L.scores[k], `pair scores ${k}`);
  }
  engine.setAdapter("fixture", null);
  return failures;
}
