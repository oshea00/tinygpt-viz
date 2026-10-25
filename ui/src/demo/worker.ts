// Runs the TypeScript forward pass off the main thread, so typing stays smooth.

import type { ModelConfig, Tensor } from "../types";
import { forward, type StageInfo } from "./forward";

let model: { weights: Record<string, Tensor>; cfg: ModelConfig; info: Map<string, StageInfo> } | null = null;

export type WorkerIn =
  | { type: "init"; weights: Record<string, Tensor>; cfg: ModelConfig; info: [string, StageInfo][] }
  | { type: "run"; id: number; tokens: number[]; text: string[] };

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const m = e.data;
  if (m.type === "init") {
    model = { weights: m.weights, cfg: m.cfg, info: new Map(m.info) };
    return;
  }
  if (!model) return;
  const t = forward(m.tokens, model.weights, model.cfg, model.info, m.text);
  // Send the stages without the id map (rebuilt on the main thread), transferring the buffers.
  const { byId: _, ...plain } = t;
  self.postMessage({ id: m.id, trace: plain }, { transfer: t.stages.map((s) => s.output.data.buffer as ArrayBuffer) });
};
