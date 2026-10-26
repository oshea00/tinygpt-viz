// Static demo mode: the app without a server. Loads exported weights, answers "run"
// commands with the TypeScript forward pass in a worker, and ignores everything else
// (training, interventions and gradients need the Python core).

import { decodeBinary, type BinaryMsg } from "../bridge";
import type { Command, Hello, JsonMsg, ModelConfig, Trace, Weights } from "../types";
import { infoFrom } from "./forward";
import type { WorkerIn } from "./worker";

interface Meta {
  config: ModelConfig;
  vocab: string[];
  chars: string[];
  probe: string;
  step: number;
  params: number;
  paramGroups: Record<string, number>;
}

async function fetchBinary(url: string): Promise<BinaryMsg> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return decodeBinary(await r.arrayBuffer());
}

export async function startDemo(onMessage: (m: JsonMsg | BinaryMsg) => void): Promise<(cmd: Command) => void> {
  const meta: Meta = await (await fetch("demo/meta.json")).json();
  const weights = ((await fetchBinary("demo/weights.bin")) as { weights: Weights }).weights;
  const reference = ((await fetchBinary("demo/reference.bin")) as { trace: Trace }).trace;
  const stoi = new Map(meta.chars.map((c, i) => [c, i]));

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.postMessage({ type: "init", weights: weights.tensors, cfg: meta.config, info: [...infoFrom(reference)] } satisfies WorkerIn);

  const prompts = new Map<number, string>();
  let nextId = 0;
  let latest = -1;
  worker.onmessage = (e: MessageEvent<{ id: number; trace: Omit<Trace, "byId"> }>) => {
    if (e.data.id !== latest) return;   // a newer request is already on its way
    const t = e.data.trace;
    const trace: Trace = { ...t, step: meta.step, prompt: prompts.get(e.data.id), reason: "run", weightsVersion: 1,
                           byId: new Map(t.stages.map((s) => [s.id, s])) };
    prompts.delete(e.data.id);
    onMessage({ type: "trace", trace });
  };

  const hello: Hello = {
    type: "hello", config: meta.config, vocab: meta.vocab, params: meta.params, paramGroups: meta.paramGroups,
    probe: meta.probe, step: meta.step, playing: false, lr: 0, batchSize: 0, traceEvery: 0, device: "browser", checkpoints: [], training: false,
  };
  onMessage(hello);
  onMessage({ type: "weights", weights });

  return (cmd: Command) => {
    if (cmd.type !== "run" || cmd.tag) return;
    const ids = [...cmd.prompt].flatMap((c) => (stoi.has(c) ? [stoi.get(c)!] : [])).slice(-meta.config.block_size);
    const tokens = ids.length ? ids : [0];
    latest = nextId++;
    prompts.set(latest, cmd.prompt);
    worker.postMessage({ type: "run", id: latest, tokens, text: tokens.map((i) => meta.vocab[i]) } satisfies WorkerIn);
  };
}
