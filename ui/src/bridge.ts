// WebSocket client and binary frame decoding (see core/trace_io.py for the layout).

import type { Command, JsonMsg, Stage, StepExplain, Tensor, Trace, Weights } from "./types";
import type { Capture } from "./lib/concepts";

interface Ref {
  shape: number[];
  offset: number;
}

export interface Frame {
  header: any;
  tensor(ref: Ref, shape?: number[]): Tensor;
}

export function decodeFrame(buf: ArrayBuffer): Frame {
  const n = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const base = 4 + n;
  return {
    header,
    tensor(ref, shape = ref.shape) {
      const count = shape.reduce((a, b) => a * b, 1);
      return { shape, data: new Float32Array(buf, base + ref.offset, count) };
    },
  };
}

function tensorGroup(f: Frame, refs: Record<string, Ref>): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(refs).map(([k, r]) => [k, f.tensor(r)]));
}

export function decodeTrace(f: Frame): Trace {
  const h = f.header;
  const stages: Stage[] = h.stages.map((s: any) => ({
    id: s.id,
    kind: s.kind,
    layer: s.layer,
    head: s.head,
    inputs: s.inputs ?? [],
    weights: s.weights ?? [],
    formula: s.formula ?? "",
    explain: s.explain ?? "",
    backward: s.backward ?? "",
    output: f.tensor(s),
    grad: s.gradOffset !== undefined ? f.tensor({ shape: s.shape, offset: s.gradOffset }) : undefined,
  }));
  return {
    step: h.step,
    loss: h.loss,
    tokens: h.tokens,
    text: h.text,
    targets: h.targets,
    stages,
    byId: new Map(stages.map((s) => [s.id, s])),
    weightGrads: tensorGroup(f, h.weightGrads),
    prompt: h.prompt,
    reason: h.reason,
    weightsVersion: h.weightsVersion,
    tag: h.tag,
    incremental: h.incremental,
    interventions: h.interventions ?? [],
  };
}

export function decodeWeights(f: Frame): Weights {
  return { step: f.header.step, version: f.header.version, tensors: tensorGroup(f, f.header.weights) };
}

export function decodeStepExplain(f: Frame): StepExplain {
  const h = f.header;
  return {
    step: h.step, prompt: h.prompt, text: h.text, lr: h.lr,
    lossBefore: h.lossBefore, lossAfter: h.lossAfter, pBefore: h.pBefore, pAfter: h.pAfter,
    before: tensorGroup(f, h.before), grad: tensorGroup(f, h.grad), delta: tensorGroup(f, h.delta),
  };
}

export function decodeCapture(f: Frame): Capture {
  const h = f.header;
  const t = f.tensor(h.vectors.x);
  const [N, S, d] = t.shape.length === 3 ? t.shape : [0, h.snapshots.length, 0];
  return { vectors: t.data, N, S, d, items: h.items, snapshots: h.snapshots, model: h.model, step: h.step };
}

export type BinaryMsg =
  | { type: "capture"; capture: Capture; tag?: string }
  | { type: "trace"; trace: Trace }
  | { type: "weights"; weights: Weights }
  | { type: "stepExplain"; explain: StepExplain };

export function decodeBinary(buf: ArrayBuffer): BinaryMsg {
  const f = decodeFrame(buf);
  switch (f.header.type) {
    case "trace": return { type: "trace", trace: decodeTrace(f) };
    case "weights": return { type: "weights", weights: decodeWeights(f) };
    case "stepExplain": return { type: "stepExplain", explain: decodeStepExplain(f) };
    case "capture": return { type: "capture", capture: decodeCapture(f), tag: f.header.tag };
    default: throw new Error(`unknown frame ${f.header.type}`);
  }
}

export class Bridge {
  private ws: WebSocket | null = null;
  private retry = 250;

  constructor(
    private url: string,
    private onMessage: (m: JsonMsg | BinaryMsg) => void,
    private onState: (connected: boolean) => void,
  ) {}

  connect(): void {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 250;
      this.onState(true);
    };
    ws.onclose = () => {
      this.onState(false);
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 5000);
    };
    ws.onmessage = (e) => {
      this.onMessage(typeof e.data === "string" ? JSON.parse(e.data) : decodeBinary(e.data));
    };
    this.ws = ws;
  }

  send(cmd: Command): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(cmd));
  }
}
