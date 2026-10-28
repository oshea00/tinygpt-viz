// Binary frames in the layout of core/trace_io.py, so the in-browser backend hands the UI
// exactly what the server would:
//
//   u32 little-endian header length (a multiple of 4)
//   JSON header, space-padded to that length
//   float32 data; header offsets are in bytes from the start of this section

import type { TraceResult } from "./model";

interface Ref {
  shape: number[];
  offset: number;
}

class Packer {
  chunks: Float32Array[] = [];
  size = 0;

  add(data: Float32Array, shape: number[]): Ref {
    const ref = { shape, offset: this.size };
    this.chunks.push(data);
    this.size += data.byteLength;
    return ref;
  }
}

/** A frame's buffer for `header` and `dataBytes` of tensor data; the caller fills `data`. */
export function frameAround(header: object, dataBytes: number): { buf: ArrayBuffer; data: Uint8Array } {
  const raw = new TextEncoder().encode(JSON.stringify(header));
  const pad = -raw.length & 3;
  const buf = new ArrayBuffer(4 + raw.length + pad + dataBytes);
  new DataView(buf).setUint32(0, raw.length + pad, true);
  const bytes = new Uint8Array(buf);
  bytes.set(raw, 4);
  bytes.fill(0x20, 4 + raw.length, 4 + raw.length + pad);
  return { buf, data: new Uint8Array(buf, 4 + raw.length + pad) };
}

function frame(header: object, pk: Packer): ArrayBuffer {
  const { buf, data } = frameAround(header, pk.size);
  let at = 0;
  for (const c of pk.chunks) {
    data.set(new Uint8Array(c.buffer, c.byteOffset, c.byteLength), at);
    at += c.byteLength;
  }
  return buf;
}

/** A trace frame's header. `offsetOf` places each tensor in the data section and returns its byte
 * offset there: by packing it (encodeTrace), or where it already is (the GPU's read-back). */
export function traceHeader(t: TraceResult, meta: (id: string) => object, step: number, text: string[], extra: object,
                            offsetOf: (a: Float32Array) => number): object {
  const stages = t.stages.map((s) => {
    const entry: Record<string, unknown> = { ...meta(s.id), shape: s.output.shape, offset: offsetOf(s.output.data) };
    if (s.grad) entry.gradOffset = offsetOf(s.grad);
    return entry;
  });
  const weightGrads = Object.fromEntries([...t.weightGrads].map(([n, g]) => [n, { shape: g.shape, offset: offsetOf(g.data) }]));
  return { type: "trace", step, loss: t.loss, tokens: t.tokens, text, targets: t.targets, stages, weightGrads, ...extra };
}

/** A trace frame. `meta` gives each stage's kind, inputs, formula... (core/stages.py). */
export function encodeTrace(t: TraceResult, meta: (id: string) => object, step: number, text: string[],
                            extra: object = {}): ArrayBuffer {
  const pk = new Packer();
  return frame(traceHeader(t, meta, step, text, extra, (a) => pk.add(a, []).offset), pk);
}

/** A frame of named tensor groups, e.g. {"weights": {"L0.W_q": ...}}. */
export function encodeTensors(type: string, groups: Record<string, Map<string, { data: Float32Array; shape: number[] }>>,
                              extra: object = {}): ArrayBuffer {
  const pk = new Packer();
  const header: Record<string, unknown> = { type };
  for (const [g, ts] of Object.entries(groups)) {
    header[g] = Object.fromEntries([...ts].map(([n, a]) => [n, pk.add(a.data, a.shape)]));
  }
  return frame({ ...header, ...extra }, pk);
}
