import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { decodeFrame, decodeWeights } from "../bridge";
import { CpuEngine } from "./engine";
import { checkLora } from "./lora.cases";
import { makeWeights } from "./model";

const read = (name: string) => {
  const b = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

it("adapters match core/lora.py on the CPU engine", async () => {
  const cfg = { n_layer: 2, n_head: 2, d_model: 32, block_size: 100, vocab_size: 16, tied: false };
  const W = makeWeights(decodeWeights(decodeFrame(read("weights.bin"))).tensors);
  expect(await checkLora(new CpuEngine(cfg), W, read("lora/trace.bin"), read("lora/train.bin"))).toEqual([]);
});
