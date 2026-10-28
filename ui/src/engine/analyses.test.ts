import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { decodeFrame, decodeWeights } from "../bridge";
import { checkAnalyses } from "./analyses.cases";
import { CpuEngine } from "./engine";
import { makeWeights } from "./model";
import { BPETokenizer } from "./tokenizer";

// tests/test_ui_fixtures.py: a tiny tied model with the real vocabulary, and what the core's
// commands return for it.
it("analyses match the core on the CPU engine", async () => {
  const fx = JSON.parse(readFileSync(new URL("../fixtures/analyses.json", import.meta.url), "utf8"));
  const wb = readFileSync(new URL("../fixtures/analyses_weights.bin", import.meta.url));
  const tok = new BPETokenizer(JSON.parse(readFileSync(new URL("../../../data/bpe4096/tokenizer.json", import.meta.url), "utf8")));
  const engine = new CpuEngine(fx.config);
  engine.setWeights(makeWeights(decodeWeights(decodeFrame(wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength))).tensors));
  expect(await checkAnalyses(engine, tok, fx)).toEqual([]);
});
