import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { CpuEngine } from "./engine";
import { checkTraining, TRAIN_CONFIG, trainFixture } from "./train.cases";

it("training matches core/train.py on the CPU engine", async () => {
  const b = readFileSync(new URL("../fixtures/train.bin", import.meta.url));
  const fx = trainFixture(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  expect(await checkTraining(new CpuEngine(TRAIN_CONFIG), fx)).toEqual([]);
});
