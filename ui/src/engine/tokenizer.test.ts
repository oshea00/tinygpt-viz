import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { BPETokenizer } from "./tokenizer";

const json = JSON.parse(readFileSync(new URL("../../../data/bpe4096/tokenizer.json", import.meta.url), "utf8"));
const fx = JSON.parse(readFileSync(new URL("../fixtures/tokenizer.json", import.meta.url), "utf8"));
const tok = new BPETokenizer(json);

/** Python offsets count code points; ours count UTF-16 units. */
function toUtf16(text: string, cp: number): number {
  return [...text].slice(0, cp).join("").length;
}

describe("BPETokenizer matches core/tokenizers.py", () => {
  for (const c of fx.cases) {
    it(JSON.stringify(c.text).slice(0, 60), () => {
      expect(tok.encode(c.text)).toEqual(c.ids);
      expect(tok.decode(c.ids)).toBe(c.decoded);
      const { ids, offsets } = tok.encodeWithOffsets(c.offsetText);
      expect(ids).toEqual(c.offsetIds);
      expect(offsets).toEqual(c.offsets.map(([a, b]: number[]) => [toUtf16(c.offsetText, a), toUtf16(c.offsetText, b)]));
    });
  }

  it("shows every token as the core does", () => {
    expect(tok.vocabSize).toBe(fx.display.length);
    fx.display.forEach((d: string, id: number) => expect(tok.tokenText(id)).toBe(d));
  });
});
