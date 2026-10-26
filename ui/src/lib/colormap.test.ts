import { describe, expect, it } from "vitest";
import { cssColor, inkOn } from "./colormap";

// WCAG relative luminance and contrast ratio.
const lum = (rgb: number[]) => {
  const [r, g, b] = rgb.map((c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const parse = (c: string) => (c.startsWith("#") ? [1, 2, 3].map((i) => parseInt(c[i].repeat(2), 16)) : c.match(/\d+/g)!.map(Number));
const contrast = (a: string, b: string) => { const [x, y] = [lum(parse(a)), lum(parse(b))].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

describe("text on a colour map", () => {
  it("is readable on every colour of both maps, in either theme", () => {
    for (let i = 0; i <= 100; i++) {
      const s = i / 100, d = s * 2 - 1;
      expect(contrast(inkOn("sequential", s), cssColor("sequential", s)), `sequential ${s}`).toBeGreaterThanOrEqual(4.5);
      expect(contrast(inkOn("diverging", d), cssColor("diverging", d)), `diverging ${d}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
