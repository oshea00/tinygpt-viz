// One colour language for every view (PROJECT.md, "Visual encoding rules"):
// signed values blue–white–red centred on zero, probabilities white → dark purple,
// masked cells a neutral hatch, never a colour.

import { interpolatePurples, interpolateRdBu } from "d3-scale-chromatic";
import { rgb } from "d3-color";

export type MapKind = "diverging" | "sequential";

function lut(f: (t: number) => string): Uint8ClampedArray {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = rgb(f(i / 255));
    out.set([c.r, c.g, c.b], i * 3);
  }
  return out;
}

// RdBu runs red → blue; flip it so negative is blue and positive is red.
export const DIVERGING = lut((t) => interpolateRdBu(1 - t));
export const SEQUENTIAL = lut((t) => interpolatePurples(0.02 + t * 0.98));
export const MASKED_RGB = [200, 200, 200];

/** Index into a LUT for a value already normalised to [-1, 1] (diverging) or [0, 1]. */
export function lutIndex(kind: MapKind, t: number): number {
  const u = kind === "diverging" ? (t + 1) / 2 : t;
  return Math.max(0, Math.min(255, Math.round(u * 255)));
}

export function cssColor(kind: MapKind, t: number): string {
  const L = kind === "diverging" ? DIVERGING : SEQUENTIAL;
  const i = lutIndex(kind, t) * 3;
  return `rgb(${L[i]},${L[i + 1]},${L[i + 2]})`;
}

/** Head identity is a hue, used the same way everywhere. */
export const HEAD_HUES = ["#d97706", "#059669", "#2563eb", "#c026d3", "#dc2626", "#0891b2", "#65a30d", "#7c3aed"];
export const headColor = (h: number) => HEAD_HUES[h % HEAD_HUES.length];
