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

/** Text colour that stays readable on cssColor(kind, t), in either theme: the theme's own
 * text colour is light in dark mode and vanishes on the pale middle of a map. */
export function inkOn(kind: MapKind, t: number): string {
  const L = kind === "diverging" ? DIVERGING : SEQUENTIAL;
  const i = lutIndex(kind, t) * 3;
  const lin = (c: number) => (c / 255 <= 0.03928 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4);
  const Y = 0.2126 * lin(L[i]) + 0.7152 * lin(L[i + 1]) + 0.0722 * lin(L[i + 2]);
  // WCAG contrast against white is 1.05 / (Y + 0.05), against black (Y + 0.05) / 0.05.
  return 1.05 / (Y + 0.05) > (Y + 0.05) / 0.05 ? "#fff" : "#000";
}

/** Head identity is a hue, used the same way everywhere. */
export const HEAD_HUES = ["#d97706", "#059669", "#2563eb", "#c026d3", "#dc2626", "#0891b2", "#65a30d", "#7c3aed"];
export const headColor = (h: number) => HEAD_HUES[h % HEAD_HUES.length];
