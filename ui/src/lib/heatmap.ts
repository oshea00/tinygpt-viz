// Canvas 2D heatmaps: one pixel per cell into an offscreen image, then scaled up with
// smoothing off, so a 100×256 tensor costs 25,600 pixel writes.

import type { Tensor } from "../types";
import { DIVERGING, MASKED_RGB, SEQUENTIAL, lutIndex } from "./colormap";
import { normalise, type Range } from "./scale";
import { dims2 } from "./tensor";

export interface DrawOpts {
  range: Range;
  cellW: number;
  cellH: number;
  markClipped?: boolean;
  maskUpper?: boolean;   // causal attention: cells right of the diagonal are masked
  reusedRows?: number;   // KV cache: rows before this came from the cache, drawn dimmed
}

let scratch: HTMLCanvasElement | null = null;
let hatch: CanvasPattern | null = null;

function hatchPattern(ctx: CanvasRenderingContext2D): CanvasPattern | null {
  if (hatch) return hatch;
  const c = document.createElement("canvas");
  c.width = c.height = 6;
  const g = c.getContext("2d")!;
  g.fillStyle = `rgb(${MASKED_RGB.join(",")})`;
  g.fillRect(0, 0, 6, 6);
  g.strokeStyle = "rgba(0,0,0,0.25)";
  g.beginPath();
  g.moveTo(0, 6);
  g.lineTo(6, 0);
  g.stroke();
  hatch = ctx.createPattern(c, "repeat");
  return hatch;
}

/** Draws `t` at (x0, y0) of `ctx`. Returns the drawn size. */
export function drawHeatmap(ctx: CanvasRenderingContext2D, t: Tensor, x0: number, y0: number,
                            o: DrawOpts): { w: number; h: number } {
  const [rows, cols] = dims2(t);
  const L = o.range.kind === "diverging" ? DIVERGING : SEQUENTIAL;
  scratch ??= document.createElement("canvas");
  scratch.width = cols;
  scratch.height = rows;
  const sctx = scratch.getContext("2d")!;
  const img = sctx.createImageData(cols, rows);
  const masked: number[] = [];
  const clipped: number[] = [];
  for (let i = 0; i < rows * cols; i++) {
    const v = t.data[i];
    // −∞: masked. NaN: never computed (KV cache: old queries against new keys).
    if (v === -Infinity || Number.isNaN(v) || (o.maskUpper && i % cols > Math.floor(i / cols))) {
      masked.push(i);
      continue;
    }
    const n = normalise(v, o.range);
    if (Math.abs(n) > 1.0001) clipped.push(i);
    const k = lutIndex(o.range.kind, Math.max(-1, Math.min(1, n))) * 3;
    img.data[i * 4] = L[k];
    img.data[i * 4 + 1] = L[k + 1];
    img.data[i * 4 + 2] = L[k + 2];
    img.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const w = cols * o.cellW;
  const h = rows * o.cellH;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, x0, y0, w, h);

  if (masked.length) {
    // One rectangle per run of adjacent masked cells in a row: a causal mask is one run per row.
    ctx.fillStyle = hatchPattern(ctx) ?? `rgb(${MASKED_RGB.join(",")})`;
    for (let k = 0; k < masked.length; ) {
      const start = masked[k];
      let end = start;
      while (k + 1 < masked.length && masked[k + 1] === end + 1 && (end + 1) % cols !== 0) end = masked[++k];
      k++;
      ctx.fillRect(x0 + (start % cols) * o.cellW, y0 + Math.floor(start / cols) * o.cellH, (end - start + 1) * o.cellW, o.cellH);
    }
  }
  if (o.markClipped && o.cellW >= 5 && o.cellH >= 5) {
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    for (const i of clipped) {
      ctx.fillRect(x0 + (i % cols) * o.cellW + o.cellW - 3, y0 + Math.floor(i / cols) * o.cellH + 1, 2, 2);
    }
  }
  if (o.reusedRows && o.reusedRows > 0) {
    ctx.fillStyle = REUSED;
    ctx.fillRect(x0, y0, w, Math.min(rows, o.reusedRows) * o.cellH);
  }
  return { w, h };
}

/** Grey veil over rows that came from the KV cache rather than being computed. */
export const REUSED = "rgba(150, 150, 150, 0.62)";

/** A small preview fitted to w×h, for pipeline thumbnails. */
export function drawThumb(canvas: HTMLCanvasElement, t: Tensor, range: Range, maskUpper = false, reusedRows = 0): void {
  const [rows, cols] = dims2(t);
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  drawHeatmap(ctx, t, 0, 0, { range, cellW: 1, cellH: 1, maskUpper, reusedRows });
}
