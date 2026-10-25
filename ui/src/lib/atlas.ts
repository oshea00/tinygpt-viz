// The network atlas: every weight and the current trace's activations as one bitmap,
// one pixel per number, laid out left to right in execution order: an input column, one
// column per block, an output column. Weights sit at the top of each column, activations
// below. Pixel position has no meaning of its own; the layout is a map.

import type { Tensor, Trace, Weights } from "../types";
import { DIVERGING, MASKED_RGB, SEQUENTIAL, lutIndex } from "./colormap";
import { normalise, tensorRange, type Range } from "./scale";
import { at, dims2, row } from "./tensor";

export interface Tile {
  kind: "weight" | "act";
  name: string;        // weight name or stage id
  label: string;
  column: number;
  x: number;
  y: number;
  rows: number;
  cols: number;
  sub?: boolean;       // a bias or norm vector drawn under its matrix (unlabelled)
}

export interface Column {
  title: string;
  x: number;
  width: number;
}

export interface Layout {
  width: number;
  height: number;
  tiles: Tile[];
  columns: Column[];
  actTop: number;      // y where the activation section starts
}

const GAP = 6;         // between tiles in a row
const LABEL = 12;      // space above each shelf for tile labels
const COL_GAP = 40;
const TITLE = 18;
const SECTION_GAP = 30;

interface Item { kind: Tile["kind"]; name: string; label: string; rows: number; cols: number; sub?: boolean; }
type Group = Item[];   // stacked vertically: a matrix with its bias under it, or a norm's γ and β

/** Weight groups per column, in reading order. */
function weightColumns(w: Weights, nLayer: number): Group[][] {
  const item = (name: string, label: string, sub = false): Item[] => {
    const t = w.tensors[name];
    if (!t) return [];
    const [rows, cols] = dims2(t);
    return [{ kind: "weight", name, label, rows, cols, sub }];
  };
  const group = (names: string[], label: string): Group[] => {
    const g = names.flatMap((n, i) => item(n, label, i > 0));
    return g.length ? [g] : [];
  };
  const input = [...group(["wte"], "token table wte"), ...group(["wpe"], "position table wpe")];
  const blocks = Array.from({ length: nLayer }, (_, l) => {
    const P = `L${l}.`;
    return [
      ...group([P + "ln1_g", P + "ln1_b"], "LN1 γ β"), ...group([P + "W_q", P + "b_q"], "W_Q"),
      ...group([P + "W_k", P + "b_k"], "W_K"), ...group([P + "W_v", P + "b_v"], "W_V"),
      ...group([P + "W_o", P + "b_o"], "W_O"), ...group([P + "ln2_g", P + "ln2_b"], "LN2 γ β"),
      ...group([P + "W_up", P + "b_up"], "W_up"), ...group([P + "W_down", P + "b_down"], "W_down"),
    ];
  });
  const output = [...group(["ln_f_g", "ln_f_b"], "final LN γ β"), ...group(["W_unembed"], "unembedding W_U")];
  return [input, ...blocks, output];
}

/** Activation stages per column (per-head splits and masked scores are left out: they repeat other tiles). */
function actColumns(trace: Trace, nLayer: number, nHead: number): Group[][] {
  const item = (id: string, label = id.replace(/^L\d+\./, "")): Item[] => {
    const s = trace.byId.get(id);
    if (!s) return [];
    const [rows, cols] = dims2(s.output);
    return [{ kind: "act", name: id, label, rows, cols }];
  };
  const input = ["embed.tok", "embed.pos", "embed.sum"].flatMap((id) => item(id));
  const blocks = Array.from({ length: nLayer }, (_, l) => {
    const P = `L${l}.`;
    return [
      ...item(P + "ln1"), ...item(P + "q"), ...item(P + "k"), ...item(P + "v"),
      ...Array.from({ length: nHead }, (_, h) => item(`${P}attn.H${h}.weights`, `H${h} attention`)).flat(),
      ...item(P + "attn.concat"), ...item(P + "attn.out"), ...item(P + "resid_attn", "+ attention"),
      ...item(P + "ln2"), ...item(P + "mlp.up"), ...item(P + "mlp.gelu"), ...item(P + "mlp.down"), ...item(P + "resid_mlp", "+ MLP"),
    ];
  });
  const output = ["ln_f", "logits", "probs"].flatMap((id) => item(id));
  return [input, ...blocks, output].map((items) => items.map((it) => [it]));
}

const tileH = (rows: number) => Math.max(rows, 2);   // vectors are drawn 2 px tall so they stay visible
const groupW = (g: Group) => Math.max(...g.map((t) => t.cols));
const groupH = (g: Group) => g.reduce((h, t) => h + tileH(t.rows), 0) + (g.length - 1);

/** Shelf packing: groups go left to right while they fit the column width, then a new shelf. */
function pack(groups: Group[], x0: number, y0: number, width: number, column: number): { tiles: Tile[]; bottom: number } {
  const tiles: Tile[] = [];
  let x = 0;
  let y = y0 + LABEL;
  let shelf = 0;
  for (const g of groups) {
    if (x > 0 && x + groupW(g) > width) {
      y += shelf + LABEL + GAP;
      x = 0;
      shelf = 0;
    }
    let gy = y;
    for (const it of g) {
      tiles.push({ ...it, column, x: x0 + x, y: gy });
      gy += tileH(it.rows) + 1;
    }
    x += groupW(g) + GAP;
    shelf = Math.max(shelf, groupH(g));
  }
  return { tiles, bottom: y + shelf };
}

export function buildLayout(trace: Trace, weights: Weights, nLayer: number, nHead: number): Layout {
  const wcols = weightColumns(weights, nLayer);
  const acols = actColumns(trace, nLayer, nHead);
  const titles = ["input", ...Array.from({ length: nLayer }, (_, l) => `block ${l}`), "output"];
  // Wide enough for the widest tile, and for Q, K, V and O side by side.
  const d = weights.tensors["wte"] ? dims2(weights.tensors["wte"])[1] : 0;
  const widths = wcols.map((w, i) => Math.max(...w.map(groupW), ...acols[i].map(groupW), i > 0 && i <= nLayer ? 4 * d + 3 * GAP : 0));
  const tiles: Tile[] = [];
  const columns: Column[] = [];
  let x = 0;
  let weightBottom = 0;
  const packedW = wcols.map((items, i) => {
    const p = pack(items, x, TITLE, widths[i], i);
    columns.push({ title: titles[i], x, width: widths[i] });
    x += widths[i] + COL_GAP;
    weightBottom = Math.max(weightBottom, p.bottom);
    return p;
  });
  packedW.forEach((p) => tiles.push(...p.tiles));
  const actTop = weightBottom + SECTION_GAP;
  let height = actTop;
  acols.forEach((items, i) => {
    const p = pack(items, columns[i].x, actTop, widths[i], i);
    tiles.push(...p.tiles);
    height = Math.max(height, p.bottom);
  });
  return { width: x - COL_GAP, height: height + 4, tiles, columns, actTop };
}

export function hitTest(layout: Layout, x: number, y: number): { tile: Tile; r: number; c: number } | null {
  for (const tile of layout.tiles) {
    const h = Math.max(tile.rows, 2);
    if (x >= tile.x && x < tile.x + tile.cols && y >= tile.y && y < tile.y + h) {
      const r = tile.rows === 1 ? 0 : Math.floor(y - tile.y);
      return { tile, r, c: Math.floor(x - tile.x) };
    }
  }
  return null;
}

export type AtlasMode = "values" | "grads" | "flow" | "diff" | "recent";

export interface AtlasSources {
  trace: Trace;
  weights: Weights;
  mode: AtlasMode;
  position: number;              // for weight × input
  pinnedTrace?: Trace | null;    // for diff
  pinnedWeights?: Weights | null;
  prevWeights?: Weights | null;  // for recent
}

/** The stage that multiplies its input by weight `name`, if any (for weight × input). */
function stageUsing(trace: Trace, name: string) {
  return trace.stages.find((s) => (s.kind === "linear" || s.kind === "logits") && s.weights[0] === name);
}

function subtract(a: Tensor, b: Tensor | undefined): Tensor | null {
  if (!b || a.data.length !== b.data.length) return null;
  return { shape: a.shape, data: a.data.map((v, i) => v - b.data[i]) };
}

/** The tensor a tile shows in a mode, or null if there is nothing to show (drawn hatched). */
export function tileTensor(tile: Tile, s: AtlasSources): Tensor | null {
  if (tile.kind === "act") {
    const st = s.trace.byId.get(tile.name)!;
    if (s.mode === "grads") return st.grad ?? null;
    if (s.mode === "diff") return subtract(st.output, s.pinnedTrace?.byId.get(tile.name)?.output);
    return st.output;
  }
  const w = s.weights.tensors[tile.name];
  switch (s.mode) {
    case "grads": return s.trace.weightGrads[tile.name] ?? null;
    case "diff": return subtract(w, s.pinnedWeights?.tensors[tile.name]);
    case "recent": return subtract(w, s.prevWeights?.tensors[tile.name]);
    case "flow": {
      // What flows through each connection for one position: x[i] · W[i, j].
      const st = stageUsing(s.trace, tile.name);
      if (!st || w.shape.length !== 2) return w;
      const x = row(s.trace.byId.get(st.inputs[0])!.output, Math.min(s.position, s.trace.tokens.length - 1));
      const [rows, cols] = dims2(w);
      const data = new Float32Array(rows * cols);
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) data[i * cols + j] = x[i] * w.data[i * cols + j];
      return { shape: w.shape, data };
    }
    default: return w;
  }
}

export function tileRange(tile: Tile, t: Tensor, mode: AtlasMode, s: AtlasSources): Range {
  const seq = tile.kind === "act" && (mode === "values" || mode === "flow") && s.trace.byId.get(tile.name)!.kind.match(/softmax|probs/);
  return tensorRange([t.data], seq ? "sequential" : "diverging", 99, false, "this tile", 4000);
}

/** Render every tile into an RGBA bitmap (alpha 0 between tiles). */
export interface Rendered {
  rgba: Uint8ClampedArray;
  ranges: Map<Tile, Range | null>;
  tensors: Map<Tile, Tensor | null>;   // what each tile shows, for hover read-outs
}

export function renderAtlas(layout: Layout, s: AtlasSources): Rendered {
  const { width: W, height: H } = layout;
  const rgba = new Uint8ClampedArray(W * H * 4);
  const ranges = new Map<Tile, Range | null>();
  const tensors = new Map<Tile, Tensor | null>();
  for (const tile of layout.tiles) {
    const t = tileTensor(tile, s);
    tensors.set(tile, t);
    const h = Math.max(tile.rows, 2);
    if (!t) {
      ranges.set(tile, null);
      for (let y = 0; y < h; y++) for (let x = 0; x < tile.cols; x++) {
        const o = ((tile.y + y) * W + tile.x + x) * 4;
        const stripe = (x + y) % 6 < 2 ? 0.75 : 1;
        rgba[o] = MASKED_RGB[0] * stripe; rgba[o + 1] = MASKED_RGB[1] * stripe; rgba[o + 2] = MASKED_RGB[2] * stripe; rgba[o + 3] = 255;
      }
      continue;
    }
    const range = tileRange(tile, t, s.mode, s);
    ranges.set(tile, range);
    const L = range.kind === "diverging" ? DIVERGING : SEQUENTIAL;
    const isMaskedAttn = tile.kind === "act" && /\.weights$/.test(tile.name);
    for (let y = 0; y < h; y++) {
      const r = tile.rows === 1 ? 0 : y;
      for (let x = 0; x < tile.cols; x++) {
        const v = at(t, r, x);
        const o = ((tile.y + y) * W + tile.x + x) * 4;
        if (v === -Infinity || (isMaskedAttn && x > r)) {
          rgba[o] = MASKED_RGB[0]; rgba[o + 1] = MASKED_RGB[1]; rgba[o + 2] = MASKED_RGB[2];
        } else {
          const k = lutIndex(range.kind, Math.max(-1, Math.min(1, normalise(v, range)))) * 3;
          rgba[o] = L[k]; rgba[o + 1] = L[k + 1]; rgba[o + 2] = L[k + 2];
        }
        rgba[o + 3] = 255;
      }
    }
  }
  return { rgba, ranges, tensors };
}
