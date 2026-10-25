<script lang="ts">
  import type { Tensor } from "../../types";
  import { drawHeatmap } from "../../lib/heatmap";
  import type { Range } from "../../lib/scale";
  import { at, dims2, fmt } from "../../lib/tensor";

  interface Props {
    tensor: Tensor;
    range: Range;
    rowLabels?: string[];
    colLabels?: string[];
    selRow?: number;
    selCol?: number;
    zoom?: number;
    maxHeight?: number;
    maxCell?: number;
    maskUpper?: boolean;
    colBands?: { start: number; end: number; color: string }[];   // e.g. each head's columns
    onpick?: (r: number, c: number) => void;
  }
  let { tensor, range, rowLabels, colLabels, selRow = -1, selCol = -1, zoom = 1, maxHeight = 620, maxCell = 28, maskUpper = false, colBands, onpick }: Props = $props();

  let width = $state(600);
  let canvas: HTMLCanvasElement;
  let hover = $state<{ r: number; c: number; x: number; y: number } | null>(null);

  const dims = $derived(dims2(tensor));
  const labelW = $derived(rowLabels ? Math.max(26, 8 + 6.5 * Math.max(...rowLabels.map((l) => l.length))) : 0);
  const cellW = $derived(Math.max(1, Math.min(maxCell, Math.floor((width - labelW) / dims[1]))) * zoom);
  const cellH = $derived(Math.max(1, Math.min(cellW / zoom, Math.max(4, Math.floor(maxHeight / dims[0])))) * zoom);
  const bandH = $derived(colBands?.length ? 6 : 0);
  // Column labels only when they fit their cells (single characters do; words usually not).
  const labelH = $derived(colLabels && cellW >= 9 && Math.max(...colLabels.map((l) => l.length)) * 5.5 <= cellW + 2 ? 14 : 0);
  const topH = $derived(bandH + labelH);
  const showRowText = $derived(cellH >= 9);

  $effect(() => {
    const [rows, cols] = dims;
    const dpr = window.devicePixelRatio || 1;
    const w = labelW + cols * cellW;
    const h = topH + rows * cellH;
    canvas.width = Math.ceil(w * dpr);
    canvas.height = Math.ceil(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drawHeatmap(ctx, tensor, labelW, topH, { range, cellW, cellH, markClipped: true, maskUpper });

    const css = getComputedStyle(canvas);
    const ink = css.getPropertyValue("--ink") || "#000";
    ctx.fillStyle = ink;
    ctx.font = `${Math.min(11, Math.max(8, cellH - 2))}px ui-monospace, monospace`;
    ctx.textBaseline = "middle";
    if (rowLabels && showRowText) {
      ctx.textAlign = "right";
      rowLabels.slice(0, rows).forEach((t, r) => ctx.fillText(t, labelW - 4, topH + r * cellH + cellH / 2));
    }
    if (colLabels && labelH) {
      ctx.textAlign = "center";
      ctx.font = "9px ui-monospace, monospace";
      colLabels.slice(0, cols).forEach((t, c) => ctx.fillText(t, labelW + c * cellW + cellW / 2, bandH + 7));
    }
    for (const b of colBands ?? []) {
      ctx.fillStyle = b.color;
      ctx.fillRect(labelW + b.start * cellW, 0, (b.end - b.start) * cellW - 1, bandH - 2);
    }
    // Selection is an outline, never a fill, so it does not compete with the data colours.
    ctx.strokeStyle = css.getPropertyValue("--select") || "#000";
    ctx.lineWidth = 1.5;
    if (selRow >= 0 && selRow < rows) ctx.strokeRect(labelW + 0.75, topH + selRow * cellH + 0.75, cols * cellW - 1.5, cellH - 1.5);
    if (selCol >= 0 && selCol < cols) {
      ctx.setLineDash([3, 2]);
      ctx.strokeRect(labelW + selCol * cellW + 0.75, topH + 0.75, cellW - 1.5, rows * cellH - 1.5);
      ctx.setLineDash([]);
    }
  });

  function cellAt(e: MouseEvent): { r: number; c: number } | null {
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left - labelW) / cellW);
    const r = Math.floor((e.clientY - rect.top - topH) / cellH);
    return r >= 0 && c >= 0 && r < dims[0] && c < dims[1] ? { r, c } : null;
  }
</script>

<div class="wrap" bind:clientWidth={width}>
  <canvas
    bind:this={canvas}
    onmousemove={(e) => {
      const p = cellAt(e);
      // Viewport coordinates, kept on screen: the tooltip is fixed-position so it never
      // overflows the scroll container (which would add scrollbars and resize the heatmap).
      hover = p ? { ...p, x: Math.min(e.clientX + 14, window.innerWidth - 240), y: Math.min(e.clientY + 12, window.innerHeight - 48) } : null;
    }}
    onmouseleave={() => (hover = null)}
    onclick={(e) => {
      const p = cellAt(e);
      if (p) onpick?.(p.r, p.c);
    }}
  ></canvas>
  {#if hover}
    <div class="tip mono" style:left="{hover.x}px" style:top="{hover.y}px">
      row {hover.r}{rowLabels ? ` "${rowLabels[hover.r]}"` : ""} · col {hover.c}{colLabels ? ` "${colLabels[hover.c]}"` : ""}<br />
      <b>{fmt(at(tensor, hover.r, hover.c), 5)}</b>
    </div>
  {/if}
</div>

<style>
  /* Horizontal scrolling only (when zoomed): a vertical scrollbar would steal width, and the
     cell size is computed from width. */
  .wrap { width: 100%; overflow-x: auto; overflow-y: hidden; }
  canvas { display: block; cursor: crosshair; }
  .tip {
    position: fixed; pointer-events: none; z-index: 50; white-space: nowrap;
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 7px; font-size: 11px; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15);
  }
</style>
