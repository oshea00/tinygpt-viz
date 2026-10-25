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
    onpick?: (r: number, c: number) => void;
  }
  let { tensor, range, rowLabels, colLabels, selRow = -1, selCol = -1, zoom = 1, maxHeight = 620, maxCell = 28, maskUpper = false, onpick }: Props = $props();

  let width = $state(600);
  let canvas: HTMLCanvasElement;
  let hover = $state<{ r: number; c: number; x: number; y: number } | null>(null);

  const dims = $derived(dims2(tensor));
  const labelW = $derived(rowLabels ? 26 : 0);
  const cellW = $derived(Math.max(1, Math.min(maxCell, Math.floor((width - labelW) / dims[1]))) * zoom);
  const cellH = $derived(Math.max(1, Math.min(cellW / zoom, Math.max(4, Math.floor(maxHeight / dims[0])))) * zoom);
  const topH = $derived(colLabels && cellW >= 9 ? 14 : 0);
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
    if (colLabels && topH) {
      ctx.textAlign = "center";
      ctx.font = "9px ui-monospace, monospace";
      colLabels.slice(0, cols).forEach((t, c) => ctx.fillText(t, labelW + c * cellW + cellW / 2, 7));
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
      hover = p ? { ...p, x: e.offsetX, y: e.offsetY } : null;
    }}
    onmouseleave={() => (hover = null)}
    onclick={(e) => {
      const p = cellAt(e);
      if (p) onpick?.(p.r, p.c);
    }}
  ></canvas>
  {#if hover}
    <div class="tip mono" style:left="{hover.x + 14}px" style:top="{hover.y + 10}px">
      row {hover.r}{rowLabels ? ` "${rowLabels[hover.r]}"` : ""} · col {hover.c}{colLabels ? ` "${colLabels[hover.c]}"` : ""}<br />
      <b>{fmt(at(tensor, hover.r, hover.c), 5)}</b>
    </div>
  {/if}
</div>

<style>
  .wrap { position: relative; width: 100%; overflow: auto; }
  canvas { display: block; cursor: crosshair; }
  .tip {
    position: absolute; pointer-events: none; z-index: 5; white-space: nowrap;
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 3px 7px; font-size: 11px; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15);
  }
</style>
