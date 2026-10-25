<script lang="ts">
  // The residual stream as a vertical spine; attention and MLP branch off it and rejoin
  // at an add node. Each stage carries a live thumbnail of its output (or gradient).
  import type { Tensor } from "../../types";
  import { app, selection } from "../../state/app.svelte";
  import { headColor } from "../../lib/colormap";
  import { drawThumb } from "../../lib/heatmap";
  import { pipelineRows } from "../../lib/pipeline";
  import { isSequential, tensorRange } from "../../lib/scale";
  import { fmt } from "../../lib/tensor";

  const rows = $derived(app.trace ? pipelineRows(app.trace) : []);
  const backward = $derived(selection.phase === "backward");

  function thumb(canvas: HTMLCanvasElement, p: { t: Tensor; seq: boolean; mask: boolean }) {
    const draw = (q: typeof p) => drawThumb(canvas, q.t, tensorRange([q.t.data], q.seq ? "sequential" : "diverging", 99, false, "", THUMB_SAMPLE), q.mask);
    draw(p);
    return { update: draw };
  }

  const ROW_H = 26;
  const THUMB_SAMPLE = 2000;   // enough for a 30-pixel preview; these redraw on every trace
</script>

<div class="map">
  {#each rows as row, i (row.key)}
    {#if i === 0 || rows[i - 1].group !== row.group}
      <div class="group">{row.group}</div>
    {/if}
    <div class="row" class:spine={row.lane === "spine"}>
      <svg class="gutter" width="34" height={ROW_H}>
        <line x1="9" x2="9" y1="0" y2={ROW_H} class="spine-line" />
        {#if row.lane === "spine"}
          {#if rows[i].stages[0].kind === "add"}
            <circle cx="9" cy={ROW_H / 2} r="6" class="add" />
            <path d="M5 {ROW_H / 2}h8M9 {ROW_H / 2 - 4}v8" class="plus" />
          {:else}
            <circle cx="9" cy={ROW_H / 2} r="3.5" class="node" />
          {/if}
        {:else}
          <line x1="26" x2="26" y1={row.segStart ? ROW_H / 2 : 0} y2={row.segEnd ? ROW_H / 2 : ROW_H} class="branch-line" />
          {#if row.segStart}<path d="M9 2 Q 9 {ROW_H / 2} 26 {ROW_H / 2}" class="branch-line" />{/if}
          {#if row.segEnd}<path d="M26 {ROW_H / 2} Q 9 {ROW_H / 2} 9 {ROW_H}" class="branch-line" />{/if}
          <circle cx="26" cy={ROW_H / 2} r="2.5" class="node" />
        {/if}
      </svg>
      <div class="label" title={row.stages.map((s) => s.id).join(", ")}>{row.label}</div>
      <div class="thumbs">
        {#each row.stages as s (s.id)}
          {@const t = backward ? s.grad : s.output}
          {@const selected = selection.stageId === s.id}
          <button
            class="thumb"
            class:selected
            style:border-color={s.head !== undefined && !selected ? headColor(s.head) : undefined}
            title={s.id}
            onclick={() => selection.select(s.id)}
          >
            {#if !t}
              <span class="none">–</span>
            {:else if s.kind === "loss"}
              <span class="mono val">{fmt(t.data[0], 3)}</span>
            {:else}
              <canvas use:thumb={{ t, seq: isSequential(s, backward), mask: s.kind === "mask" || s.kind === "softmax" }}></canvas>
            {/if}
          </button>
        {/each}
      </div>
    </div>
  {/each}
</div>

<style>
  .map { font-size: 12px; }
  .group { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; padding: 8px 0 2px 2px; }
  .row { display: flex; align-items: center; height: 26px; }
  .gutter { flex: none; overflow: visible; }
  .spine-line { stroke: var(--ink); stroke-width: 2.5; opacity: 0.35; }
  .branch-line { stroke: var(--accent); stroke-width: 1.5; fill: none; opacity: 0.7; }
  .add { fill: var(--panel); stroke: var(--ink); stroke-width: 1.5; }
  .plus { stroke: var(--ink); stroke-width: 1.5; }
  .node { fill: var(--ink); opacity: 0.6; }
  .label { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 4px; }
  .spine .label { font-weight: 600; }
  .thumbs { display: flex; gap: 3px; flex: none; }
  .thumb { padding: 0; width: 30px; height: 22px; border: 1.5px solid var(--line); border-radius: 3px; overflow: hidden; display: grid; place-items: center; background: var(--panel); }
  .thumb.selected { border-color: var(--select); box-shadow: 0 0 0 1.5px var(--select); }
  canvas { width: 100%; height: 100%; image-rendering: pixelated; display: block; }
  .val { font-size: 9px; }
  .none { color: var(--muted); }
</style>
