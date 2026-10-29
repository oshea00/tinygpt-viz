<script lang="ts">
  // A small line chart over training steps: one y-axis, a legend, and a readout of the nearest
  // step under the pointer.
  interface Series { label: string; color: string; points: [number, number][]; dots?: boolean }
  let { series, xMax, format = (v: number) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2), label }: {
    series: Series[]; xMax: number; format?: (v: number) => string; label: string;
  } = $props();

  const W = 320, H = 140, PAD = { l: 40, r: 8, t: 8, b: 18 };
  const all = $derived(series.flatMap((s) => s.points));
  const y0 = $derived(all.length ? Math.min(...all.map((p) => p[1])) : 0);
  const y1 = $derived(all.length ? Math.max(...all.map((p) => p[1])) : 1);
  const lo = $derived(y0 - (y1 - y0 || Math.abs(y0) || 1) * 0.08), hi = $derived(y1 + (y1 - y0 || Math.abs(y1) || 1) * 0.08);
  const x1 = $derived(Math.max(1, xMax, ...all.map((p) => p[0])));
  const sx = (x: number) => PAD.l + (x / x1) * (W - PAD.l - PAD.r);
  const sy = (y: number) => PAD.t + (1 - (y - lo) / (hi - lo || 1)) * (H - PAD.t - PAD.b);
  const path = (p: [number, number][]) => p.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");

  let hoverX = $state<number | null>(null);
  function move(e: PointerEvent) {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const x = (((e.clientX - r.left) / r.width) * W - PAD.l) / (W - PAD.l - PAD.r) * x1;
    hoverX = all.length && x >= 0 && x <= x1 * 1.02 ? x : null;
  }
  // The nearest point of each series to the pointer's step.
  const readout = $derived(hoverX === null ? [] : series.map((s) => {
    let best: [number, number] | null = null;
    for (const p of s.points) if (!best || Math.abs(p[0] - hoverX!) < Math.abs(best[0] - hoverX!)) best = p;
    return { s, p: best };
  }).filter((r) => r.p));
</script>

<div class="curve">
  <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label={label} onpointermove={move} onpointerleave={() => (hoverX = null)}>
    <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} class="axis" />
    <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} class="axis" />
    {#if all.length}
      <text x={PAD.l - 4} y={sy(y1) + 3} text-anchor="end">{format(y1)}</text>
      <text x={PAD.l - 4} y={sy(y0) + 3} text-anchor="end">{format(y0)}</text>
    {/if}
    <text x={PAD.l} y={H - 4}>0</text>
    <text x={W - PAD.r} y={H - 4} text-anchor="end">step {x1}</text>
    {#each series as s}
      {#if s.points.length > 1}<path d={path(s.points)} stroke={s.color} class="line" />{/if}
      {#if s.dots}{#each s.points as [x, y]}<circle cx={sx(x)} cy={sy(y)} r="2.5" fill={s.color} class="dot" />{/each}{/if}
    {/each}
    {#if readout.length}
      <line x1={sx(readout[0].p![0])} x2={sx(readout[0].p![0])} y1={PAD.t} y2={H - PAD.b} class="cross" />
      {#each readout as r}<circle cx={sx(r.p![0])} cy={sy(r.p![1])} r="3.5" fill={r.s.color} class="dot" />{/each}
    {/if}
  </svg>
  <div class="legend">
    {#each series as s}<span><span class="sw" style:background={s.color}></span>{s.label}</span>{/each}
    {#if readout.length}
      <span class="read mono">step {readout[0].p![0]}: {readout.map((r) => `${r.s.label} ${format(r.p![1])}`).join(" · ")}</span>
    {/if}
  </div>
</div>

<style>
  .curve { min-width: 0; }
  svg { display: block; touch-action: none; }
  text { font-size: 9px; fill: var(--muted); }
  .axis { stroke: var(--line); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
  .dot { stroke: var(--panel); stroke-width: 1.5; }
  .cross { stroke: var(--muted); stroke-dasharray: 2 2; }
  .legend { font-size: 11px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; align-items: center; min-height: 16px; }
  .legend > span { display: inline-flex; gap: 4px; align-items: center; }
  .sw { display: inline-block; width: 12px; height: 3px; border-radius: 2px; }
  .read { color: var(--ink); }
</style>
