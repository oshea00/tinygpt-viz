<script lang="ts">
  let { train, val }: { train: [number, number][]; val: [number, number][] } = $props();

  const W = 300, H = 130, PAD = { l: 34, r: 8, t: 8, b: 18 };

  // Bucket to at most ~300 points, averaging, so a long run stays cheap to draw.
  const pts = $derived.by(() => {
    if (train.length <= 300) return train;
    const n = Math.ceil(train.length / 300);
    const out: [number, number][] = [];
    for (let i = 0; i < train.length; i += n) {
      const b = train.slice(i, i + n);
      out.push([b[b.length - 1][0], b.reduce((s, p) => s + p[1], 0) / b.length]);
    }
    return out;
  });
  const all = $derived([...pts, ...val]);
  const x0 = $derived(all.length ? Math.min(...all.map((p) => p[0])) : 0);
  const x1 = $derived(all.length ? Math.max(x0 + 1, ...all.map((p) => p[0])) : 1);
  const y1 = $derived(all.length ? Math.max(...all.map((p) => p[1])) * 1.05 : 4);
  const y0 = $derived(all.length ? Math.max(0, Math.min(...all.map((p) => p[1])) * 0.9) : 0);
  const sx = (x: number) => PAD.l + ((x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (y: number) => PAD.t + (1 - (y - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b);
  const path = (p: [number, number][]) => p.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
</script>

<svg width="100%" viewBox="0 0 {W} {H}" role="img" aria-label="loss curve">
  <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} class="axis" />
  <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} class="axis" />
  <text x={PAD.l - 4} y={sy(y1) + 4} text-anchor="end">{y1.toFixed(2)}</text>
  <text x={PAD.l - 4} y={sy(y0)} text-anchor="end">{y0.toFixed(2)}</text>
  <text x={PAD.l} y={H - 4}>{x0}</text>
  <text x={W - PAD.r} y={H - 4} text-anchor="end">step {x1}</text>
  {#if pts.length > 1}<path d={path(pts)} class="train" />{/if}
  {#if val.length > 1}<path d={path(val)} class="val" />{/if}
  {#each val as [x, y]}<circle cx={sx(x)} cy={sy(y)} r="2" class="valdot" />{/each}
</svg>
<div class="legend"><span class="sw train-sw"></span>train <span class="sw val-sw"></span>validation</div>

<style>
  svg { display: block; }
  text { font-size: 9px; fill: var(--muted); }
  .axis { stroke: var(--line); }
  .train { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  .val { fill: none; stroke: #d97706; stroke-width: 1.5; }
  .valdot { fill: #d97706; }
  .legend { font-size: 11px; color: var(--muted); display: flex; gap: 6px; align-items: center; }
  .sw { display: inline-block; width: 12px; height: 3px; border-radius: 2px; margin-left: 6px; }
  .train-sw { background: var(--accent); }
  .val-sw { background: #d97706; }
</style>
