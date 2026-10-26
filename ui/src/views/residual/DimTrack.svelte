<script lang="ts">
  // One residual dimension followed through the network, and which dimensions are large.
  // Most dimensions mean little on their own; a few carry values far larger than the rest,
  // which is why the concept explorer centres vectors before comparing them.
  import { app, selection } from "../../state/app.svelte";
  import { cssColor } from "../../lib/colormap";
  import { dimSizes } from "../../lib/embeddings";
  import { snapshots, writes } from "../../lib/residual";
  import { at, dims2, fmt } from "../../lib/tensor";

  const trace = $derived(app.trace!);
  const pos = $derived(selection.query(trace));
  const T = $derived(trace.tokens.length);
  const snaps = $derived(snapshots(trace));
  const ws = $derived(writes(trace));
  const d = $derived(dims2(trace.byId.get("embed.sum")!.output)[1]);
  const k = $derived(selection.trackDim >= 0 && selection.trackDim < d ? selection.trackDim : -1);

  // Outliers: mean |x| per dimension at one snapshot (the last by default).
  let sizeSnap = $state("");
  const snapId = $derived(snaps.some((s) => s.id === sizeSnap) ? sizeSnap : snaps[snaps.length - 1].id);
  const sizes = $derived(dimSizes(trace, snapId));
  const sizeMax = $derived(Math.max(...sizes));
  const median = $derived([...sizes].sort((a, b) => a - b)[Math.floor(sizes.length / 2)] || 1);
  const top = $derived([...sizes.keys()].sort((a, b) => sizes[b] - sizes[a]).slice(0, 5));
  const BW = 3, BH = 70;

  // The tracked dimension at every snapshot: one line per position.
  const CW = 560, CH = 200, L = 64, R = 10, TOP = 10, BOT = 44;
  const values = $derived(k < 0 ? [] : Array.from({ length: T }, (_, t) => snaps.map((s) => at(trace.byId.get(s.id)!.output, t, k))));
  // The y range covers the values and zero, not ±max: one sign often dominates.
  const vHi = $derived(Math.max(0, ...values.flat()));
  const vLo = $derived(Math.min(0, ...values.flat()));
  const sx = (s: number) => L + (s * (CW - L - R)) / Math.max(1, snaps.length - 1);
  const sy = (v: number) => TOP + ((vHi - v) / (vHi - vLo || 1)) * (CH - TOP - BOT);
  const line = (vs: number[]) => vs.map((v, s) => `${s ? "L" : "M"}${sx(s).toFixed(1)},${sy(v).toFixed(1)}`).join("");
  // What each sublayer wrote to this dimension at the selected position.
  const deltas = $derived(k < 0 ? [] : ws.map((w) => ({ label: w.label, v: at(trace.byId.get(w.delta)!.output, pos, k) })));
  const dMax = $derived(Math.max(1e-9, ...deltas.map((x) => Math.abs(x.v))));
</script>

<section>
  <div class="bar">
    <h2>Dimension sizes <span class="muted">mean |x| over the positions at</span>
      <select value={snapId} onchange={(e) => (sizeSnap = e.currentTarget.value)}>{#each snaps as s}<option value={s.id}>{s.label}</option>{/each}</select></h2>
  </div>
  <svg viewBox="0 0 {d * BW} {BH}" width="100%" preserveAspectRatio="none" class="sizes">
    {#each sizes as v, i (i)}
      {@const h = (v / sizeMax) * (BH - 2)}
      <rect x={i * BW} y={BH - h} width={BW - 0.6} height={h} class="bar-r" class:tracked={i === k} role="button" tabindex="-1"
            onclick={() => (selection.trackDim = i)} onkeydown={() => {}}><title>dimension {i}: {fmt(v, 3)}</title></rect>
    {/each}
  </svg>
  <p class="muted note">Largest: {#each top as i, j}<button class="link mono" onclick={() => (selection.trackDim = i)}>dim {i}</button>
    {fmt(sizes[i], 2)} (×{(sizes[i] / median).toFixed(1)} the median){j < top.length - 1 ? " · " : ""}{/each}.
    Click a bar, or a column of the stream above, to follow that dimension.</p>
</section>

{#if k >= 0}
  <section>
    <div class="bar">
      <h2>Dimension {k} through the network</h2>
      <button class="link" onclick={() => (selection.trackDim = -1)}>stop tracking ×</button>
    </div>
    <div class="two">
      <svg viewBox="0 0 {CW} {CH}" width="100%" class="chart">
        <line x1={L} x2={CW - R} y1={sy(0)} y2={sy(0)} class="zero" />
        <text x={L - 6} y={sy(vHi) + 4} class="num" text-anchor="end">{fmt(vHi, 2)}</text>
        {#if vLo < 0 && vHi > 0}<text x={L - 6} y={sy(0) + 4} class="num" text-anchor="end">0</text>{/if}
        <text x={L - 6} y={sy(vLo) + 4} class="num" text-anchor="end">{fmt(vLo, 2)}</text>
        {#each values as vs, t (t)}
          {#if t !== pos}<path d={line(vs)} stroke={cssColor("sequential", 0.3 + (0.7 * t) / Math.max(1, T - 1))} class="ln" />{/if}
        {/each}
        {#if values[pos]}<path d={line(values[pos])} class="ln sel" />{/if}
        {#each snaps as s, i (s.id)}
          <text x={sx(i)} y={CH - BOT + 14} class="num" text-anchor="end" transform="rotate(-35 {sx(i)} {CH - BOT + 14})">{s.label}</text>
        {/each}
      </svg>
      <div>
        <h2 class="sub">Written to dim {k} at position {pos} “{trace.text[pos]}”</h2>
        <svg viewBox="0 0 300 {deltas.length * 16 + 4}" width="100%" class="writes">
          <line x1="200" x2="200" y1="0" y2={deltas.length * 16 + 4} class="zero" />
          {#each deltas as w, i (w.label)}
            {@const len = (Math.abs(w.v) / dMax) * 90}
            <text x="0" y={i * 16 + 13} class="num">{w.label}</text>
            <rect x={w.v < 0 ? 200 - len : 200} y={i * 16 + 4} width={len} height="10" fill={cssColor("diverging", w.v / dMax)} />
            <text x={w.v < 0 ? 196 - len : 204 + len} y={i * 16 + 13} class="num" text-anchor={w.v < 0 ? "end" : "start"}>{fmt(w.v, 2)}</text>
          {/each}
        </svg>
      </div>
    </div>
    <p class="muted note">Left: dimension {k} of the stream at each snapshot, one line per position; the selected position is
      bold. Right: what each sublayer added to it at the selected position (the steps of the bold line). The inspector and the
      atlas also mark this dimension, in every stage that uses the stream's coordinates.</p>
  </section>
{/if}

<style>
  section { margin-bottom: 16px; }
  .bar { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .bar select { font-size: 11px; padding: 0 4px; }
  .two { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; align-items: start; }
  .sizes { height: 70px; display: block; }
  .bar-r { fill: var(--muted); opacity: 0.6; cursor: pointer; }
  .bar-r:hover { opacity: 1; }
  .bar-r.tracked { fill: var(--accent); opacity: 1; }
  .chart, .writes { max-width: 640px; display: block; }
  .zero { stroke: var(--line); }
  .ln { fill: none; stroke-width: 1; opacity: 0.5; }
  .ln.sel { stroke: var(--accent); stroke-width: 2.4; opacity: 1; }
  .num { font: 9.5px var(--mono); fill: var(--muted); }
  .sub { margin: 0 0 4px; }
  .note { font-size: 11px; margin: 4px 0 0; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 11px; }
</style>
