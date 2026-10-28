<script lang="ts">
  // Does the residual stream lay beliefs out the way behaviour does? One centroid per word in each
  // space (the mean over the sentences that express it most), the distances between them, and the
  // correlation of the two distance matrices at every snapshot.
  import { beliefs } from "../../beliefs/store.svelte";
  import { defaultTopK, geometryByLayer } from "../../lib/beliefs";
  import { wordColor } from "../concepts/colors";

  const b = $derived(beliefs.current!);
  const u = $derived(beliefs.untrained && beliefs.untrained.N === b.N ? beliefs.untrained : null);
  const s = $derived(beliefs.snapIndex(b));
  const geo = $derived(geometryByLayer(b));
  const geoU = $derived(u ? geometryByLayer(b, u) : null);
  const K = $derived(b.K);

  const nearest = (m: Float64Array, i: number) => {
    let best = -1;
    for (let j = 0; j < K; j++) if (j !== i && (best < 0 || m[i * K + j] < m[i * K + best])) best = j;
    return best;
  };

  const C = 34, OFF = 66;
  const MW = $derived(OFF + K * C + 40);   // room for the last rotated column label
  const W = 560, H = 200, P = 34;
  const bw = $derived((W - 2 * P) / b.S);
  const y = (v: number) => H / 2 - v * (H / 2 - P / 2);
</script>

<p class="muted note">Each word's centroid is the mean over the {defaultTopK(b)} sentences with the highest share of it, at most 3
  from any one story. Belief-space centroids are 6-number readouts; residual-stream centroids are full vectors at the snapshot.</p>

<div class="two">
  {#each [["belief space (readout)", geo.behaviour], [`residual stream at ${b.snapshots[s]}`, geo.layers[s].activation]] as const as [title, m]}
    {@const max = Math.max(...m)}
    <section>
      <h2>{title}</h2>
      <svg viewBox="0 0 {MW} {OFF + K * C + 4}" width="100%" class="matrix">
        {#each b.words as w, i}
          <text x={OFF - 4} y={OFF + i * C + C / 2 + 3} text-anchor="end" fill={wordColor(i)}>{w}</text>
          <text x={OFF + i * C + C / 2} y={OFF - 6} text-anchor="start" transform="rotate(-45 {OFF + i * C + C / 2} {OFF - 6})" fill={wordColor(i)}>{w}</text>
          {#each b.words as _, j}
            {@const v = m[i * K + j] / (max || 1)}
            <rect x={OFF + j * C} y={OFF + i * C} width={C - 1} height={C - 1} class="cell" style:fill-opacity={1 - v}>
              <title>{b.words[i]} ↔ {b.words[j]}: {m[i * K + j].toFixed(3)}</title>
            </rect>
            {#if i !== j}<text x={OFF + j * C + C / 2} y={OFF + i * C + C / 2 + 3} text-anchor="middle" class="val" class:light={v < 0.45}>{v.toFixed(2)}</text>{/if}
          {/each}
        {/each}
      </svg>
      <p class="muted near">nearest: {b.words.map((w, i) => `${w} → ${b.words[nearest(m, i)]}`).join(" · ")}</p>
    </section>
  {/each}
</div>
<p class="muted note">Distances are scaled so the largest is 1; darker is closer.</p>

<h2>Agreement at each snapshot <span class="muted">Pearson r between the two distance matrices (pairs of words) · at {b.snapshots[s]}: {geo.layers[s].r.toFixed(2)}</span></h2>
<svg viewBox="0 0 {W} {H}" width="100%" class="chart">
  {#each [-1, 0, 1] as v}<line x1={P} x2={W - P} y1={y(v)} y2={y(v)} class="grid" /><text x={P - 4} y={y(v) + 3} text-anchor="end">{v}</text>{/each}
  {#each geo.layers as l, i}
    {@const x0 = P + i * bw}
    {#if i === s}<rect x={x0} y={P / 2} width={bw} height={H - P} class="sel" />{/if}
    <rect x={x0 + 2} y={Math.min(y(l.r), y(0))} width={bw / 2 - 2} height={Math.abs(y(l.r) - y(0))} class="tr"><title>trained {l.layer}: r = {l.r.toFixed(2)}</title></rect>
    {#if geoU}
      {@const g = geoU.layers[i]}
      <rect x={x0 + bw / 2} y={Math.min(y(g.r), y(0))} width={bw / 2 - 2} height={Math.abs(y(g.r) - y(0))} class="un"><title>untrained {g.layer}: r = {g.r.toFixed(2)}</title></rect>
    {/if}
    {#if i % 2 === 0}<text x={x0 + bw / 2} y={H - 4} text-anchor="middle">{l.layer.replace(".resid_", " ").replace("embed.sum", "x₀")}</text>{/if}
  {/each}
</svg>
<p class="muted note"><span class="k tr">trained</span> {#if geoU}<span class="k un">untrained model's vectors</span>{/if}
  · r near 1: words that behaviour treats as close (sad and scared, say) are also close in the stream. With six words there are only
  15 pairs, so treat r as a rough guide. The paper reports r = .92 for emotions in an 8-billion-parameter model.</p>

<style>
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; max-width: 760px; }
  .matrix { max-width: 340px; display: block; }
  .cell { fill: var(--accent); stroke: var(--line); stroke-width: 0.5; }
  text { font: 9px var(--mono); fill: var(--muted); }
  .val { fill: var(--ink); } .val.light { fill: #fff; }
  .near { font-size: 11px; margin: 2px 0; }
  .chart { max-width: 700px; }
  .grid { stroke: var(--line); }
  .sel { fill: var(--accent-soft); }
  .tr { fill: var(--accent); } .un { fill: #d97706; }
  .note { font-size: 11px; max-width: 760px; }
  .k { padding: 0 4px; border-radius: 3px; color: #fff; }
  .k.tr { background: var(--accent); } .k.un { background: #d97706; }
</style>
