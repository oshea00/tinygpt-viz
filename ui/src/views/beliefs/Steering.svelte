<script lang="ts">
  // Does pushing the stream along a word's direction change the belief, and what else moves with
  // it? Steer held-out stories with a probe direction or a mean difference, at one snapshot or a
  // band of layers, and compare the readout with the plain run. Steering every word in turn gives
  // the entanglement matrix: in the paper, words close in belief space move together.
  import { untrack } from "svelte";
  import { app } from "../../state/app.svelte";
  import { beliefs } from "../../beliefs/store.svelte";
  import {
    centroids, columnMeans, distanceMatrix, isTestStory, meanDiffDirection, meanLogDelta, meanNorm, pearson,
    readoutRows, scaledDirection, snapshotRows,
  } from "../../lib/beliefs";
  import type { Intervention } from "../../types";
  import { wordColor } from "../concepts/colors";

  const b = $derived(beliefs.current!);
  const s = $derived(beliefs.snapIndex(b));
  const layers = $derived(b.snapshots.filter((x) => x.endsWith(".resid_mlp")).length);
  let target = $state(1);
  let mode = $state<"mean" | "probe">("mean");
  let band = $state(false);
  let from = $state(1);
  let to = $state(3);
  let alpha = $state(0.5);
  let prompt = $state("Once upon a time, there was a little girl named Lily. One day, she went to the park. Everyone was very");

  const stages = $derived(band ? Array.from({ length: Math.max(0, to - from + 1) }, (_, i) => b.snapshots.indexOf(`L${from + i}.resid_mlp`)).filter((i) => i >= 0) : [s]);
  const needProbes = $derived(mode === "probe" && !beliefs.probes);
  const testStories = $derived(b.stories.filter((_, i) => isTestStory(i)));
  const where = $derived(band ? `L${from}–L${to} (after each MLP)` : b.snapshots[s]);

  function specs(k: number): Intervention[] {
    return stages.map((si) => {
      const rows = snapshotRows(b, si);
      const v = mode === "probe" ? beliefs.probes!.trained[si].fit.dirs[k] : meanDiffDirection(b, rows, k);
      return { kind: "steer", stage: b.snapshots[si], vector: scaledDirection(v, alpha, meanNorm(rows)), scale: 1,
               label: `${b.words[k]} (${mode === "probe" ? "probe" : "mean diff"}, α ${alpha})` };
    });
  }
  function steer(ks: number[]) {
    if (!beliefs.steer.plain) beliefs.steerWith("plain", testStories, []);
    for (const k of ks) beliefs.steerWith(b.words[k], testStories, specs(k));
  }

  // Steered results belong to the settings that made them; the plain run does not depend on them.
  const key = $derived(JSON.stringify([mode, band, from, to, alpha, band ? 0 : s, b.step, b.N]));
  $effect(() => {
    void key;
    untrack(() => {
      const plain = beliefs.steer.plain;
      beliefs.steer = plain ? { plain } : {};
    });
  });

  const plain = $derived(beliefs.steer.plain);
  const one = $derived(beliefs.steer[b.words[target]]);
  const delta = (k: number) => {
    const r = beliefs.steer[b.words[k]];
    return plain && r ? meanLogDelta(plain.readout, r.readout, plain.mass, r.mass) : null;
  };
  const matrix = $derived(b.words.map((_, k) => delta(k)));
  const complete = $derived(matrix.every((r) => r !== null));
  const dist = $derived(distanceMatrix(centroids(b, readoutRows(b))));
  const scatter = $derived.by(() => {
    if (!complete) return null;
    const pts: { k: number; j: number; d: number; v: number }[] = [];
    for (let k = 0; k < b.K; k++) for (let j = 0; j < b.K; j++) if (j !== k) pts.push({ k, j, d: dist[k * b.K + j], v: matrix[k]![j] });
    return { pts, r: pearson(pts.map((p) => p.d), pts.map((p) => p.v)),
             dmax: Math.max(...pts.map((p) => p.d)) || 1, vmax: Math.max(1e-6, ...pts.map((p) => Math.abs(p.v))) };
  });


  // Share chart.
  const SW = 560, SH = 24 + 20 * 6, BAR = 300;
  // Matrix.
  const C = 40, OFF = 66;
  const absMax = $derived(complete ? Math.max(1e-6, ...matrix.flatMap((r) => r!.map(Math.abs))) : 1);
  const cellFill = (v: number) => (v >= 0 ? "#2563eb" : "#dc2626");
  // Scatter.
  const PW = 360, PH = 220, PP = 34;
</script>

<div class="row">
  <label>word <select bind:value={target}>{#each b.words as w, k}<option value={k}>{w}</option>{/each}</select></label>
  <label>direction <select bind:value={mode}><option value="mean">mean difference</option><option value="probe">probe weights</option></select></label>
  <label><input type="checkbox" bind:checked={band} /> band of layers</label>
  {#if band}
    <label>from L<select bind:value={from}>{#each Array.from({ length: layers }, (_, i) => i) as l}<option value={l}>{l}</option>{/each}</select></label>
    <label>to L<select bind:value={to}>{#each Array.from({ length: layers }, (_, i) => i).filter((l) => l >= from) as l}<option value={l}>{l}</option>{/each}</select></label>
  {:else}
    <span class="muted">at <span class="mono">{b.snapshots[s]}</span> (the snapshot slider)</span>
  {/if}
  <label>α <input type="range" min="-2" max="2" step="0.05" bind:value={alpha} /> <span class="mono">{alpha}</span></label>
</div>
<div class="row">
  {#if needProbes}
    <button class="primary" onclick={() => setTimeout(() => beliefs.fitProbes(), 30)}>fit probes first</button>
  {:else}
    <button class="primary" disabled={beliefs.steerPending > 0} onclick={() => steer([target])}>steer {b.words[target]}</button>
    <button disabled={beliefs.steerPending > 0} onclick={() => steer(b.words.map((_, k) => k))}>steer each word (entanglement)</button>
    <button onclick={() => specs(target).forEach((iv) => app.addIntervention(iv))} title="apply it to every view, as an intervention">add as intervention</button>
  {/if}
  <span class="muted">{beliefs.steerPending ? "running…" : `on the ${testStories.length} held-out stories`}</span>
</div>
<p class="muted note">The direction is scaled to α times the typical length of the residual vector at each layer it is added to,
  at every position. <b>Mean difference</b>: the average vector of sentences whose strongest belief is the word, minus the rest.
  <b>Probe weights</b>: the ridge probe's weights for that word. The paper found a probe direction at a single layer barely moves
  behaviour, because later layers repair it, while a band of layers works; mean differences worked at a single layer.</p>

{#if plain && one}
  {@const before = columnMeans(plain.readout)}
  {@const after = columnMeans(one.readout)}
  {@const d = delta(target)!}
  <h2>Steering {b.words[target]} at {where} <span class="muted">average share of each word, plain and steered</span></h2>
  <svg viewBox="0 0 {SW} {SH}" width="100%" class="share">
    {#each b.words as w, k}
      {@const yy = 20 + k * 20}
      <text x="60" y={yy + 4} text-anchor="end" fill={wordColor(k)}>{w}</text>
      <rect x="66" y={yy - 7} width={before[k] * BAR} height="6" class="before"><title>plain: {(before[k] * 100).toFixed(1)}%</title></rect>
      <rect x="66" y={yy} width={after[k] * BAR} height="6" fill={wordColor(k)}><title>steered: {(after[k] * 100).toFixed(1)}%</title></rect>
      <text x={66 + BAR + 10} y={yy + 4}>{(before[k] * 100).toFixed(0)}% → {(after[k] * 100).toFixed(0)}% · Δlog p {d[k] >= 0 ? "+" : ""}{d[k].toFixed(2)}</text>
    {/each}
  </svg>
  <p class="muted note">Grey: plain · coloured: steered. Δlog p is the mean change in the log probability of the word itself (not its
    renormalised share), so words can rise or fall together.</p>
{/if}

{#if complete && scatter}
  <div class="two">
    <section>
      <h2>Entanglement <span class="muted">Δlog p of each word (column) when steering each word (row)</span></h2>
      <svg viewBox="0 0 {OFF + b.K * C + 40} {OFF + b.K * C + 4}" width="100%" class="matrix">
        {#each b.words as w, i}
          <text x={OFF - 4} y={OFF + i * C + C / 2 + 3} text-anchor="end" fill={wordColor(i)}>{w}</text>
          <text x={OFF + i * C + C / 2} y={OFF - 6} transform="rotate(-45 {OFF + i * C + C / 2} {OFF - 6})" fill={wordColor(i)}>{w}</text>
          {#each matrix[i]! as v, j}
            <rect x={OFF + j * C} y={OFF + i * C} width={C - 1} height={C - 1} fill={cellFill(v)} fill-opacity={Math.abs(v) / absMax} class="cell">
              <title>steer {b.words[i]}: Δlog p({b.words[j]}) = {v.toFixed(3)}</title>
            </rect>
            <text x={OFF + j * C + C / 2} y={OFF + i * C + C / 2 + 3} text-anchor="middle" class="val" class:light={Math.abs(v) / absMax > 0.55}>{v.toFixed(2)}</text>
          {/each}
        {/each}
      </svg>
      <p class="muted note">Blue: up · red: down. The diagonal is the intended effect; everything else is entanglement.</p>
    </section>
    <section>
      <h2>Off-target change against distance <span class="muted">r = {scatter.r.toFixed(2)}</span></h2>
      <svg viewBox="0 0 {PW} {PH}" width="100%" class="scatter">
        <line x1={PP} x2={PW - 8} y1={PH / 2} y2={PH / 2} class="grid" />
        <line x1={PP} x2={PP} y1="8" y2={PH - PP + 8} class="grid" />
        <text x={(PW + PP) / 2} y={PH - 4} text-anchor="middle">distance in belief space →</text>
        <text x="10" y={PH / 2} transform="rotate(-90 10 {PH / 2})" text-anchor="middle">Δlog p</text>
        {#each scatter.pts as p}
          <circle cx={PP + (p.d / scatter.dmax) * (PW - PP - 16)} cy={PH / 2 - (p.v / scatter.vmax) * (PH / 2 - 16)} r="4" fill={wordColor(p.j)} stroke={wordColor(p.k)} stroke-width="2">
            <title>steer {b.words[p.k]} → {b.words[p.j]}: distance {p.d.toFixed(3)}, Δlog p {p.v.toFixed(3)}</title>
          </circle>
        {/each}
      </svg>
      <p class="muted note">One dot per (steered word, other word): ring colour is the steered word, fill the one that moved. A negative
        r means close words are dragged along more than distant ones; the paper reports this correlation for emotions.</p>
    </section>
  </div>
{/if}

<h2>Generate with the steering</h2>
<div class="row">
  <input class="prompt mono" bind:value={prompt} />
  <button class="primary" disabled={needProbes} onclick={() => beliefs.generate(prompt, specs(target))}>generate both</button>
</div>
{#if beliefs.generated.plain !== undefined || beliefs.generated.steered !== undefined}
  <div class="two">
    <div><div class="muted">without steering (greedy)</div><p class="gen mono">{prompt}<b>{beliefs.generated.plain ?? "…"}</b></p></div>
    <div><div class="muted">steering {b.words[target]} at {where}, α {alpha}</div><p class="gen mono">{prompt}<b>{beliefs.generated.steered ?? "…"}</b></p></div>
  </div>
{/if}

<style>
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  .note { font-size: 11px; max-width: 820px; }
  text { font: 9px var(--mono); fill: var(--muted); }
  .share { max-width: 700px; display: block; }
  .before { fill: #9ca3af; }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; max-width: 900px; }
  .matrix { max-width: 340px; display: block; }
  .cell { stroke: var(--line); stroke-width: 0.5; }
  .val { fill: var(--ink); } .val.light { fill: #fff; }
  .scatter { max-width: 420px; display: block; background: var(--panel); border: 1px solid var(--line); }
  .grid { stroke: var(--line); }
  .prompt { flex: 1; min-width: 260px; }
  .gen { white-space: pre-wrap; font-size: 12px; background: var(--bg); padding: 6px; border-radius: 6px; margin: 2px 0; }
</style>
