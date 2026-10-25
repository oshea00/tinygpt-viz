<script lang="ts">
  // Attention as search, made literal: one query vector against every earlier key vector,
  // the raw dot products, the scaled scores, and the softmax weights they become.
  import { app, selection, settings } from "../../state/app.svelte";
  import { qkRows } from "../../lib/attention";
  import { cssColor, headColor } from "../../lib/colormap";
  import { fmt } from "../../lib/tensor";

  const trace = $derived(app.trace!);
  const q = $derived(selection.query(trace));
  const data = $derived(qkRows(trace, selection.layer, selection.head, q));
  const rows = $derived(settings.qkSort === "weight" ? [...data.rows].sort((a, b) => b.weight - a.weight) : data.rows);
  // One scale for every vector so their bars can be compared across rows.
  const vmax = $derived(Math.max(1e-9, ...Array.from(data.q, Math.abs), ...data.rows.flatMap((r) => Array.from(r.k, Math.abs))));
  const pmax = $derived(Math.max(1e-9, ...data.rows.flatMap((r) => r.products.map(Math.abs))));
  const maxScore = $derived(Math.max(...data.rows.map((r) => r.score)));
  const BW = 4, BH = 18;
</script>

{#snippet bars(values: ArrayLike<number>, max: number)}
  <svg width={values.length * BW} height={BH}>
    {#each Array.from(values) as v, i}
      {@const t = v / max}
      <rect x={i * BW} width={BW - 1} y={t >= 0 ? BH / 2 - (t * BH) / 2 : BH / 2} height={Math.max(0.5, Math.abs(t * BH) / 2)} fill={cssColor("diverging", t)} />
    {/each}
  </svg>
{/snippet}

<div class="top">
  <div>
    <div class="muted">query {q} “{trace.text[q]}” · layer {selection.layer} · <span style:color={headColor(selection.head)}>head {selection.head}</span></div>
    <div class="qvec">{@render bars(data.q, vmax)} <span class="muted">query vector ({data.q.length} dims)</span></div>
  </div>
  <label class="muted">order
    <select bind:value={settings.qkSort}>
      <option value="position">by position</option>
      <option value="weight">by weight</option>
    </select>
  </label>
</div>

<div class="table-wrap">
  <table class="mono">
    <thead>
      <tr><th>key</th><th>key vector</th><th>q ⊙ k</th><th>q · k</th><th>÷ √{data.q.length}</th><th>weight</th></tr>
    </thead>
    <tbody>
      {#each rows as r (r.key)}
        <tr class:sel={selection.dim === r.key} onclick={() => (selection.dim = r.key)}>
          <td class="tok">{r.key} “{trace.text[r.key]}”</td>
          <td>{@render bars(r.k, vmax)}</td>
          <td>{@render bars(r.products, pmax)}</td>
          <td class="num">{fmt(r.dot, 3)}</td>
          <td class="num" class:best={r.score === maxScore}>{fmt(r.score, 3)}</td>
          <td class="w">
            <span class="track"><span class="fill" style:width="{r.weight * 100}%" style:background={headColor(selection.head)}></span></span>
            <span class="num">{(r.weight * 100).toFixed(1)}%</span>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
<p class="muted note">Each key's score is its dot product with the query: large where their bars agree in sign.
  Softmax over the scores turns them into weights summing to 1. Keys after position {q} are masked and not listed.</p>

<style>
  .top { display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; margin-bottom: 6px; font-size: 12px; }
  .qvec { display: flex; gap: 8px; align-items: center; margin-top: 2px; }
  .table-wrap { max-height: 460px; overflow: auto; }
  table { border-collapse: collapse; font-size: 11.5px; }
  th { text-align: left; font-weight: 600; color: var(--muted); position: sticky; top: 0; background: var(--panel); padding: 2px 8px 4px 0; }
  td { padding: 1px 10px 1px 0; vertical-align: middle; }
  tr { cursor: pointer; }
  tr:hover td { background: var(--accent-soft); }
  tr.sel td { box-shadow: inset 0 1px var(--select), inset 0 -1px var(--select); }
  .tok { white-space: nowrap; }
  .num { text-align: right; }
  .best { font-weight: 700; }
  .w { display: flex; align-items: center; gap: 6px; min-width: 110px; }
  .track { flex: 1; height: 9px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .fill { display: block; height: 100%; }
  svg { display: block; }
  .note { font-size: 11px; margin: 6px 0 0; }
</style>
