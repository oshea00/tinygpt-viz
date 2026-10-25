<script lang="ts">
  // Where does a concept become readable? A linear probe per snapshot, trained on some words
  // and tested on held-out words, beside two baselines: shuffled labels and the untrained model.
  import { concepts } from "../../concepts/store.svelte";
  import { heldOutWords, probeByLayer } from "../../lib/concepts";

  type Row = { layer: string; testAcc: number };
  let result = $state<{ trained: Row[]; shuffled: Row[]; untrained: Row[] | null } | null>(null);
  let running = $state(false);
  const cats = $derived([...new Set(concepts.current?.items.map((i) => i.category) ?? [])]);
  const held = $derived(concepts.current ? [...heldOutWords(concepts.current.items)].sort() : []);

  function run() {
    const c = concepts.current;
    if (!c) return;
    running = true;
    // Let the button repaint before the (blocking) fits.
    setTimeout(() => {
      result = { trained: probeByLayer(c), shuffled: probeByLayer(c, true), untrained: concepts.untrained ? probeByLayer(concepts.untrained) : null };
      running = false;
    }, 30);
  }
  $effect(() => {
    void concepts.current;
    result = null;
  });

  const W = 560, H = 220, P = 30;
  const S = $derived(concepts.current?.S ?? 1);
  const bw = $derived((W - 2 * P) / S);
  const y = (v: number) => H - P - v * (H - 2 * P);
</script>

<div class="row">
  <button class="primary" disabled={running || !concepts.current} onclick={run}>{running ? "fitting…" : "fit probes at every snapshot"}</button>
  <span class="muted">{cats.length} categories · held out (never trained on): {held.join(", ")}</span>
</div>
{#if result}
  <svg viewBox="0 0 {W} {H}" width="100%" class="chart">
    {#each [0, 0.5, 1] as v}<line x1={P} x2={W - P} y1={y(v)} y2={y(v)} class="grid" /><text x={P - 4} y={y(v) + 3} text-anchor="end">{v * 100}%</text>{/each}
    <line x1={P} x2={W - P} y1={y(1 / cats.length)} y2={y(1 / cats.length)} class="chance" />
    {#each result.trained as r, s}
      {@const x0 = P + s * bw}
      <rect x={x0 + 2} y={y(r.testAcc)} width={bw / 3 - 2} height={y(0) - y(r.testAcc)} class="tr"><title>trained {r.layer}: {(r.testAcc * 100).toFixed(0)}%</title></rect>
      <rect x={x0 + bw / 3} y={y(result.shuffled[s].testAcc)} width={bw / 3 - 2} height={y(0) - y(result.shuffled[s].testAcc)} class="sh"><title>shuffled labels: {(result.shuffled[s].testAcc * 100).toFixed(0)}%</title></rect>
      {#if result.untrained}
        <rect x={x0 + (2 * bw) / 3} y={y(result.untrained[s].testAcc)} width={bw / 3 - 2} height={y(0) - y(result.untrained[s].testAcc)} class="un"><title>untrained: {(result.untrained[s].testAcc * 100).toFixed(0)}%</title></rect>
      {/if}
      {#if s % 2 === 0}<text x={x0 + bw / 2} y={H - 10} text-anchor="middle" class="lbl">{r.layer.replace(".resid_", " ").replace("embed.sum", "x₀")}</text>{/if}
    {/each}
  </svg>
  <p class="muted note"><span class="k tr">trained</span> <span class="k sh">shuffled labels</span> <span class="k un">untrained model</span>
    · dashed: chance. Accuracy on held-out words only, so a probe cannot just memorise words. A probe shows the concept is
    <i>readable</i>, not that the model uses it: test that with steering or ablation. If the untrained model also scores well,
    the probe is reading something other than learned meaning: here, each category has its own sentence templates, so
    position and context give the category away. Templates shared across categories remove that.</p>
{/if}

<style>
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .chart { max-width: 700px; }
  .grid { stroke: var(--line); }
  .chance { stroke: var(--muted); stroke-dasharray: 4 3; }
  text { font: 9px var(--mono); fill: var(--muted); }
  .tr { fill: var(--accent); } .sh { fill: #9ca3af; } .un { fill: #d97706; }
  .note { font-size: 11px; }
  .k { padding: 0 4px; border-radius: 3px; color: #fff; }
  .k.tr { background: var(--accent); } .k.sh { background: #9ca3af; } .k.un { background: #d97706; }
</style>
