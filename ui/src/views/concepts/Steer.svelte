<script lang="ts">
  // Is there a direction for a concept, and does it change behaviour? The mean difference
  // between two groups is a candidate; adding it to the stream during generation tests it.
  import { app } from "../../state/app.svelte";
  import { concepts } from "../../concepts/store.svelte";
  import { direction, layer } from "../../lib/concepts";
  import { categoryColors } from "./colors";

  const c = $derived(concepts.current);
  const cats = $derived([...new Set(c?.items.map((i) => i.category) ?? [])].sort());
  let a = $state("");
  let b = $state("(all others)");
  $effect(() => {
    if (!cats.includes(a)) a = cats[0] ?? "";
  });
  const s = $derived(concepts.snapIndex(c));
  const stage = $derived(c?.snapshots[s] ?? "");
  const dir = $derived.by(() => {
    if (!c || !a) return null;
    const rows = layer(c, s);
    const ia = c.items.map((it, i) => (it.category === a ? i : -1)).filter((i) => i >= 0);
    const ib = c.items.map((it, i) => ((b === "(all others)" ? it.category !== a : it.category === b) ? i : -1)).filter((i) => i >= 0);
    if (!ia.length || !ib.length) return null;
    const v = direction(rows, ia, ib);
    const norm = Math.hypot(...v);
    return { v, norm, proj: rows.map((r) => r.reduce((acc, x, k) => acc + x * v[k], 0) / norm) };
  });
  const colors = $derived(categoryColors(cats));

  let prompt = $state("Once upon a time, there was a little girl. She saw a");
  let scale = $state(1);
  const W = 560, H = 24 + 16 * 6;
  const pmin = $derived(dir ? Math.min(...dir.proj) : 0), pmax = $derived(dir ? Math.max(...dir.proj) : 1);
  // Dots start right of the category labels (9px monospace, about 5.6px a character).
  const left = $derived(12 + 5.6 * Math.max(0, ...cats.map((cat) => cat.length)));
  const px = (v: number) => left + ((v - pmin) / (pmax - pmin || 1)) * (W - left - 10);

  function steerSpec() {
    return { kind: "steer" as const, stage, vector: Array.from(dir!.v), scale, label: `${a} − ${b}` };
  }
  function generate() {
    concepts.generated = {};
    app.send({ type: "generateWith", prompt, n: 40, tag: "plain" });
    app.send({ type: "generateWith", prompt, n: 40, interventions: [steerSpec()], tag: "steered" });
  }
</script>

<div class="row">
  <label>direction <select bind:value={a}>{#each cats as cat}<option value={cat}>{cat}</option>{/each}</select></label>
  <span>minus</span>
  <select bind:value={b}><option>(all others)</option>{#each cats.filter((x) => x !== a) as cat}<option value={cat}>{cat}</option>{/each}</select>
  <span class="muted">at <span class="mono">{stage}</span> (change it with the snapshot slider) · length {dir?.norm.toFixed(2)}</span>
</div>

{#if dir && c}
  <h2>Every captured word projected onto the direction</h2>
  <svg viewBox="0 0 {W} {H}" width="100%" class="strip">
    {#each cats as cat, row}<text x="2" y={20 + row * 16 + 4} class="lbl">{cat}</text>{/each}
    {#each c.items as it, i}
      <circle cx={px(dir.proj[i])} cy={20 + cats.indexOf(it.category) * 16} r="3.5" fill={colors.get(it.category)}><title>{it.word}: {dir.proj[i].toFixed(2)} · {it.prompt}</title></circle>
    {/each}
  </svg>

  <h2>Steering: add the direction to the stream during generation</h2>
  <div class="row">
    <input class="prompt" bind:value={prompt} />
    <label>scale <input type="range" min="-4" max="4" step="0.25" bind:value={scale} /> <span class="mono">{scale}</span></label>
    <button class="primary" onclick={generate}>generate both</button>
    <button onclick={() => app.addIntervention(steerSpec())} title="apply it to every view, as an intervention">add as intervention</button>
  </div>
  {#if concepts.generated.plain !== undefined || concepts.generated.steered !== undefined}
    <div class="two">
      <div><div class="muted">without steering (greedy)</div><p class="gen mono">{prompt}<b>{concepts.generated.plain ?? "…"}</b></p></div>
      <div><div class="muted">with {a} − {b} × {scale}</div><p class="gen mono">{prompt}<b>{concepts.generated.steered ?? "…"}</b></p></div>
    </div>
  {/if}
  <p class="muted note">A probe finding is a correlation; steering tests cause. If adding the direction moves the text towards the
    concept (and subtracting moves it away), the model uses that direction. Too large a scale just breaks the text.</p>
{/if}

<style>
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  .strip { max-width: 700px; display: block; margin-bottom: 10px; }
  .lbl { font: 9px var(--mono); fill: var(--muted); }
  .prompt { flex: 1; min-width: 260px; font-family: var(--mono); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .gen { white-space: pre-wrap; font-size: 12px; background: var(--bg); padding: 6px; border-radius: 6px; margin: 2px 0; }
  .note { font-size: 11px; }
</style>
