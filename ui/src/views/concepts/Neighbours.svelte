<script lang="ts">
  // What sits near a word: in the token embedding table (before any context), and among the
  // captured words at the chosen snapshot (in context).
  import { app } from "../../state/app.svelte";
  import { concepts } from "../../concepts/store.svelte";
  import { cosine, nearestRows, wordMeans } from "../../lib/concepts";

  const words = $derived([...new Set(concepts.current?.items.map((i) => i.word) ?? [])].sort());
  const vocab = $derived(app.hello?.vocab ?? []);
  const tokenId = $derived(Math.max(vocab.indexOf(`␣${concepts.word}`), vocab.indexOf(concepts.word)));
  const table = $derived.by(() => {
    const wte = app.weights?.tensors["wte"];
    if (!wte || tokenId < 0) return [];
    return nearestRows(wte.data, wte.shape[0], wte.shape[1], tokenId, 15).map((n) => ({ text: vocab[n.i], cos: n.cos }));
  });
  const inContext = $derived.by(() => {
    const c = concepts.current;
    if (!c) return [];
    const means = wordMeans(c, concepts.snapIndex(c));
    const me = means.find((m) => m.word === concepts.word);
    if (!me) return [];
    return means.filter((m) => m !== me).map((m) => ({ text: m.word, cat: m.category, cos: cosine(me.v, m.v) }))
      .sort((a, b) => b.cos - a.cos).slice(0, 12);
  });
</script>

<label class="pick">word <select bind:value={concepts.word}>{#each words as w}<option value={w}>{w}</option>{/each}</select></label>
<div class="two">
  <section>
    <h2>Token embedding table <span class="muted">(no context)</span></h2>
    {#if tokenId < 0}<p class="muted">“{concepts.word}” is not a single token in this vocabulary.</p>{/if}
    {#each table as n}
      <div class="nb"><span class="mono">{n.text}</span><span class="bar"><span style:width="{Math.max(0, n.cos) * 100}%"></span></span><span class="mono num">{n.cos.toFixed(2)}</span></div>
    {/each}
  </section>
  <section>
    <h2>Captured words at {concepts.current?.snapshots[concepts.snapIndex(concepts.current)]} <span class="muted">(in context, centred)</span></h2>
    {#each inContext as n}
      <div class="nb"><span class="mono">{n.text} <span class="muted">{n.cat}</span></span><span class="bar"><span style:width="{Math.max(0, n.cos) * 100}%"></span></span><span class="mono num">{n.cos.toFixed(2)}</span></div>
    {/each}
  </section>
</div>

<style>
  .pick { font-size: 12px; display: flex; gap: 6px; align-items: center; margin-bottom: 8px; }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  .nb { display: grid; grid-template-columns: 150px 1fr 44px; gap: 6px; align-items: center; font-size: 12px; margin: 1px 0; }
  .bar { height: 8px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .bar span { display: block; height: 100%; background: var(--accent); }
  .num { text-align: right; }
</style>
