<script lang="ts">
  // Which words does the model treat as alike, at this snapshot? Cosine similarity of
  // centred word vectors, grouped by category, beside the untrained model's.
  import { concepts } from "../../concepts/store.svelte";
  import { similarityMatrix, wordMeans, type Capture } from "../../lib/concepts";
  import type { Range } from "../../lib/scale";
  import Heatmap from "../inspector/Heatmap.svelte";

  const range: Range = { kind: "diverging", lo: -1, hi: 1, symlog: false, label: "cosine" };
  function view(c: Capture | null) {
    if (!c) return null;
    const means = wordMeans(c, concepts.snapIndex(c));
    return { words: means.map((m) => m.word), cats: means.map((m) => m.category), t: { shape: [means.length, means.length], data: similarityMatrix(means.map((m) => m.v)) } };
  }
  const trained = $derived(view(concepts.current));
  const base = $derived(view(concepts.untrained));
  // Mean similarity within categories vs across them: a one-number summary of the picture.
  function contrast(v: ReturnType<typeof view>) {
    if (!v) return null;
    let win = 0, nw = 0, xs = 0, nx = 0;
    const n = v.words.length;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (v.cats[i] === v.cats[j]) { win += v.t.data[i * n + j]; nw++; } else { xs += v.t.data[i * n + j]; nx++; }
    }
    return { within: win / nw, across: xs / nx };
  }
</script>

<div class="two">
  {#each [["trained model", trained], ["untrained (step 0)", base]] as const as [title, v]}
    {#if v}
      {@const k = contrast(v)}
      <section>
        <h2>{title}</h2>
        <Heatmap tensor={v.t} {range} rowLabels={v.words} colLabels={v.words} maxCell={12} maxHeight={560} />
        {#if k}<p class="muted note">mean cosine within a category <b class="mono">{k.within.toFixed(2)}</b> · across categories <b class="mono">{k.across.toFixed(2)}</b></p>{/if}
      </section>
    {/if}
  {/each}
</div>
<p class="muted note">Rows and columns are words, sorted by category, each averaged over its prompts and centred (the mean over
  all captured words is subtracted, so the shared component does not make everything look alike). Structure the untrained
  model also shows comes from the inputs, not from learning.</p>

<style>
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  .note { font-size: 11px; margin: 4px 0; }
</style>
