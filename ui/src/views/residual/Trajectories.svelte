<script lang="ts">
  // Each position's vector at every snapshot, projected into one shared 2D view and drawn as a
  // path from the embedding to the last layer. Paths that run together are positions the
  // model is treating alike at that depth.
  import { app, selection, settings } from "../../state/app.svelte";
  import { cssColor } from "../../lib/colormap";
  import { fitBox, trajectories } from "../../lib/embeddings";

  const W = 620, H = 380, PAD = 22;
  const trace = $derived(app.trace!);
  const T = $derived(trace.tokens.length);
  const pos = $derived(selection.query(trace));
  const tr = $derived(T >= 3 ? trajectories(trace, settings.trajNormalise) : null);
  const S = $derived(tr?.labels.length ?? 0);
  // Fit all points together, then cut them back into one path per position.
  const paths = $derived.by(() => {
    if (!tr) return [];
    const flat = fitBox(tr.paths.flat(), W, H, PAD);
    return tr.paths.map((_, t) => flat.slice(t * S, (t + 1) * S));
  });
  const d = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const colour = (t: number) => cssColor("sequential", 0.3 + (0.7 * t) / Math.max(1, T - 1));
</script>

<section>
  <div class="bar">
    <h2>Trajectories through the stream
      {#if tr}<span class="muted">PC1 {(tr.explained[0] * 100).toFixed(0)}% · PC2 {(tr.explained[1] * 100).toFixed(0)}% of variance</span>{/if}</h2>
    <label class="opt"><input type="checkbox" bind:checked={settings.trajNormalise} /> normalise each snapshot</label>
  </div>
  {#if !tr}
    <p class="muted">Needs at least 3 tokens in the prompt.</p>
  {:else}
    <svg viewBox="0 0 {W} {H}" width="100%">
      <rect x="0" y="0" width={W} height={H} class="bg" />
      {#each paths as p, t (t)}
        {#if t !== pos}
          <path d={d(p)} stroke={colour(t)} class="path" role="button" tabindex="-1"
                onclick={() => (selection.position = t)} onkeydown={() => {}}><title>position {t} “{trace.text[t]}”</title></path>
        {/if}
      {/each}
      {#if paths[pos]}
        {@const p = paths[pos]}
        <path d={d(p)} stroke={colour(pos)} class="path sel" />
        {#each p as [x, y], s (s)}
          <circle cx={x} cy={y} r={s === 0 ? 3.5 : 2.8} fill={s === 0 ? "var(--panel)" : colour(pos)} stroke={colour(pos)} class="mark">
            <title>position {pos} “{trace.text[pos]}” at {tr.labels[s]}</title>
          </circle>
        {/each}
        <text x={p[0][0] + 6} y={p[0][1] - 6} class="lbl muted-lbl">x₀</text>
        <text x={p[S - 1][0] + 6} y={p[S - 1][1] + 4} class="lbl">{trace.text[pos]}</text>
      {/if}
    </svg>
    <p class="muted note">Every position's vector at each of the {S} snapshots, projected onto the top two directions of all of
      them together; darker paths are later positions. The selected position is bold: it starts at the open circle (x₀) and
      ends at its token. {settings.trajNormalise
        ? "Each snapshot is centred and its vectors scaled to unit length, so later layers' larger vectors do not dominate."
        : "Raw vectors: the later layers' larger vectors dominate the picture."} Click a path to select its position.</p>
  {/if}
</section>

<style>
  section { margin-bottom: 16px; }
  .bar { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .opt { display: flex; gap: 4px; align-items: center; font-size: 12px; }
  svg { max-width: 720px; display: block; }
  .bg { fill: var(--panel); stroke: var(--line); }
  .path { fill: none; stroke-width: 1.2; opacity: 0.45; cursor: pointer; }
  .path:hover { opacity: 0.9; stroke-width: 2; }
  .path.sel { stroke-width: 2.4; opacity: 1; cursor: default; }
  .mark { stroke-width: 1.5; }
  .lbl { font: 11px var(--mono); fill: var(--ink); pointer-events: none; }
  .muted-lbl { fill: var(--muted); }
  .note { font-size: 11px; margin: 4px 0 0; }
</style>
