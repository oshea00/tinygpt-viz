<script lang="ts">
  // Do categories form clusters, and from which snapshot? The top two principal components
  // of the centred vectors, one dot per captured occurrence. Click a dot to open its trace.
  import { app, selection, settings } from "../../state/app.svelte";
  import { concepts } from "../../concepts/store.svelte";
  import { layer, pca, project, type Capture } from "../../lib/concepts";
  import { categoryColors } from "./colors";

  const W = 380, H = 300, PAD = 14;
  function map(c: Capture | null) {
    if (!c || c.N < 3) return null;
    const rows = layer(c, concepts.snapIndex(c));
    const { components, variance, total } = pca(rows, 2);
    const xy = project(rows, components);
    const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    return {
      c, explained: variance.map((v) => v / total),
      points: xy.map(([x, y], i) => ({ i, x: PAD + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * PAD), y: PAD + ((y - y0) / (y1 - y0 || 1)) * (H - 2 * PAD) })),
    };
  }
  const maps = $derived([["trained model", map(concepts.current)], ["untrained (step 0)", map(concepts.untrained)]] as const);
  const colors = $derived(categoryColors(concepts.current?.items.map((i) => i.category) ?? []));

  function open(c: Capture, i: number) {
    const it = c.items[i];
    app.prompt = it.prompt;
    app.promptEdited();
    app.run();
    selection.select(c.snapshots[concepts.snapIndex(c)], "forward");
    selection.position = it.position;
    settings.tab = "inspector";
  }
</script>

<div class="legend">{#each [...colors] as [cat, col]}<span><i style:background={col}></i>{cat}</span>{/each}</div>
<div class="two">
  {#each maps as [title, m]}
    {#if m}
      <section>
        <h2>{title} <span class="muted">PC1 {(m.explained[0] * 100).toFixed(0)}% · PC2 {(m.explained[1] * 100).toFixed(0)}% of variance</span></h2>
        <svg viewBox="0 0 {W} {H}" width="100%">
          <rect x="0" y="0" width={W} height={H} class="bg" />
          {#each m.points as p (p.i)}
            {@const it = m.c.items[p.i]}
            <circle cx={p.x} cy={p.y} r="4.5" fill={colors.get(it.category)} role="button" tabindex="-1"
                    onclick={() => open(m.c, p.i)} onkeydown={() => {}}>
              <title>{it.word} ({it.category}): {it.prompt}</title>
            </circle>
          {/each}
        </svg>
      </section>
    {/if}
  {/each}
</div>
<p class="muted note">PCA is linear and fast; clusters it shows are real directions in the stream. Hover a dot for its prompt,
  click to open that prompt's trace at this snapshot.</p>

<style>
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  .bg { fill: var(--panel); stroke: var(--line); }
  circle { cursor: pointer; stroke: var(--panel); stroke-width: 1; }
  .legend { display: flex; gap: 12px; font-size: 12px; margin-bottom: 6px; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .note { font-size: 11px; }
</style>
