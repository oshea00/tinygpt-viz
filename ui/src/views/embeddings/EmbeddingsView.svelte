<script lang="ts">
  // The model's two learned input tables as maps: the position table (one row per position)
  // and the token table (one row per vocabulary entry), each projected onto its top two
  // principal components. Nearby points are rows the model treats as alike.
  import { untrack } from "svelte";
  import type { Weights } from "../../types";
  import { app, selection } from "../../state/app.svelte";
  import { cssColor, HEAD_HUES } from "../../lib/colormap";
  import { nearestRows } from "../../lib/concepts";
  import { fitBox, tableMap, tokenKind, TOKEN_KINDS, type Map2D } from "../../lib/embeddings";
  import { dims2 } from "../../lib/tensor";

  const W = 460, H = 380, PAD = 18;
  const trace = $derived(app.trace!);
  const vocab = $derived(app.hello?.vocab ?? []);
  const T = $derived(trace.tokens.length);
  const pos = $derived(selection.query(trace));

  // PCA of the 4,096-row token table takes about a second, so the maps are recomputed when the
  // weights change, except while training runs (then on request).
  let shown = $state.raw<{ version: number; step: number; wpe: Map2D; wte: Map2D } | null>(null);
  function compute(w: Weights) {
    shown = { version: w.version, step: w.step, wpe: tableMap(w.tensors["wpe"]), wte: tableMap(w.tensors["wte"]) };
  }
  const stale = $derived(!!app.weights && !!shown && app.weights.version !== shown.version);
  $effect(() => {
    const w = app.weights;
    if (w && (!untrack(() => shown) || (!app.status?.playing && untrack(() => shown)!.version !== w.version))) compute(w);
  });

  const posPts = $derived(shown ? fitBox(shown.wpe.points, W, H, PAD) : []);
  const tokPts = $derived(shown ? fitBox(shown.wte.points, W, H, PAD) : []);
  const posPath = $derived(posPts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(""));

  const kindColor = new Map(TOKEN_KINDS.map((k, i) => [k, HEAD_HUES[i % HEAD_HUES.length]]));
  const kinds = $derived(vocab.map(tokenKind));
  const inPrompt = $derived(new Set(trace.tokens));
  let search = $state("");
  const matches = $derived(search.trim() ? new Set(vocab.flatMap((v, i) => (v.replace(/^␣/, "").includes(search.trim()) ? [i] : []))) : new Set<number>());
  let picked = $state(-1);
  const neighbours = $derived.by(() => {
    const t = app.weights?.tensors["wte"];
    if (picked < 0 || !t) return [];
    const [V, d] = dims2(t);
    return nearestRows(t.data, V, d, picked, 12);
  });
</script>

{#if !shown}
  <p class="muted">Waiting for the weights…</p>
{:else}
  <p class="muted status">
    weights at step {shown.step}
    {#if stale}· training has moved on <button class="link" onclick={() => app.weights && compute(app.weights)}>refresh the maps</button>{/if}
  </p>
  <div class="two">
    <section>
      <h2>Position table <span class="muted">PC1 {(shown.wpe.explained[0] * 100).toFixed(0)}% · PC2 {(shown.wpe.explained[1] * 100).toFixed(0)}% of variance</span></h2>
      <svg viewBox="0 0 {W} {H}" width="100%">
        <rect x="0" y="0" width={W} height={H} class="bg" />
        <path d={posPath} class="curve" />
        {#each posPts as [x, y], i (i)}
          <circle cx={x} cy={y} r={i < T ? 4 : 2.5} fill={cssColor("sequential", 0.25 + (0.75 * i) / (posPts.length - 1))}
                  class:outside={i >= T} class:sel={i === pos} role="button" tabindex="-1"
                  onclick={() => i < T && (selection.position = i)} onkeydown={() => {}}>
            <title>position {i}{i < T ? `: “${trace.text[i]}”` : " (beyond this prompt)"}</title>
          </circle>
          {#if i % 10 === 0}<text x={x + 6} y={y - 6} class="lbl">{i}</text>{/if}
        {/each}
      </svg>
      <p class="muted note">One point per row of <code>wpe</code>, joined in position order; darker is later. The
        prompt's {T} positions are larger, and the selected one is ringed. Neighbouring positions usually get similar
        vectors, so the line is smooth.</p>
    </section>

    <section>
      <h2>Token table <span class="muted">PC1 {(shown.wte.explained[0] * 100).toFixed(0)}% · PC2 {(shown.wte.explained[1] * 100).toFixed(0)}% of variance</span></h2>
      <div class="tools">
        <input type="search" bind:value={search} placeholder="highlight tokens containing…" spellcheck="false" />
        {#if search.trim()}<span class="muted">{matches.size} match{matches.size === 1 ? "" : "es"}</span>{/if}
      </div>
      <div class="legend">{#each TOKEN_KINDS as k}<span><i style:background={kindColor.get(k)}></i>{k}</span>{/each}</div>
      <svg viewBox="0 0 {W} {H}" width="100%">
        <rect x="0" y="0" width={W} height={H} class="bg" />
        {#each tokPts as [x, y], i (i)}
          {@const hi = inPrompt.has(i) || matches.has(i) || i === picked}
          <circle cx={x} cy={y} r={hi ? 3.5 : 1.6} fill={kindColor.get(kinds[i])} class:dim={!hi && (matches.size > 0 || picked >= 0)}
                  class:sel={i === picked} role="button" tabindex="-1" onclick={() => (picked = i === picked ? -1 : i)} onkeydown={() => {}}>
            <title>{vocab[i]} ({kinds[i]}, id {i})</title>
          </circle>
        {/each}
        {#each [...inPrompt] as i (i)}
          <text x={tokPts[i][0] + 5} y={tokPts[i][1] - 4} class="lbl">{vocab[i]}</text>
        {/each}
      </svg>
      <p class="muted note">One point per row of <code>wte</code> (also the output layer: the embeddings are tied). The prompt's
        tokens are labelled; click a point for its nearest rows. Tokens that never occur in training end up almost the
        same vector, so they collapse to one dot at the edge of the map.</p>
      {#if picked >= 0}
        <h2>Nearest to <span class="mono">{vocab[picked]}</span> <button class="link" onclick={() => (picked = -1)}>clear</button></h2>
        <ol class="nn mono">
          {#each neighbours as n (n.i)}
            <li><button class="link" onclick={() => (picked = n.i)}>{vocab[n.i]}</button> <span class="muted">{n.cos.toFixed(2)}</span></li>
          {/each}
        </ol>
      {/if}
    </section>
  </div>
{/if}

<style>
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  @media (max-width: 1400px) { .two { grid-template-columns: minmax(0, 1fr); } }
  svg { max-width: 560px; display: block; }
  .bg { fill: var(--panel); stroke: var(--line); }
  .curve { fill: none; stroke: var(--muted); stroke-width: 1; opacity: 0.6; }
  circle { cursor: pointer; stroke: var(--panel); stroke-width: 0.5; }
  circle.outside { opacity: 0.55; cursor: default; }
  circle.dim { opacity: 0.15; }
  circle.sel { stroke: var(--select); stroke-width: 2; }
  .lbl { font: 10px var(--mono); fill: var(--ink); pointer-events: none; }
  .tools { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; font-size: 12px; }
  .tools input { width: 220px; }
  .legend { display: flex; gap: 12px; font-size: 12px; margin-bottom: 6px; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .note { font-size: 11px; margin: 4px 0 0; }
  .status { font-size: 11px; margin: 0 0 8px; }
  .nn { columns: 3; font-size: 12px; margin: 4px 0 0; padding-left: 22px; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 12px; }
</style>
