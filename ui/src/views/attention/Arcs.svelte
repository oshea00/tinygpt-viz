<script lang="ts">
  // The prompt as characters; the selected position draws arcs back to the earlier
  // characters it attends to, thickness by weight, colour by head.
  import { app, selection, settings } from "../../state/app.svelte";
  import { headColor } from "../../lib/colormap";
  import { at } from "../../lib/tensor";

  const CW = 15;        // px per token, or per character of longer (BPE) tokens
  const BASE = 96;      // y of the text baseline; arcs rise above it
  const MIN_W = 0.02;   // weights below this are not drawn

  const trace = $derived(app.trace!);
  const T = $derived(trace.tokens.length);
  const q = $derived(selection.query(trace));
  const heads = $derived(
    settings.arcsAllHeads ? Array.from({ length: app.hello?.config.n_head ?? 0 }, (_, h) => h) : [selection.head],
  );
  const weightsOf = (h: number) => trace.byId.get(`L${selection.layer}.attn.H${h}.weights`)!.output;

  const arcs = $derived(
    heads.flatMap((h) => {
      const W = weightsOf(h);
      return Array.from({ length: q + 1 }, (_, j) => ({ h, j, w: at(W, q, j) })).filter((a) => a.w >= MIN_W);
    }),
  );
  // Tokens can be single characters or whole words: lay them out by length.
  const widths = $derived(trace.text.map((t) => Math.max(CW, t.length * 8 + 4)));
  const xs = $derived(widths.reduce<number[]>((acc, w, i) => [...acc, i ? acc[i - 1] + widths[i - 1] : 0], []));
  const total = $derived(widths.reduce((a, b) => a + b, 0));
  const cx = (i: number) => xs[i] + widths[i] / 2;
  function arcPath(from: number, to: number, h: number): string {
    // Self-attention is drawn as a small loop above the character.
    if (from === to) return `M${cx(from) - 4},${BASE - 14} a4,6 0 1,1 8,0`;
    const height = Math.min(BASE - 8, 12 + Math.abs(from - to) * 3.5) - h * 2;
    const mid = (cx(from) + cx(to)) / 2;
    return `M${cx(from)},${BASE - 14} Q${mid},${BASE - 14 - height * 2} ${cx(to)},${BASE - 14}`;
  }
  // Tint mode: one weight per character, from the selected head.
  const tint = $derived(Array.from({ length: T }, (_, j) => (j <= q ? at(weightsOf(selection.head), q, j) : 0)));
</script>

<div class="opts">
  <label><input type="checkbox" bind:checked={settings.arcsAllHeads} disabled={settings.arcsTint} /> all heads in layer {selection.layer}</label>
  <label><input type="checkbox" bind:checked={settings.arcsTint} /> tint characters instead (wraps long prompts)</label>
  <span class="muted">query position {q} “{trace.text[q]}” · click a character to move it</span>
</div>

{#if settings.arcsTint}
  <div class="tinted mono">
    {#each trace.text as t, j}
      <button
        class:q={j === q}
        style:background={j <= q ? `color-mix(in srgb, ${headColor(selection.head)} ${Math.round(tint[j] * 100)}%, transparent)` : undefined}
        title="weight {tint[j].toFixed(3)}"
        onclick={() => (selection.position = j)}
      >{t}</button>
    {/each}
  </div>
{:else}
  <div class="scroll">
    <svg width={total} height={BASE + 8}>
      {#each arcs as a (`${a.h}-${a.j}`)}
        <path d={arcPath(q, a.j, a.h)} stroke={headColor(a.h)} stroke-width={0.6 + a.w * 7}
              stroke-opacity={Math.min(1, 0.25 + a.w)} fill="none">
          <title>head {a.h}: “{trace.text[q]}” → “{trace.text[a.j]}” {a.w.toFixed(3)}</title>
        </path>
      {/each}
      {#each trace.text as t, i}
        <g class="ch" role="button" tabindex="-1" onclick={() => (selection.position = i)} onkeydown={() => {}}>
          <rect x={xs[i]} y={BASE - 13} width={widths[i]} height={20} class:q={i === q} class:future={i > q} />
          <text x={cx(i)} y={BASE + 2} text-anchor="middle" class:future={i > q}>{t}</text>
        </g>
      {/each}
    </svg>
  </div>
{/if}

<style>
  .opts { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; font-size: 12px; margin-bottom: 4px; }
  .opts label { display: flex; gap: 4px; align-items: center; }
  .scroll { overflow-x: auto; }
  svg { display: block; }
  .ch { cursor: pointer; }
  rect { fill: transparent; }
  rect.q { fill: none; stroke: var(--select); stroke-width: 1.5; }
  text { font: 13px var(--mono); fill: var(--ink); }
  text.future { opacity: 0.3; }
  .tinted { display: flex; flex-wrap: wrap; gap: 1px; }
  .tinted button { padding: 0 2px; min-width: 15px; border-radius: 2px; border-color: transparent; font-size: 13px; }
  .tinted button.q { outline: 1.5px solid var(--select); }
</style>
