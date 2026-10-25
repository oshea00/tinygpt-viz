<script lang="ts">
  // Does a word's representation shift with context? Mean pairwise cosine between its
  // occurrences, at every snapshot: 1 means context changes nothing.
  import { concepts } from "../../concepts/store.svelte";
  import { drift } from "../../lib/concepts";

  const counts = $derived.by(() => {
    const m = new Map<string, number>();
    for (const it of concepts.current?.items ?? []) m.set(it.word, (m.get(it.word) ?? 0) + 1);
    return m;
  });
  const words = $derived([...counts].filter(([, n]) => n > 1).map(([w]) => w).sort());
  const lines = $derived([
    ["trained", concepts.current ? drift(concepts.current, concepts.word) : null, "var(--accent)"],
    ["untrained", concepts.untrained ? drift(concepts.untrained, concepts.word) : null, "#d97706"],
  ] as const);
  const S = $derived(concepts.current?.S ?? 1);
  const W = 520, H = 200, P = 30;
  const x = (s: number) => P + (s / Math.max(1, S - 1)) * (W - 2 * P);
  const y = (v: number) => H - P - ((v + 1) / 2) * (H - 2 * P);
</script>

<label class="pick">word <select bind:value={concepts.word}>{#each words as w}<option value={w}>{w} ({counts.get(w)} contexts)</option>{/each}</select></label>
<svg viewBox="0 0 {W} {H}" width="100%" class="chart">
  {#each [-1, 0, 1] as v}<line x1={P} x2={W - P} y1={y(v)} y2={y(v)} class="grid" /><text x={P - 4} y={y(v) + 3} text-anchor="end">{v}</text>{/each}
  {#each lines as [name, d, col]}
    {#if d}
      <polyline points={d.map((p, s) => `${x(s)},${y(p.similarity)}`).join(" ")} fill="none" stroke={col} stroke-width="2" />
      {#each d as p, s}<circle cx={x(s)} cy={y(p.similarity)} r="3" fill={col}><title>{name}, {p.layer}: {p.similarity.toFixed(3)}</title></circle>{/each}
    {/if}
  {/each}
  {#each concepts.current?.snapshots ?? [] as name, s}
    {#if s % 2 === 0}<text x={x(s)} y={H - 10} text-anchor="middle" class="lbl">{name.replace(".resid_", " ").replace("embed.sum", "x₀")}</text>{/if}
  {/each}
</svg>
<p class="muted note"><span class="k tr">trained</span> <span class="k un">untrained</span> · mean cosine between the word's centred
  vectors in different prompts. At x₀ every occurrence is the same token plus a position, so similarity starts high;
  how far it falls is how much the layers mix context into the word.</p>

<style>
  .pick { font-size: 12px; display: flex; gap: 6px; align-items: center; margin-bottom: 8px; }
  .chart { max-width: 640px; }
  .grid { stroke: var(--line); }
  text { font: 9px var(--mono); fill: var(--muted); }
  .note { font-size: 11px; }
  .k { padding: 0 4px; border-radius: 3px; color: #fff; }
  .tr { background: var(--accent); }
  .un { background: #d97706; }
</style>
