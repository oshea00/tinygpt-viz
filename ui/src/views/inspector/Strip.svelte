<script lang="ts">
  // A vector as a row of signed bars: the "input row" and "weight column" of a dot product.
  import { cssColor } from "../../lib/colormap";
  import { fmt } from "../../lib/tensor";

  let { label, values, mark = -1 }: { label: string; values: number[]; mark?: number } = $props();
  const H = 34;
  const max = $derived(Math.max(1e-12, ...values.map((v) => (Number.isFinite(v) ? Math.abs(v) : 0))));
  const bw = $derived(Math.max(1.5, Math.min(8, 560 / values.length)));
</script>

<div class="strip">
  <div class="label muted">{label} <span class="mono">(|max| {fmt(max, 3)})</span></div>
  <svg width={values.length * bw} height={H}>
    <line x1="0" x2={values.length * bw} y1={H / 2} y2={H / 2} stroke="var(--line)" />
    {#each values as v, i}
      {@const t = Number.isFinite(v) ? v / max : 0}
      <rect x={i * bw} width={Math.max(1, bw - 0.5)} y={t >= 0 ? H / 2 - (t * H) / 2 : H / 2} height={Math.abs(t * H) / 2}
            fill={cssColor("diverging", t)} stroke={i === mark ? "var(--select)" : "none"}>
        <title>[{i}] {fmt(v, 5)}</title>
      </rect>
    {/each}
  </svg>
</div>

<style>
  .strip { margin: 6px 0; overflow-x: auto; }
  .label { font-size: 11px; margin-bottom: 2px; }
  svg { display: block; }
</style>
