<script lang="ts">
  import type { Stage, Trace, Weights } from "../../types";
  import { gradProvenance, provenance, type Provenance } from "../../lib/provenance";
  import { fmt } from "../../lib/tensor";
  import Strip from "./Strip.svelte";
  import { finetune } from "../../finetune/store.svelte";

  let { trace, weights, stage, r, c, backward }: {
    trace: Trace; weights: Weights | null; stage: Stage; r: number; c: number; backward: boolean;
  } = $props();

  // With an adapter on, its A and B are weights too.
  const all = $derived.by((): Weights | null => {
    const extra = trace.adapter ? finetune.weightsOf(trace.adapter.name) : undefined;
    return weights && extra ? { ...weights, tensors: { ...weights.tensors, ...extra } } : weights;
  });
  const result = $derived.by((): { p: Provenance | null; err: string } => {
    try {
      return { p: backward ? gradProvenance(trace, stage, r, c) : provenance(trace, all, stage, r, c), err: "" };
    } catch (e) {
      return { p: null, err: String((e as Error).message) };
    }
  });
  const matches = $derived(
    result.p ? (result.p.stored === -Infinity ? result.p.value === -Infinity : Math.abs(result.p.value - result.p.stored) <= 1e-4 * Math.max(1, Math.abs(result.p.stored))) : false,
  );
</script>

<div class="prov">
  <h2>Where did this number come from? <span class="mono">[{r}, {c}]</span></h2>
  {#if result.err}
    <p class="muted">{result.err}</p>
  {:else if result.p}
    {@const p = result.p}
    <p class="title">{p.title}</p>
    {#each p.strips as s}
      <Strip label={s.label} values={s.values} />
    {/each}
    <table class="mono">
      <tbody>
        {#each p.lines as [label, v]}
          <tr><td class="muted">{label}</td><td class="num">{typeof v === "number" ? fmt(v, 5) : v}</td></tr>
        {/each}
        <tr class="check">
          <td>stored in trace</td>
          <td class="num">{fmt(p.stored, 5)} <span class={matches ? "ok" : "bad"}>{matches ? "✓ matches" : "≠"}</span></td>
        </tr>
      </tbody>
    </table>
  {:else}
    <p class="muted">No breakdown for this stage.</p>
  {/if}
</div>

<style>
  .prov { margin-top: 12px; }
  .title { margin: 0 0 6px; }
  table { border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  td { padding: 1px 12px 1px 0; }
  .num { text-align: right; }
  .check td { border-top: 1px solid var(--line); padding-top: 3px; }
  .ok { color: var(--ok); }
  .bad { color: var(--bad); }
</style>
