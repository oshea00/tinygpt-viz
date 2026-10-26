<script lang="ts">
  // One head's T×T attention pattern: rows are queries, columns the keys they look at.
  import { app, selection } from "../../state/app.svelte";
  import { tensorRange } from "../../lib/scale";
  import { fmt } from "../../lib/tensor";
  import Heatmap from "../inspector/Heatmap.svelte";

  const trace = $derived(app.trace!);
  const stage = $derived(trace.byId.get(`L${selection.layer}.attn.H${selection.head}.weights`));
  const backward = $derived(selection.phase === "backward" && !!stage?.grad);
  const tensor = $derived(stage ? (backward ? stage.grad! : stage.output) : null);
  // Weights are probabilities, drawn on a fixed 0…1 scale so heads can be compared by eye.
  const range = $derived(
    tensor ? (backward ? tensorRange([tensor.data], "diverging", 99, false, "this tensor")
                       : { kind: "sequential" as const, lo: 0, hi: 1, symlog: false, label: "fixed 0…1" }) : null,
  );
  const q = $derived(selection.query(trace));
</script>

{#if tensor && range}
  <div class="bar">
    <span class="muted">rows: query position · columns: key position · {backward ? "gradient ∂L/∂A" : "weight"}
      <span class="mono">({range.kind === "sequential" ? "0 … 1" : `±${fmt(range.hi, 3)}`})</span></span>
    <span>
      <button class:on={!backward} onclick={() => (selection.phase = "forward")}>weights</button>
      <button class:on={backward} disabled={!stage?.grad} onclick={() => (selection.phase = "backward")}>gradient</button>
    </span>
  </div>
  <Heatmap
    {tensor} {range} rowLabels={trace.text} colLabels={trace.text}
    selRow={q} selCol={selection.dim < trace.tokens.length ? selection.dim : -1}
    maskUpper maxHeight={480} maxCell={14} reusedRows={backward ? 0 : trace.incremental?.reused ?? 0}
    onpick={(r, c) => { selection.position = r; selection.dim = c; }}
  />
{/if}

<style>
  .bar { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; flex-wrap: wrap; }
  button.on { background: var(--accent-soft); border-color: var(--accent); }
</style>
