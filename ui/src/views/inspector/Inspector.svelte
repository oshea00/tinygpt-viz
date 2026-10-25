<script lang="ts">
  import { app, selection, settings } from "../../state/app.svelte";
  import { stageRange, type ScaleMode } from "../../lib/scale";
  import { dims2, fmt } from "../../lib/tensor";
  import Formula from "./Formula.svelte";
  import Heatmap from "./Heatmap.svelte";
  import Provenance from "./Provenance.svelte";

  const trace = $derived(app.trace);
  const stage = $derived(selection.stage(trace));
  const backward = $derived(selection.phase === "backward" && !!stage?.grad);
  const tensor = $derived(stage ? (backward ? stage.grad! : stage.output) : null);
  const range = $derived(trace && stage ? stageRange(trace, stage, backward, settings.scaleMode, settings.clip, settings.symlog) : null);
  const dims = $derived(tensor ? dims2(tensor) : [0, 0]);

  // Rows are positions labelled with their token. Columns are dimensions, except where
  // they are positions (attention) or vocabulary entries (logits, probs).
  const isTT = $derived(stage ? ["attn_scores", "mask", "softmax"].includes(stage.kind) : false);
  const isVocab = $derived(stage?.kind === "logits" || stage?.kind === "probs");
  const rowLabels = $derived(trace && tensor && tensor.shape.length >= 2 ? trace.text : undefined);
  const colLabels = $derived(isTT ? trace?.text : isVocab ? app.hello?.vocab : undefined);
  const colWord = $derived(isTT ? "key position" : isVocab ? "token" : "dim");

  const pickedR = $derived(selection.position >= 0 && selection.position < dims[0] ? selection.position : -1);
  const pickedC = $derived(selection.dim >= 0 && selection.dim < dims[1] ? selection.dim : -1);
</script>

{#if !trace || !stage || !tensor || !range}
  <p class="muted">Waiting for a trace…</p>
{:else}
  <div class="head">
    <div>
      <span class="id mono">{stage.id}</span>
      <span class="muted">· {stage.kind} · [{tensor.shape.join(" × ") || "scalar"}] · step {trace.step}</span>
    </div>
    <div class="phase">
      <button class:on={!backward} onclick={() => (selection.phase = "forward")}>value</button>
      <button class:on={backward} disabled={!stage.grad} onclick={() => (selection.phase = "backward")}>gradient ∂L/∂x</button>
    </div>
  </div>

  <div class="controls">
    <label>scale
      <select bind:value={settings.scaleMode}>
        {#each [["tensor", "per tensor"], ["stageType", "per stage type, all layers"], ["global", "global"]] as [v, l]}
          <option value={v as ScaleMode}>{l}</option>
        {/each}
      </select>
    </label>
    <label>clip at
      <select bind:value={settings.clip}>
        {#each [100, 99.9, 99, 95] as p}<option value={p}>{p === 100 ? "max" : `${p}th pct`}</option>{/each}
      </select>
    </label>
    <label><input type="checkbox" bind:checked={settings.symlog} disabled={range.kind === "sequential"} /> symmetric log</label>
    <label>zoom
      <select bind:value={settings.cell}>
        {#each [0, 2, 3, 4] as z}<option value={z}>{z === 0 ? "fit" : `${z}×`}</option>{/each}
      </select>
    </label>
    <span class="range mono">
      {range.kind === "diverging" ? `${fmt(range.lo, 3)} … 0 … ${fmt(range.hi, 3)}` : `0 … ${fmt(range.hi, 3)}`}
      <span class="muted">({range.label}{settings.clip < 100 ? `, clipped at ${settings.clip}th pct, marked •` : ""})</span>
    </span>
  </div>

  {#if stage.kind === "loss"}
    <p class="loss mono">{fmt(tensor.data[0], 5)}</p>
  {:else}
    <Heatmap
      {tensor} {range} {rowLabels} {colLabels}
      selRow={pickedR} selCol={pickedC} zoom={settings.cell || 1}
      maskUpper={stage.kind === "mask" || stage.kind === "softmax"}
      onpick={(r, c) => { selection.position = r; selection.dim = c; }}
    />
    <p class="axes muted">rows: positions (token) · columns: {colWord}</p>
  {/if}

  <div class="lesson">
    <Formula tex={stage.formula} />
    <p>{stage.explain}</p>
    {#if stage.weights.length}<p class="muted">weights: <span class="mono">{stage.weights.join(", ")}</span></p>{/if}
    {#if stage.inputs.length}
      <p class="muted">reads:
        {#each stage.inputs as id}<button class="link mono" onclick={() => selection.select(id)}>{id}</button>{/each}
      </p>
    {/if}
  </div>

  {#if stage.kind === "loss"}
    <Provenance {trace} weights={app.weights} {stage} r={0} c={0} {backward} />
  {:else if pickedR >= 0 && pickedC >= 0}
    <Provenance {trace} weights={app.weights} {stage} r={pickedR} c={pickedC} {backward} />
  {:else}
    <p class="muted">Click a cell to see how it was computed.</p>
  {/if}
{/if}

<style>
  .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
  .id { font-size: 15px; font-weight: 600; }
  .phase button.on { background: var(--accent-soft); border-color: var(--accent); }
  .controls { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin-bottom: 8px; font-size: 12px; }
  .range { font-size: 11px; }
  .axes { font-size: 11px; margin: 4px 0 0; }
  .loss { font-size: 28px; margin: 12px 0; }
  .lesson { margin-top: 10px; max-width: 760px; }
  .lesson p { margin: 4px 0; }
  .link { border: none; background: none; color: var(--accent); padding: 0 6px 0 0; text-decoration: underline; }
</style>
