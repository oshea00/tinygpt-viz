<script lang="ts">
  import { app, selection, settings } from "../../state/app.svelte";
  import { stageRange, type ScaleMode } from "../../lib/scale";
  import { dims2, fmt } from "../../lib/tensor";
  import { headColor } from "../../lib/colormap";
  import { comparable, difference } from "../../lib/interventions";
  import { tensorRange } from "../../lib/scale";
  import Formula from "./Formula.svelte";
  import Heatmap from "./Heatmap.svelte";
  import Provenance from "./Provenance.svelte";

  const trace = $derived(app.trace);
  const stage = $derived(selection.stage(trace));
  const backward = $derived(selection.phase === "backward" && !!stage?.grad);
  // With interventions active, the stage can be shown as its change from the original run.
  const base = $derived(app.interventions.length && comparable(trace, app.baseline) ? app.baseline : null);
  const diff = $derived(base && stage && settings.compare && !backward ? difference(stage.output, base.byId.get(stage.id)?.output) : null);
  const tensor = $derived(stage ? (diff ?? (backward ? stage.grad! : stage.output)) : null);
  const range = $derived(
    trace && stage && tensor
      ? diff ? tensorRange([diff.data], "diverging", settings.clip, settings.symlog, "change vs original")
             : stageRange(trace, stage, backward, settings.scaleMode, settings.clip, settings.symlog)
      : null,
  );
  const dims = $derived(tensor ? dims2(tensor) : [0, 0]);

  // Rows are positions labelled with their token. Columns are dimensions, except where
  // they are positions (attention) or vocabulary entries (logits, probs).
  const isTT = $derived(stage ? ["attn_scores", "mask", "softmax"].includes(stage.kind) : false);
  const isVocab = $derived(stage?.kind === "logits" || stage?.kind === "probs");
  const rowLabels = $derived(trace && tensor && tensor.shape.length >= 2 ? trace.text : undefined);
  const colLabels = $derived(isTT ? trace?.text : isVocab ? app.hello?.vocab : undefined);
  const colWord = $derived(isTT ? "key position" : isVocab ? "token" : "dim");

  // Q, K, V and the concatenated head outputs are cut into one column band per head.
  const colBands = $derived.by(() => {
    if (!stage || !app.hello || !/^L\d+\.([qkv]|attn\.concat)$/.test(stage.id)) return undefined;
    const { n_head, d_model } = app.hello.config;
    const dh = d_model / n_head;
    return Array.from({ length: n_head }, (_, h) => ({ start: h * dh, end: (h + 1) * dh, color: headColor(h) }));
  });

  // Backward phase: the stage's backward rule, checked against a hand-written backward pass.
  $effect(() => {
    if (backward && trace) app.requestManual();
  });
  const check = $derived(
    backward && stage && app.manual && trace && app.manual.prompt === trace.prompt && app.manual.step === trace.step
      ? app.manual.stages[stage.id] : null,
  );

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
      <button class:on={!backward && !diff} onclick={() => { selection.phase = "forward"; settings.compare = false; }}>value</button>
      <button class:on={backward} disabled={!stage.grad} onclick={() => { selection.phase = "backward"; settings.compare = false; }}>gradient ∂L/∂x</button>
      {#if base}
        <button class:on={!!diff} class="change" onclick={() => { selection.phase = "forward"; settings.compare = true; }}>change from intervention</button>
      {/if}
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
      {colBands}
      onpick={(r, c) => { selection.position = r; selection.dim = c; }}
    />
    <p class="axes muted">rows: positions (token) · columns: {colWord}</p>
  {/if}

  <div class="lesson">
    {#if backward && stage.backward}
      <div class="muted small">backward rule</div>
      <Formula tex={stage.backward} />
      {#if app.interventions.length}
        <p class="muted small">Hand-written check unavailable while interventions are active.</p>
      {:else if check}
        <p class="check small">
          Hand-written backward (<span class="mono">core/manual_backward.py</span>) vs autograd for this stage:
          max |Δ| <span class="mono">{fmt(check.maxAbsDiff, 2)}</span>
          {#if check.nearZero}
            · <b>zero in exact arithmetic</b>; both sides are rounding noise
          {:else}
            · relative <span class="mono">{fmt(check.relative, 2)}</span>
          {/if}
          <span class={check.relative < 1e-4 ? "ok" : "bad"}>{check.relative < 1e-4 ? "✓ agree" : "✗ differ"}</span>
          {#if app.manual}<span class="muted">· worst over all {Object.keys(app.manual.stages).length} stages and {Object.keys(app.manual.weights).length} weights: {fmt(app.manual.worstRelative, 2)}</span>{/if}
        </p>
      {/if}
      <div class="muted small">forward</div>
    {/if}
    <Formula tex={stage.formula} />
    <p>{stage.explain}</p>
    {#if stage.weights.length}<p class="muted">weights: <span class="mono">{stage.weights.join(", ")}</span></p>{/if}
    {#if stage.inputs.length}
      <p class="muted">reads:
        {#each stage.inputs as id}<button class="link mono" onclick={() => selection.select(id)}>{id}</button>{/each}
      </p>
    {/if}
  </div>

  {#if diff}
    <p class="muted">Showing (with interventions) − (original). Values outside the stages an intervention can reach stay at 0.</p>
  {:else if stage.kind === "loss"}
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
  .phase .change.on { background: #fde68a; border-color: #d97706; color: #78350f; }
  .controls { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin-bottom: 8px; font-size: 12px; }
  .range { font-size: 11px; }
  .axes { font-size: 11px; margin: 4px 0 0; }
  .loss { font-size: 28px; margin: 12px 0; }
  .lesson { margin-top: 10px; max-width: 760px; }
  .small { font-size: 11px; }
  .check { background: var(--accent-soft); padding: 4px 8px; border-radius: 6px; }
  .ok { color: var(--ok); font-weight: 600; }
  .bad { color: var(--bad); font-weight: 600; }
  .lesson p { margin: 4px 0; }
  .link { border: none; background: none; color: var(--accent); padding: 0 6px 0 0; text-decoration: underline; }
</style>
