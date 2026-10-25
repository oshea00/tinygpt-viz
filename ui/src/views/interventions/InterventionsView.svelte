<script lang="ts">
  // Change one thing inside the network and watch what follows: every intervention is
  // shown against a baseline run of the same prompt without it.
  import type { Intervention } from "../../types";
  import { app, selection, settings } from "../../state/app.svelte";
  import { comparable, describe, stageChanges, targetProbs } from "../../lib/interventions";
  import { at, dims2, fmt, row } from "../../lib/tensor";

  const cfg = $derived(app.hello!.config);
  const trace = $derived(app.trace!);
  const range = (n: number) => Array.from({ length: n }, (_, i) => i);
  const T = $derived(trace.tokens.length);

  // Form state for adding interventions.
  let headL = $state(0), headH = $state(0), headMode = $state<"zero" | "mean">("zero");
  let nL = $state(0), nIndex = $state(0), nMode = $state<"zero" | "mean">("zero");
  let bL = $state(0), bPart = $state<"attn" | "mlp">("attn");
  let aL = $state(0), aH = $state(0), aPos = $state(0);
  let setValue = $state(0);
  let seed = $state(0);

  const add = (iv: Intervention) => app.addIntervention(iv);
  const remove = (i: number) => app.setInterventions(app.interventions.filter((_, j) => j !== i));

  // The cell selected in the inspector, for "set value".
  const selStage = $derived(trace.byId.get(selection.stageId));
  const cellOk = $derived(!!selStage && selStage.output.shape.length === 2 && selection.position >= 0 && selection.position < dims2(selStage.output)[0] && selection.dim >= 0 && selection.dim < dims2(selStage.output)[1]);
  const cellValue = $derived(cellOk ? at(selStage!.output, selection.position, selection.dim) : 0);
  const gelu = $derived(/^L(\d+)\.mlp\.gelu$/.exec(selection.stageId));

  const base = $derived(comparable(trace, app.baseline) ? app.baseline : null);
  const changes = $derived(base ? stageChanges(trace, base).filter((c) => c.change > 0) : []);
  const probs = $derived(base ? targetProbs(trace, base) : []);
  const lossDelta = $derived(base && base.loss !== null && trace.loss !== null ? trace.loss - base.loss : null);

  // Next-token distribution at the selected (or last) position, before and after.
  const pos = $derived(selection.query(trace));
  const nextTok = $derived.by(() => {
    if (!base) return [];
    const a = row(trace.byId.get("probs")!.output, pos), b = row(base.byId.get("probs")!.output, pos);
    const ids = new Set([...[...a.keys()].sort((x, y) => a[y] - a[x]).slice(0, 6), ...[...b.keys()].sort((x, y) => b[y] - b[x]).slice(0, 6)]);
    return [...ids].map((i) => ({ i, before: b[i], after: a[i] })).sort((x, y) => Math.max(y.before, y.after) - Math.max(x.before, x.after));
  });

  // Relative change on a log scale: 1e-4 … 10.
  const logW = (r: number) => Math.max(0, Math.min(1, (Math.log10(Math.max(r, 1e-6)) + 4) / 5));

  function openStage(id: string) {
    selection.select(id, "forward");
    settings.compare = true;
    settings.tab = "inspector";
  }
</script>

<div class="active">
  <h2>Active interventions</h2>
  {#if app.interventions.length === 0}
    <p class="muted">None. Add one below; every view then shows the changed model, and this tab compares it with the original.</p>
  {:else}
    <ul>
      {#each app.interventions as iv, i}
        <li><span>{describe(iv)}</span> <button class="x" title="remove" onclick={() => remove(i)}>×</button></li>
      {/each}
    </ul>
    <button onclick={() => app.setInterventions([])}>clear all</button>
  {/if}
</div>

<div class="forms">
  <fieldset>
    <legend>Ablate a head</legend>
    <label>layer <select bind:value={headL}>{#each range(cfg.n_layer) as l}<option value={l}>{l}</option>{/each}</select></label>
    <label>head <select bind:value={headH}>{#each range(cfg.n_head) as h}<option value={h}>{h}</option>{/each}</select></label>
    <label><select bind:value={headMode}><option value="zero">zero</option><option value="mean">mean</option></select></label>
    <button onclick={() => add({ kind: "head", layer: headL, head: headH, mode: headMode })}>add</button>
  </fieldset>

  <fieldset>
    <legend>Ablate an MLP neuron</legend>
    <label>layer <select bind:value={nL}>{#each range(cfg.n_layer) as l}<option value={l}>{l}</option>{/each}</select></label>
    <label>neuron <input type="number" min="0" max={cfg.d_mlp - 1} bind:value={nIndex} /></label>
    <label><select bind:value={nMode}><option value="zero">zero</option><option value="mean">mean</option></select></label>
    <button onclick={() => add({ kind: "neuron", layer: nL, index: Math.max(0, Math.min(cfg.d_mlp - 1, nIndex)), mode: nMode })}>add</button>
    {#if gelu && selection.dim >= 0}
      <button class="link" onclick={() => { nL = +gelu[1]; nIndex = selection.dim; }}>use L{gelu[1]} neuron {selection.dim} from the inspector</button>
    {/if}
  </fieldset>

  <fieldset>
    <legend>Switch off a sublayer</legend>
    <label>layer <select bind:value={bL}>{#each range(cfg.n_layer) as l}<option value={l}>{l}</option>{/each}</select></label>
    <label><select bind:value={bPart}><option value="attn">attention</option><option value="mlp">MLP</option></select></label>
    <button onclick={() => add({ kind: "block", layer: bL, part: bPart })}>add</button>
  </fieldset>

  <fieldset>
    <legend>Force attention</legend>
    <label>layer <select bind:value={aL}>{#each range(cfg.n_layer) as l}<option value={l}>{l}</option>{/each}</select></label>
    <label>head <select bind:value={aH}>{#each range(cfg.n_head) as h}<option value={h}>{h}</option>{/each}</select></label>
    <label>to position <input type="number" min="0" max={T - 1} bind:value={aPos} /></label>
    <button onclick={() => add({ kind: "attend", layer: aL, head: aH, position: aPos })}>add</button>
    <span class="muted hint">“{trace.text[Math.max(0, Math.min(T - 1, aPos))]}”; earlier positions attend to themselves</span>
  </fieldset>

  <fieldset>
    <legend>Edit a value</legend>
    {#if cellOk}
      <span class="mono">{selection.stageId}[{selection.position}, {selection.dim}]</span>
      <span class="muted">now {fmt(cellValue, 4)}</span>
      <label>set to <input type="number" step="0.1" bind:value={setValue} /></label>
      <button onclick={() => add({ kind: "set", stage: selection.stageId, row: selection.position, col: selection.dim, value: setValue })}>add</button>
    {:else}
      <span class="muted">Click a cell in the inspector first.</span>
    {/if}
  </fieldset>

  <fieldset>
    <legend>Shuffle position embeddings</legend>
    <label>seed <input type="number" bind:value={seed} /></label>
    <button onclick={() => add({ kind: "shuffle_pos", seed })}>add</button>
    <span class="muted hint">the model loses track of order</span>
  </fieldset>
</div>

{#if app.interventions.length && !base}
  <p class="muted">Waiting for the baseline run…</p>
{:else if base}
  <h2>Effect</h2>
  <div class="summary">
    {#if lossDelta !== null}
      <div><div class="muted">loss on this prompt</div>
        <b class="mono">{fmt(base.loss!, 4)} → {fmt(trace.loss!, 4)}</b>
        <span class={lossDelta > 0 ? "bad" : "ok"}>{lossDelta > 0 ? "+" : ""}{fmt(lossDelta, 4)}</span></div>
    {/if}
    <div><div class="muted">first stage changed</div><b class="mono">{changes[0]?.id ?? "none"}</b></div>
    <div><div class="muted">stages changed</div><b class="mono">{changes.length} / {trace.stages.length}</b></div>
  </div>

  <div class="two">
    <section>
      <h2>Next token after position {pos} “{trace.text[pos]}”: before ▪ after ▪</h2>
      {#each nextTok as t}
        <div class="pair">
          <span class="mono tok">{app.hello?.vocab[t.i]}</span>
          <span class="bars2">
            <span class="b before" style:width="{t.before * 100}%"></span>
            <span class="b after" style:width="{t.after * 100}%"></span>
          </span>
          <span class="mono num">{(t.before * 100).toFixed(1)} → {(t.after * 100).toFixed(1)}%</span>
        </div>
      {/each}
    </section>

    <section>
      <h2>Where the change goes <span class="muted">(‖Δ‖ / ‖before‖, log scale)</span></h2>
      <div class="changes">
        {#each changes as c}
          <button class="crow" onclick={() => openStage(c.id)} title="open in the inspector, showing the change">
            <span class="mono cid">{c.id}</span>
            <span class="ctrack"><span class="cfill" style:width="{logW(c.relative) * 100}%"></span></span>
            <span class="mono num">{c.relative < 1e-3 ? c.relative.toExponential(1) : (c.relative * 100).toFixed(1) + "%"}</span>
          </button>
        {/each}
      </div>
    </section>
  </div>

  {#if probs.length}
    <h2>p(actual next token) at each position: before ▪ after ▪</h2>
    <div class="pchart">
      {#each probs as p, i}
        <div class="col" title="position {i} “{trace.text[i]}”: {(p.before * 100).toFixed(1)}% → {(p.after * 100).toFixed(1)}%">
          <div class="cbars">
            <span class="b before" style:height="{p.before * 100}%"></span>
            <span class="b after" class:down={p.after < p.before - 1e-4} style:height="{p.after * 100}%"></span>
          </div>
          <span class="mono lab" class:word={trace.text[i].length > 1}>{trace.text[i]}</span>
        </div>
      {/each}
    </div>
  {/if}
{/if}

<style>
  .active ul { margin: 0 0 6px; padding-left: 18px; }
  .active li { margin: 2px 0; }
  .x { padding: 0 6px; margin-left: 6px; }
  .forms { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 8px; margin: 12px 0 16px; }
  fieldset { border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; }
  legend { font-weight: 600; padding: 0 4px; }
  label { display: flex; gap: 4px; align-items: center; }
  input[type="number"] { width: 68px; }
  .hint { font-size: 11px; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 11px; }
  .summary { display: flex; gap: 28px; flex-wrap: wrap; margin: 4px 0 12px; }
  .summary b { font-size: 14px; }
  .ok { color: var(--ok); }
  .bad { color: var(--bad); }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-bottom: 12px; }
  @media (max-width: 1300px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .pair { display: grid; grid-template-columns: 20px 1fr 110px; gap: 6px; align-items: center; font-size: 12px; margin: 2px 0; }
  .tok { text-align: center; }
  .bars2 { display: grid; gap: 1px; }
  .bars2 .b { display: block; height: 6px; border-radius: 2px; min-width: 1px; }
  .before { background: #a5b4fc; }
  .after { background: var(--ok); }
  .after.down { background: var(--bad); }
  .num { text-align: right; font-size: 11px; }
  .changes { max-height: 300px; overflow-y: auto; }
  .crow { display: grid; grid-template-columns: 150px 1fr 60px; gap: 6px; align-items: center; width: 100%; border: none; background: none; padding: 1px 0; font-size: 11px; text-align: left; }
  .crow:hover { background: var(--accent-soft); }
  .cid { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ctrack { height: 7px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .cfill { display: block; height: 100%; background: #d97706; }
  .pchart { display: flex; gap: 1px; align-items: flex-end; overflow-x: auto; padding-bottom: 4px; }
  .col { display: flex; flex-direction: column; align-items: center; width: 11px; flex: none; }
  .cbars { height: 80px; width: 100%; display: flex; align-items: flex-end; gap: 1px; }
  .cbars .b { flex: 1; display: block; min-height: 1px; }
  .lab { font-size: 9px; }
  /* Multi-character (BPE) tokens would overlap at this column width: run them downwards. */
  .lab.word { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; line-height: 11px; max-height: 70px; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
</style>
