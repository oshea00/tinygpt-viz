<script lang="ts">
  // What does one MLP neuron respond to? Its strongest activations over validation text.
  import { app, selection } from "../../state/app.svelte";
  import { concepts } from "../../concepts/store.svelte";

  let layerIdx = $state(0);
  let neuron = $state(0);
  const cfg = $derived(app.hello?.config);
  const fromInspector = $derived(/^L(\d+)\.mlp\.(gelu|up)$/.exec(selection.stageId));
  const res = $derived(concepts.neurons);
</script>

<div class="row">
  <label>layer <select bind:value={layerIdx}>{#each Array.from({ length: cfg?.n_layer ?? 0 }, (_, l) => l) as l}<option value={l}>{l}</option>{/each}</select></label>
  <label>neuron <input type="number" min="0" max={(cfg?.d_mlp ?? 1) - 1} bind:value={neuron} /></label>
  {#if fromInspector && selection.dim >= 0}
    <button class="link" onclick={() => { layerIdx = +fromInspector[1]; neuron = selection.dim; }}>use L{fromInspector[1]} neuron {selection.dim} from the inspector</button>
  {/if}
  <button class="primary" onclick={() => app.send({ type: "neuronTop", layer: layerIdx, neuron })}>find what it responds to</button>
</div>
{#if res}
  <h2>L{res.layer} neuron {res.neuron}: strongest activations in 512 validation windows</h2>
  {#each res.examples as ex}
    <div class="ex mono"><span class="v">{ex.value.toFixed(2)}</span><span class="ctx">{ex.before}<b>{ex.token}</b>{ex.after}</span></div>
  {/each}
  <p class="muted note">Bold: the token where it fires. Many neurons respond to several unrelated things (polysemantic); a
    sparse autoencoder, the stretch goal in the design, would look for cleaner features.</p>
{/if}

<style>
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  input[type="number"] { width: 70px; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 12px; }
  .ex { display: grid; grid-template-columns: 50px 1fr; gap: 8px; font-size: 12px; padding: 2px 0; border-bottom: 1px solid var(--line); }
  .v { text-align: right; color: var(--muted); }
  .ctx { white-space: pre-wrap; }
  .ctx b { background: #fde68a; color: #78350f; padding: 0 2px; border-radius: 2px; }
  .note { font-size: 11px; }
</style>
