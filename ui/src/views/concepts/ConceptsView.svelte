<script lang="ts">
  // Find conceptual associations: run a prompt set, capture each target word's representation
  // at every snapshot, compare, then test findings by intervening.
  import { app } from "../../state/app.svelte";
  import { concepts, type Panel } from "../../concepts/store.svelte";
  import Attribution from "./Attribution.svelte";
  import Drift from "./Drift.svelte";
  import MapPanel from "./MapPanel.svelte";
  import Neighbours from "./Neighbours.svelte";
  import Neurons from "./Neurons.svelte";
  import Probes from "./Probes.svelte";
  import Similarity from "./Similarity.svelte";
  import Steer from "./Steer.svelte";

  let editing = $state(false);
  const c = $derived(concepts.current);
  const s = $derived(concepts.snapIndex(c));
  const panels: [Panel, string][] = [
    ["similarity", "similarity"], ["map", "projection map"], ["neighbours", "neighbours"], ["drift", "context drift"],
    ["probes", "probes"], ["steer", "directions & steering"], ["attribution", "attribution"], ["neurons", "neurons"],
  ];
  const needsCapture = $derived(!["attribution", "neurons"].includes(concepts.panel));
</script>

<div class="setup">
  <div class="row">
    <b>Prompt set</b>
    <span class="muted">{concepts.parsed.prompts.length} prompts · {Object.keys(concepts.parsed.categories).length} categories</span>
    <button class="link" onclick={() => (editing = !editing)}>{editing ? "hide" : "edit"}</button>
    <label>pool <select bind:value={concepts.pool}><option value="last">last sub-token</option><option value="mean">mean of sub-tokens</option></select></label>
    <label><input type="checkbox" bind:checked={concepts.withBaseline} /> also the untrained model (step 0)</label>
    <button class="primary" disabled={concepts.pending > 0 || !concepts.parsed.prompts.length} onclick={() => concepts.capture()}>
      {concepts.pending ? "capturing…" : "capture"}
    </button>
  </div>
  {#if editing}
    <textarea rows="10" bind:value={concepts.source} spellcheck="false"></textarea>
    <p class="muted note">Lines <code>@category: word word …</code> define categories; each other line is a template with one
      <code>{"{category}"}</code> slot, filled with every word of that category. The filled-in word is what gets captured.</p>
  {/if}
  {#each concepts.parsed.errors as e}<p class="bad note">{e}</p>{/each}
  {#if c}
    <div class="row">
      <span class="muted">captured {c.N} words from step {c.step}{concepts.untrained ? " (and the untrained baseline)" : ""} · snapshot</span>
      <input type="range" min="0" max={c.S - 1} value={s} oninput={(e) => (concepts.snap = +(e.target as HTMLInputElement).value)} />
      <b class="mono">{c.snapshots[s]}</b>
    </div>
  {/if}
</div>

<nav class="panels">
  {#each panels as [p, label]}
    <button class:on={concepts.panel === p} onclick={() => (concepts.panel = p)}>{label}</button>
  {/each}
</nav>

{#if needsCapture && !c}
  <div class="start">
    <p>Press <b>capture</b> (top right) to run the prompt set through the model. It records the vector of each
      filled-in word (the animal, colour, emotion…) at every layer; this panel then shows what those vectors have in common.</p>
    <p class="muted">Keep “also the untrained model” ticked to compare against random weights: structure that also appears
      there was not learned. Attribution and neurons work without capturing.</p>
    {#if app.hello?.config.tokenizer !== "bpe"}
      <p class="muted">This model reads characters, so each word spans several tokens: choose “mean of sub-tokens”, or
        restart the server with <code>--config concepts</code> for a model with whole-word tokens.</p>
    {/if}
  </div>
{:else if concepts.panel === "similarity"}<Similarity />
{:else if concepts.panel === "map"}<MapPanel />
{:else if concepts.panel === "neighbours"}<Neighbours />
{:else if concepts.panel === "drift"}<Drift />
{:else if concepts.panel === "probes"}<Probes />
{:else if concepts.panel === "steer"}<Steer />
{:else if concepts.panel === "attribution"}<Attribution />
{:else}<Neurons />{/if}

<style>
  .setup { margin-bottom: 8px; }
  .start { max-width: 720px; }
  .start p { margin: 0 0 6px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 6px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  textarea { width: 100%; font-family: var(--mono); font-size: 12px; }
  .note { font-size: 11px; margin: 2px 0 6px; }
  .bad { color: var(--bad); }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 12px; }
  .panels { display: flex; gap: 4px; flex-wrap: wrap; margin: 4px 0 12px; }
  .panels button.on { background: var(--accent-soft); border-color: var(--accent); }
  code { font-family: var(--mono); background: var(--accent-soft); padding: 0 3px; border-radius: 3px; }
</style>
