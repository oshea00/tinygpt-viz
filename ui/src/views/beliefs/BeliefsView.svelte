<script lang="ts">
  // Belief trajectories, after Bigelow et al., "Stories in Space" (arXiv 2605.12412): read stories
  // one sentence at a time, record what the model believes after each (a cloze readout over a word
  // set) and the residual stream holding it, then probe, compare geometry and steer.
  import { beliefs, type BeliefSection } from "../../beliefs/store.svelte";
  import { isTestStory } from "../../lib/beliefs";
  import Geometry from "./Geometry.svelte";
  import Probes from "./Probes.svelte";
  import Steering from "./Steering.svelte";
  import Trajectory from "./Trajectory.svelte";

  let own = $state(false);
  const b = $derived(beliefs.current);
  const s = $derived(beliefs.snapIndex(b));
  const medianMass = $derived.by(() => {
    if (!b?.N) return NaN;
    const m = Array.from(b.mass).sort((x, y) => x - y);
    return m[Math.floor(m.length / 2)];
  });
  const held = $derived(b ? b.stories.filter((_, i) => isTestStory(i)).length : 0);
  const sections: [BeliefSection, string][] = [["trajectory", "trajectories"], ["probes", "probes"], ["geometry", "geometry"], ["steering", "steering & entanglement"]];
</script>

<div class="row">
  <label>cloze <input class="mono cloze" bind:value={beliefs.cloze} /></label>
  <label>words <input class="mono words" bind:value={beliefs.words} /></label>
  {#if !own}<label>stories <input type="number" min="5" max="400" step="5" bind:value={beliefs.n} /></label>{/if}
  <label><input type="checkbox" bind:checked={own} /> my own stories</label>
  <label><input type="checkbox" bind:checked={beliefs.withBaseline} /> also the untrained model</label>
  <button class="primary" disabled={beliefs.pending > 0} onclick={() => { if (!own) beliefs.custom = ""; beliefs.run(); }}>
    {beliefs.pending ? "reading…" : "read stories"}
  </button>
</div>
{#if own}
  <textarea rows="6" bind:value={beliefs.custom} spellcheck="false" placeholder="One story per paragraph; separate stories with a blank line."></textarea>
{/if}
<p class="muted note">After every sentence, the story so far is followed by the cloze, and the next-token probabilities of the
  words (each must be one token), renormalised over them, are the model's <b>belief</b>. The vectors are read at the sentence's
  last token, before the cloze, so they are the story's own. Without your own stories, the first N validation stories are used:
  text the model never trained on. Every fifth story is held out from the probes.</p>

{#if b}
  <div class="row">
    <span class="muted">{b.N} sentences from {b.stories.length} stories ({held} held out) · step {b.step}
      · median probability on the words <b class="mono">{(medianMass * 100).toFixed(0)}%</b>{beliefs.untrained ? " · untrained baseline captured" : ""}
      · snapshot</span>
    <input type="range" min="0" max={b.S - 1} value={s} oninput={(e) => (beliefs.snap = +(e.target as HTMLInputElement).value)} />
    <b class="mono">{b.snapshots[s]}</b>
  </div>
  <nav class="sections">
    {#each sections as [key, label]}
      <button class:on={beliefs.section === key} onclick={() => (beliefs.section = key)}>{label}</button>
    {/each}
  </nav>
  {#if beliefs.section === "trajectory"}<Trajectory />
  {:else if beliefs.section === "probes"}<Probes />
  {:else if beliefs.section === "geometry"}<Geometry />
  {:else}<Steering />{/if}
{:else}
  <p>Press <b>read stories</b> to run them through the model, one sentence at a time. A hundred stories take a few seconds.</p>
{/if}

<style>
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 6px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  .cloze { width: 150px; } .words { width: 290px; }
  input[type="number"] { width: 60px; }
  textarea { width: 100%; font-family: var(--mono); font-size: 12px; }
  .note { font-size: 11px; margin: 2px 0 8px; max-width: 900px; }
  .sections { display: flex; gap: 4px; flex-wrap: wrap; margin: 4px 0 10px; }
  .sections button { font-size: 12px; }
  .sections button.on { background: var(--accent-soft); border-color: var(--accent); }
</style>
