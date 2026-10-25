<script lang="ts">
  // Jump to any saved checkpoint, or play through them all: watch attention heads organise,
  // embeddings cluster and the text go from noise to almost-English, without retraining.
  import { app } from "../../state/app.svelte";

  const steps = $derived(app.status?.checkpoints ?? []);
  const current = $derived(steps.indexOf(app.status?.step ?? -1));
  let index = $state(0);
  let playing = $state(false);
  let timer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    if (current >= 0 && !playing) index = current;
  });

  function load(i: number) {
    index = i;
    clearTimeout(timer);
    timer = setTimeout(() => app.send({ type: "loadCheckpoint", step: steps[i] }), 150);
  }

  // Play-through: advance only once the previous checkpoint has loaded.
  $effect(() => {
    if (!playing) return;
    if (index >= steps.length - 1) {
      playing = false;
      return;
    }
    if (current !== index) return;
    const t = setTimeout(() => load(index + 1), 700);
    return () => clearTimeout(t);
  });
</script>

{#if steps.length}
  <div class="replay">
    <div class="head">
      <span class="muted">replay checkpoints</span>
      <span class="mono">step {steps[index]}</span>
    </div>
    <input type="range" min="0" max={steps.length - 1} value={index} disabled={app.status?.playing}
           oninput={(e) => { playing = false; load(+(e.target as HTMLInputElement).value); }} />
    <div class="row">
      <button disabled={app.status?.playing} onclick={() => { if (index >= steps.length - 1) load(0); playing = !playing; }}>
        {playing ? "stop" : "play through"}
      </button>
      <button disabled={app.status?.playing} onclick={() => app.send({ type: "save" })} title="save the current weights as a checkpoint">save now</button>
      <span class="muted note">{steps.length} saved · log-spaced, so early learning is shown in detail</span>
    </div>
  </div>
{/if}

<style>
  .replay { margin: 10px 0 6px; font-size: 12px; }
  .head { display: flex; justify-content: space-between; }
  input[type="range"] { width: 100%; }
  .row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .note { font-size: 10.5px; }
</style>
