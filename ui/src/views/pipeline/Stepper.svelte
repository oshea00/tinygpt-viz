<script lang="ts">
  // Walks down the network stage by stage, past the loss, and back up through the gradients.
  import { app, selection } from "../../state/app.svelte";
  import { stepSequence } from "../../lib/pipeline";

  const seq = $derived(app.trace ? stepSequence(app.trace) : []);
  const index = $derived(seq.findIndex((s) => s.id === selection.stageId && s.phase === selection.phase));
  const layers = $derived(app.hello ? Array.from({ length: app.hello.config.n_layer }, (_, l) => l) : []);

  let playing = $state(false);
  let speed = $state(4);   // stages per second

  function go(i: number) {
    const s = seq[Math.max(0, Math.min(seq.length - 1, i))];
    if (s) selection.select(s.id, s.phase);
  }

  $effect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      if (index >= seq.length - 1) playing = false;
      else go(index + 1);
    }, 1000 / speed);
    return () => clearInterval(timer);
  });

  function onKey(e: KeyboardEvent) {
    if ((e.target as HTMLElement).closest("input, textarea, select")) return;
    if (e.key === "ArrowRight" || e.key === "j") go(index + 1);
    else if (e.key === "ArrowLeft" || e.key === "k") go(index - 1);
    else return;
    e.preventDefault();
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="stepper">
  <button onclick={() => go(0)} title="first stage">⇤</button>
  <button onclick={() => go(index - 1)} title="previous stage (←)">◀</button>
  <button class="primary" onclick={() => (playing = !playing)}>{playing ? "pause" : "play"}</button>
  <button onclick={() => go(index + 1)} title="next stage (→)">▶</button>
  <label class="muted">speed <input type="range" min="1" max="20" bind:value={speed} /></label>
  <label class="muted">jump to
    <select onchange={(e) => {
      const v = (e.target as HTMLSelectElement).value;
      if (v === "in") selection.select("embed.tok", "forward");
      else if (v === "out") selection.select("logits", "forward");
      else if (v === "loss") selection.select("loss", "backward");
      else selection.select(`L${v}.ln1`, "forward");
      (e.target as HTMLSelectElement).value = "";
    }}>
      <option value="">…</option>
      <option value="in">input</option>
      {#each layers as l}<option value={l}>block {l}</option>{/each}
      <option value="out">output</option>
      <option value="loss">start of backward pass</option>
    </select>
  </label>
  <span class="pos">
    {#if index >= 0}
      stage {index + 1} / {seq.length}
      <span class="phase" class:back={selection.phase === "backward"}>{selection.phase === "forward" ? "forward" : "backward ∂L/∂x"}</span>
    {/if}
  </span>
</div>

<style>
  .stepper { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  label { display: flex; align-items: center; gap: 4px; font-size: 12px; margin-left: 6px; }
  input[type="range"] { width: 80px; }
  .pos { margin-left: auto; font-size: 12px; }
  .phase { margin-left: 6px; padding: 1px 6px; border-radius: 10px; background: var(--accent-soft); }
  .phase.back { background: #fde68a; color: #78350f; }
</style>
