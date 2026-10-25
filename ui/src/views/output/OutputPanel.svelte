<script lang="ts">
  // What comes next: the distribution at the selected (or last) position, and sampling.
  import { app, selection } from "../../state/app.svelte";
  import { row } from "../../lib/tensor";

  const probs = $derived(app.trace?.byId.get("probs")?.output);
  const pos = $derived(
    app.trace ? (selection.position >= 0 && selection.position < app.trace.tokens.length ? selection.position : app.trace.tokens.length - 1) : -1,
  );
  const top = $derived.by(() => {
    if (!probs || pos < 0 || !app.hello) return [];
    const p = Array.from(row(probs, pos));
    return p.map((v, i) => ({ i, v, t: app.hello!.vocab[i] })).sort((a, b) => b.v - a.v).slice(0, 12);
  });
  const actual = $derived(app.trace && pos >= 0 ? app.trace.targets[pos] : -1);

  let temperature = $state(0.8);
</script>

<div>
  <h2>Next token after position {pos} {#if app.trace && pos >= 0}<span class="mono tokname">“{app.trace.text[pos]}”</span>{/if}</h2>
  <div class="bars">
    {#each top as b (b.i)}
      <div class="bar" class:actual={b.i === actual}>
        <span class="mono t">{b.t}</span>
        <span class="track"><span class="fill" style:width="{b.v * 100}%"></span></span>
        <span class="mono p">{(b.v * 100).toFixed(1)}%</span>
      </div>
    {/each}
  </div>
  {#if actual >= 0}<p class="muted note">outlined: the token that actually comes next in the prompt</p>{/if}

  <div class="row">
    <label class="muted">temperature <input type="range" min="0" max="2" step="0.05" bind:value={temperature} /> <span class="mono">{temperature.toFixed(2)}</span></label>
    <button onclick={() => app.send({ type: "sample", prompt: app.prompt, temperature, n: 120 })}>sample 120</button>
  </div>
  {#if app.sample}
    <p class="gen mono">
      {#each app.sample.tokens as t, i}<span style:background-color="rgb(220 38 38 / {(1 - app.sample.chosenProbs[i]) * 0.55})" title="p = {app.sample.chosenProbs[i].toFixed(3)}">{t === "␣" ? " " : t === "⏎" ? "\n" : t}</span>{/each}
    </p>
    <p class="muted note">redder = less likely choice</p>
  {/if}
</div>

<style>
  .tokname { text-transform: none; color: var(--ink); }
  .bars { display: grid; gap: 2px; }
  .bar { display: grid; grid-template-columns: 22px 1fr 48px; align-items: center; gap: 6px; font-size: 12px; border-radius: 3px; }
  .bar.actual { outline: 1.5px solid var(--select); }
  .t { text-align: center; }
  .track { height: 10px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .fill { display: block; height: 100%; background: #6d28d9; }
  .p { text-align: right; font-size: 11px; }
  .row { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .row label { display: flex; gap: 4px; align-items: center; font-size: 12px; }
  .row input { width: 90px; }
  .gen { white-space: pre-wrap; font-size: 12px; margin: 6px 0 0; max-height: 180px; overflow: auto; }
  .note { font-size: 11px; margin: 2px 0; }
</style>
