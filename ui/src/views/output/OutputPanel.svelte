<script lang="ts">
  // What comes next, and how sampling chooses: the full next-token distribution reshaped
  // live by temperature, top-k and top-p, then generation one token at a time.
  import { app, selection } from "../../state/app.svelte";
  import { describeSettings, shape } from "../../lib/sampling";
  import { row } from "../../lib/tensor";

  const trace = $derived(app.trace);
  const V = $derived(app.hello?.config.vocab_size ?? 0);
  const block = $derived(app.hello?.config.block_size ?? 100);
  const last = $derived(trace ? trace.tokens.length - 1 : -1);
  const pos = $derived(trace ? selection.query(trace) : -1);
  const travelling = $derived(app.travel !== null);
  const settings = $derived(travelling ? app.generated[app.travel!].settings : app.sampling);

  const shaped = $derived(trace && pos >= 0 ? shape(row(trace.byId.get("logits")!.output, pos), settings) : null);
  let sortByP = $state(true);
  const bars = $derived.by(() => {
    if (!shaped || !app.hello) return [];
    const b = shaped.probs.map((p, i) => ({ i, p, t: shaped.tempered[i], label: app.hello!.vocab[i] }));
    return sortByP ? b.sort((x, y) => y.p - x.p || y.t - x.t) : b;
  });
  const excluded = $derived(shaped ? shaped.probs.filter((p) => p === 0).length : 0);
  // Outline the token that actually came next: the chosen one when time travelling,
  // otherwise the prompt's own next character.
  const actual = $derived(travelling && pos === last ? app.generated[app.travel!].token : trace && pos >= 0 ? trace.targets[pos] : -1);

  // The generated text with its context window: characters the tokenizer knows, of which
  // only the last block_size are visible to the model.
  const known = $derived(new Set((app.hello?.vocab ?? []).map((v) => ({ "␣": " ", "⏎": "\n", "␄": "\x04" })[v] ?? v)));
  const genAt = $derived(new Map(app.generated.map((g, i) => [g.pos, i])));
  const viewEnd = $derived(travelling ? app.generated[app.travel!].pos : app.prompt.length);
  const windowStart = $derived.by(() => {
    let n = 0;
    for (let i = viewEnd - 1; i >= 0; i--) {
      if (known.has(app.prompt[i]) && ++n === block) return i;
    }
    return 0;
  });
  const shown = (c: string) => (c === "\x04" ? "␄\n" : c);
  const meanP = $derived(app.generated.length ? app.generated.reduce((s, g) => s + g.p, 0) / app.generated.length : 0);
</script>

<div>
  <h2>Next token after position {pos} {#if trace && pos >= 0}<span class="mono tokname">“{trace.text[pos]}”</span>{/if}</h2>

  <div class="controls">
    <label>temperature <input type="range" min="0" max="2" step="0.05" bind:value={app.sampling.temperature} disabled={travelling} />
      <span class="mono">{app.sampling.temperature === 0 ? "greedy" : app.sampling.temperature.toFixed(2)}</span></label>
    <label>top-k <input type="range" min="0" max={V} step="1" disabled={travelling}
                        value={app.sampling.topK ?? 0} oninput={(e) => { const k = +(e.target as HTMLInputElement).value; app.sampling.topK = k > 0 ? k : null; }} />
      <span class="mono">{app.sampling.topK ?? "off"}</span></label>
    <label>top-p <input type="range" min="0.05" max="1" step="0.01" disabled={travelling}
                        value={app.sampling.topP ?? 1} oninput={(e) => { const p = +(e.target as HTMLInputElement).value; app.sampling.topP = p < 1 ? p : null; }} />
      <span class="mono">{app.sampling.topP?.toFixed(2) ?? "off"}</span></label>
  </div>
  <div class="dist-head muted">
    <span>{describeSettings(settings)}{travelling ? " (as sampled)" : ""} · {excluded} of {V} excluded</span>
    <button class="link" onclick={() => (sortByP = !sortByP)}>{sortByP ? "vocab order" : "sort by p"}</button>
  </div>
  <div class="bars">
    {#each bars as b (b.i)}
      <div class="bar" class:actual={b.i === actual} class:off={b.p === 0}>
        <span class="mono t">{b.label}</span>
        <span class="track"><span class="fill" style:width="{(b.p > 0 ? b.p : b.t) * 100}%"></span></span>
        <span class="mono p">{((b.p > 0 ? b.p : b.t) * 100).toFixed(b.p > 0 && b.p < 0.001 ? 3 : 1)}%</span>
      </div>
    {/each}
  </div>
  {#if actual >= 0}<p class="muted note">outlined: {travelling && pos === last ? "the token that was sampled" : "the token that actually comes next"} · grey: excluded by top-k / top-p</p>{/if}

  <div class="gen-controls">
    <button class="primary" onclick={() => app.sampleNext()} disabled={app.auto || travelling}>sample one</button>
    {#if app.auto}
      <button onclick={() => app.stopAuto()}>stop</button>
    {:else}
      <button onclick={() => app.startAuto()} disabled={travelling}>auto</button>
    {/if}
    <label class="muted">speed <input type="range" min="0" max="600" step="20" value={600 - app.autoDelay}
                                      oninput={(e) => (app.autoDelay = 600 - +(e.target as HTMLInputElement).value)} /></label>
    <label class="muted"><input type="checkbox" bind:checked={app.stopAtEnd} /> stop at ␄</label>
  </div>

  {#if app.generated.length}
    <div class="gen-meta muted">
      {app.generated.length} generated · mean p {meanP.toFixed(2)} · model sees the last {block} characters (boxed)
      {#if travelling}
        <br />viewing character {app.travel! + 1}, sampled at step {app.generated[app.travel!].step}
        {#if trace && trace.step !== app.generated[app.travel!].step}<span class="warn">(weights have changed since)</span>{/if}
        · <button class="link" onclick={() => app.travelTo(null)}>back to latest</button>
      {:else}
        · click a character to see the trace that produced it
      {/if}
    </div>
    <p class="gen mono">
      {#each app.prompt as c, i}
        {@const g = genAt.get(i)}
        {#if g !== undefined}
          {@const tok = app.generated[g]}
          <button class="ch gen-ch" class:inwin={i >= windowStart && i < viewEnd} class:future={i > viewEnd}
                  class:here={app.travel === g}
                  style:background-color="rgb(220 38 38 / {(1 - tok.p) * 0.6})"
                  title="p = {tok.p.toFixed(3)} · rank {tok.rank} · {describeSettings(tok.settings)}"
                  onclick={() => app.travelTo(g)}>{shown(c)}</button>
        {:else}
          <span class="ch" class:inwin={i >= windowStart && i < viewEnd} class:future={i >= viewEnd}>{shown(c)}</span>
        {/if}
      {/each}
    </p>
    <p class="muted note">redder = less likely choice</p>
  {/if}
</div>

<style>
  .tokname { text-transform: none; color: var(--ink); }
  .controls { display: grid; gap: 2px; font-size: 12px; margin-bottom: 6px; }
  .controls label { display: grid; grid-template-columns: 80px 1fr 52px; align-items: center; gap: 6px; }
  .dist-head { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px; }
  .bars { display: grid; gap: 1px; max-height: 260px; overflow-y: auto; padding: 2px; }
  .bar { display: grid; grid-template-columns: 20px 1fr 50px; align-items: center; gap: 6px; font-size: 11.5px; border-radius: 3px; }
  .bar.actual { outline: 1.5px solid var(--select); }
  .t { text-align: center; }
  .track { height: 9px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .fill { display: block; height: 100%; background: #6d28d9; }
  .off .fill { background: var(--muted); opacity: 0.45; }
  .off .p, .off .t { opacity: 0.5; text-decoration: line-through; }
  .p { text-align: right; font-size: 10.5px; }
  .gen-controls { display: flex; gap: 6px; align-items: center; margin-top: 10px; flex-wrap: wrap; font-size: 12px; }
  .gen-controls label { display: flex; gap: 4px; align-items: center; }
  .gen-controls input[type="range"] { width: 70px; }
  .gen-meta { font-size: 11px; margin-top: 8px; }
  .gen { white-space: pre-wrap; font-size: 12px; margin: 4px 0 0; max-height: 220px; overflow: auto; line-height: 1.7; }
  .ch { opacity: 0.4; }
  .ch.inwin { opacity: 1; box-shadow: inset 0 1px var(--accent), inset 0 -1px var(--accent); }
  .ch.future { opacity: 0.2; }
  .gen-ch { border: none; padding: 0; border-radius: 0; font: inherit; cursor: pointer; }
  .gen-ch.here { outline: 2px solid var(--select); }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 11px; }
  .warn { color: var(--bad); }
  .note { font-size: 11px; margin: 2px 0; }
</style>
