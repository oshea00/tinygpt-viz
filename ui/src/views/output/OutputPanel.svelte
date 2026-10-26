<script lang="ts">
  // What comes next, and how sampling chooses: the full next-token distribution reshaped
  // live by temperature, top-k and top-p, then generation one token at a time.
  import { app, rawText, selection } from "../../state/app.svelte";
  import { describeSettings, shape } from "../../lib/sampling";
  import { row } from "../../lib/tensor";
  import HelpButton from "../../help/HelpButton.svelte";

  const trace = $derived(app.trace);
  const V = $derived(app.hello?.config.vocab_size ?? 0);
  const block = $derived(app.hello?.config.block_size ?? 100);
  const last = $derived(trace ? trace.tokens.length - 1 : -1);
  const pos = $derived(trace ? selection.query(trace) : -1);
  const travelling = $derived(app.travel !== null);
  const settings = $derived(travelling ? app.generated[app.travel!].settings : app.sampling);

  const shaped = $derived(trace && pos >= 0 ? shape(row(trace.byId.get("logits")!.output, pos), settings) : null);
  let sortByP = $state(true);
  const big = $derived(V > 256);
  const bars = $derived.by(() => {
    if (!shaped || !app.hello) return [];
    const b = shaped.probs.map((p, i) => ({ i, p, t: shaped.tempered[i], label: app.hello!.vocab[i] }));
    // A large vocabulary shows its top entries only: thousands of near-zero bars say nothing.
    return big ? b.sort((x, y) => y.p - x.p || y.t - x.t).slice(0, 40) : sortByP ? b.sort((x, y) => y.p - x.p || y.t - x.t) : b;
  });
  const excluded = $derived(shaped ? shaped.probs.filter((p) => p === 0).length : 0);
  // Outline the token that actually came next: the chosen one when time travelling,
  // otherwise the prompt's own next character.
  const actual = $derived(travelling && pos === last ? app.generated[app.travel!].token : trace && pos >= 0 ? trace.targets[pos] : -1);

  // The text as segments: the prompt, then one segment per generated token (a character or,
  // with BPE, usually a word). The model sees only the text its trace covers: the window.
  const viewEnd = $derived(travelling ? app.generated[app.travel!].pos : app.prompt.length);
  const windowStart = $derived(trace && trace.prompt === app.viewPrompt() ? Math.max(0, viewEnd - rawText(trace.text.join("")).length) : 0);
  const segments = $derived.by(() => {
    const out: { start: number; text: string; gen: number }[] = [];
    const firstGen = app.generated[0]?.pos ?? app.prompt.length;
    // The prompt part is split at the window start so its boxed part is exact.
    const cut = Math.min(Math.max(windowStart, 0), firstGen);
    if (cut > 0) out.push({ start: 0, text: app.prompt.slice(0, cut), gen: -1 });
    if (firstGen > cut) out.push({ start: cut, text: app.prompt.slice(cut, firstGen), gen: -1 });
    app.generated.forEach((g, i) => {
      const end = app.generated[i + 1]?.pos ?? app.prompt.length;
      out.push({ start: g.pos, text: app.prompt.slice(g.pos, end), gen: i });
    });
    return out;
  });
  const shown = (t: string) => t.replace(/\x04/g, "␄\n");
  const fmtMacs = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}k`);
  const meanP = $derived(app.generated.length ? app.generated.reduce((s, g) => s + g.p, 0) / app.generated.length : 0);
</script>

<div>
  <h2>Next token after position {pos} {#if trace && pos >= 0}<span class="mono tokname">“{trace.text[pos]}”</span>{/if}
    <HelpButton topic="generation" label="next token and generation" /></h2>

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
    {#if big}<span>top 40 of {V}</span>
    {:else}<button class="link" onclick={() => (sortByP = !sortByP)}>{sortByP ? "vocab order" : "sort by p"}</button>{/if}
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
    <label class="muted" title="reuse earlier positions' keys and values; compute only the new position (no gradients)">
      <input type="checkbox" bind:checked={app.kvCache} onchange={() => app.run()} /> KV cache</label>
  </div>
  {#if app.kvCache}
    {@const inc = trace?.incremental}
    {@const k = app.kvTotals}
    <div class="kvstats">
      {#if inc}
        {#if inc.reused}
          last step: computed {inc.computed} of {inc.reused + inc.computed} positions,
          <b>{fmtMacs(inc.multiplyAdds)}</b> multiply-adds instead of {fmtMacs(inc.multiplyAddsFull)}
        {:else}
          <span class="reset">cache rebuilt: {inc.reason}</span>
        {/if}
        {#if k.steps > 1}<br />so far: {fmtMacs(k.withCache)} vs {fmtMacs(k.without)} without the cache
          ({(k.without / Math.max(1, k.withCache)).toFixed(1)}× less work){k.resets ? `, ${k.resets} rebuild${k.resets > 1 ? "s" : ""}` : ""}{/if}
      {:else}
        <span class="muted">Gradients are off while the cache is on. Generate to see it work.</span>
      {/if}
    </div>
  {/if}

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
      {#each segments as seg}
        {@const inwin = seg.start >= windowStart && seg.start < viewEnd}
        {#if seg.gen >= 0}
          {@const tok = app.generated[seg.gen]}
          <button class="ch gen-ch" class:inwin class:future={seg.start > viewEnd} class:here={app.travel === seg.gen}
                  style:background-color="rgb(220 38 38 / {(1 - tok.p) * 0.6})"
                  title="p = {tok.p.toFixed(3)} · rank {tok.rank} · {describeSettings(tok.settings)}"
                  onclick={() => app.travelTo(seg.gen)}>{shown(seg.text)}</button>
        {:else}
          <span class="ch" class:inwin class:future={seg.start >= viewEnd}>{shown(seg.text)}</span>
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
  .bar { display: grid; grid-template-columns: minmax(20px, auto) 1fr 50px; align-items: center; gap: 6px; font-size: 11.5px; border-radius: 3px; }
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
  .kvstats { font-size: 11px; margin-top: 6px; padding: 4px 6px; border-radius: 6px; background: var(--accent-soft); line-height: 1.5; }
  .kvstats .reset { color: var(--bad); }
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
