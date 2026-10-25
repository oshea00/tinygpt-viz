<script lang="ts">
  // Read a prediction out of every residual snapshot, not just the last: apply the final
  // layer norm and unembedding early. Reading down a column shows the guess forming.
  import { app, selection, settings } from "../../state/app.svelte";
  import { cssColor } from "../../lib/colormap";
  import { lensRow, logitLens, snapshots } from "../../lib/residual";

  const trace = $derived(app.trace!);
  const weights = $derived(app.weights);
  const stale = $derived(!!weights && trace.weightsVersion !== undefined && weights.version !== trace.weightsVersion);
  const vocab = $derived(app.hello?.vocab ?? []);
  const pos = $derived(selection.query(trace));

  // Small vocabularies: decode every snapshot here. Large ones (BPE): ask the server for a
  // compact summary, since decoding 13 × 100 rows against 4,096 tokens is GPU work.
  const big = $derived((app.hello?.config.vocab_size ?? 0) > 256 && !app.demo);
  $effect(() => {
    if (big) app.requestLens();
  });
  const topk = (p: ArrayLike<number>, k: number) =>
    Array.from(p, (v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, k);

  interface LensView {
    snaps: { label: string }[];
    T: number;
    top(s: number, t: number, k: number): { i: number; v: number }[];
    actual(s: number, t: number): number | null;   // p(actual next token), null where there is none
  }
  const lens = $derived.by((): LensView | null => {
    if (!big && weights) {
      const l = logitLens(trace, weights);
      return {
        snaps: l.snaps, T: l.T,
        top: (s, t, k) => topk(lensRow(l, s, t), k),
        actual: (s, t) => (trace.targets[t] >= 0 ? lensRow(l, s, t)[trace.targets[t]] : null),
      };
    }
    const r = app.serverLens;
    if (big && r && r.prompt === trace.prompt && r.step === trace.step) {
      return {
        snaps: snapshots(trace), T: trace.tokens.length,
        top: (s, t, k) => r.topIds[s][t].slice(0, k).map((i, j) => ({ i, v: r.topP[s][t][j] })),
        actual: (s, t) => (r.pActual[s][t] >= 0 ? r.pActual[s][t] : null),
      };
    }
    return null;
  });
  const snapSel = $derived(lens ? (settings.lensSnap >= 0 && settings.lensSnap < lens.snaps.length ? settings.lensSnap : lens.snaps.length - 1) : 0);

  const cells = $derived.by(() => {
    if (!lens) return [];
    return lens.snaps.map((_, s) =>
      Array.from({ length: lens.T }, (_, t) => {
        const best = lens.top(s, t, 1)[0];
        const target = trace.targets[t];
        return { top: best.i, pTop: best.v, pActual: lens.actual(s, t), correct: target >= 0 && best.i === target };
      }),
    );
  });

  const shade = (c: { pTop: number; pActual: number | null }) => (settings.lensMode === "confidence" ? c.pTop : c.pActual ?? 0);

  // Detail for the selected cell: its top predictions, and p(actual next) down the column.
  const detail = $derived.by(() => {
    if (!lens) return null;
    const target = trace.targets[pos];
    const column = target >= 0 ? lens.snaps.map((_, s) => lens.actual(s, pos) ?? 0) : null;
    return { top: lens.top(snapSel, pos, 8), target, column };
  });
  const accuracy = $derived(
    cells.map((r) => {
      const scored = r.filter((c) => c.pActual !== null);
      return scored.length ? scored.filter((c) => c.correct).length / scored.length : 0;
    }),
  );
</script>

{#if !lens}
  <p class="muted">{big ? "Decoding every snapshot on the server…" : "Waiting for the weights…"}</p>
{:else}
  <div class="bar">
    <span class="opts">
      shade by
      <button class:on={settings.lensMode === "confidence"} onclick={() => (settings.lensMode = "confidence")}>confidence in the top guess</button>
      <button class:on={settings.lensMode === "actual"} onclick={() => (settings.lensMode = "actual")}>p(actual next token)</button>
    </span>
    <span class="muted">underlined: the top guess is the token that actually comes next{stale ? " · weights changed since this trace" : ""}</span>
  </div>

  <div class="grid-wrap">
    <table class="grid mono" class:words={big}>
      <thead>
        <tr><th class="rl">input</th>{#each trace.text as t, i}<th class:sel={i === pos}>{t}</th>{/each}<th class="acc">top-1</th></tr>
      </thead>
      <tbody>
        {#each cells as r, s}
          <tr class:selrow={s === snapSel}>
            <th class="rl">{lens.snaps[s].label}</th>
            {#each r as c, t}
              {@const v = shade(c)}
              <td class:sel={t === pos} class:correct={c.correct}
                  style:background={cssColor("sequential", v)} style:color={v > 0.55 ? "#fff" : undefined}
                  title="{lens.snaps[s].label}, after “{trace.text[t]}”: top “{vocab[c.top]}” {(c.pTop * 100).toFixed(1)}%{c.pActual !== null ? `, actual “${vocab[trace.targets[t]]}” ${(c.pActual * 100).toFixed(1)}%` : ''}"
                  onclick={() => { selection.position = t; settings.lensSnap = s; }}>{vocab[c.top]}</td>
            {/each}
            <td class="acc">{(accuracy[s] * 100).toFixed(0)}%</td>
          </tr>
        {/each}
        <tr class="actual">
          <th class="rl">actual next</th>
          {#each trace.targets as tg, i}<td class:sel={i === pos}>{tg >= 0 ? vocab[tg] : "?"}</td>{/each}
          <td></td>
        </tr>
      </tbody>
    </table>
  </div>

  {#if detail}
    <div class="detail">
      <div>
        <h2>{lens.snaps[snapSel].label}, after position {pos} “{trace.text[pos]}”</h2>
        {#each detail.top as b}
          <div class="row" class:actual={b.i === detail.target}>
            <span class="mono t">{vocab[b.i]}</span>
            <span class="track"><span class="fill" style:width="{b.v * 100}%"></span></span>
            <span class="mono p">{(b.v * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>
      {#if detail.column}
        <div>
          <h2>p(“{vocab[detail.target]}”) through the layers</h2>
          <svg viewBox="0 0 {detail.column.length * 40} 110" width="100%" class="col">
            {#each detail.column as p, s}
              <rect x={s * 40 + 8} y={90 - p * 80} width="24" height={Math.max(1, p * 80)} class:sel={s === snapSel} />
              <text x={s * 40 + 20} y={86 - p * 80} text-anchor="middle" class="v">{(p * 100).toFixed(0)}</text>
              <text x={s * 40 + 20} y="104" text-anchor="middle" class="l">{lens.snaps[s].label.replace(" embed", "")}</text>
            {/each}
          </svg>
        </div>
      {/if}
    </div>
  {/if}
  <p class="muted note">Each row decodes one residual snapshot as if it were the last. The final row is the model's
    real prediction; earlier rows show what the stream “would say” so far. Right column: how often the top guess is right.</p>
{/if}

<style>
  .bar { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; font-size: 12px; margin-bottom: 6px; }
  .opts { display: flex; gap: 4px; align-items: center; }
  button.on { background: var(--accent-soft); border-color: var(--accent); }
  .grid-wrap { overflow-x: auto; }
  .grid { border-collapse: collapse; font-size: 11px; }
  .grid th, .grid td { width: 15px; min-width: 15px; height: 19px; padding: 0; text-align: center; cursor: pointer; }
  .grid th { font-weight: 400; color: var(--muted); }
  .grid th.rl { position: sticky; left: 0; background: var(--panel); text-align: right; padding-right: 6px; white-space: nowrap; width: auto; z-index: 1; }
  .grid.words th, .grid.words td { width: auto; padding: 0 3px; white-space: nowrap; }
  .grid td.correct { text-decoration: underline; text-decoration-thickness: 2px; font-weight: 700; }
  .grid .sel { outline: 1.5px solid var(--select); outline-offset: -1.5px; }
  .grid tr.selrow th.rl { color: var(--ink); font-weight: 600; }
  .grid .acc { width: 40px; color: var(--muted); padding-left: 6px; }
  .grid tr.actual td { color: var(--ink); font-weight: 600; border-top: 1px solid var(--line); }
  .detail { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 14px; }
  .row { display: grid; grid-template-columns: minmax(20px, max-content) 1fr 50px; gap: 6px; align-items: center; font-size: 12px; border-radius: 3px; }
  .row.actual { outline: 1.5px solid var(--select); }
  .t { text-align: center; }
  .track { height: 9px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .fill { display: block; height: 100%; background: #6d28d9; }
  .p { text-align: right; font-size: 11px; }
  .col rect { fill: #a78bfa; }
  .col rect.sel { fill: #6d28d9; }
  .col .v { font: 9px var(--mono); fill: var(--muted); }
  .col .l { font: 8.5px var(--mono); fill: var(--muted); }
  .note { font-size: 11px; margin: 8px 0 0; }
</style>
