<script lang="ts">
  // Activation patching: run a corrupted prompt, copy one activation in from a clean prompt,
  // and see how much of the clean answer comes back. Where that works is where the model
  // carries the information the two prompts differ in.
  import { app } from "../../state/app.svelte";
  import { concepts, PATCH_EXAMPLE, type PatchWhat } from "../../concepts/store.svelte";
  import { cssColor, headColor, inkOn } from "../../lib/colormap";
  import type { PatchResult } from "../../types";

  const inp = $derived(concepts.patchInput);
  const r = $derived(concepts.patches[concepts.patchWhat] ?? null);
  const stale = $derived(!!r && (r.clean !== inp.clean || r.corrupt !== inp.corrupt || r.step !== app.status?.step));
  const cell = $derived(concepts.patchCell);

  const value = (res: PatchResult, row: number, col: number) => res.grid.data[row * res.grid.shape[1] + col];
  const clamp = (v: number) => Math.max(-1, Math.min(1, v));
  const differs = (res: PatchResult, p: number) => res.cleanText[p] !== res.corruptText[p];
  const isOn = (what: PatchWhat, row: number, col: number) =>
    !!cell && cell.what === what && cell.row === row && cell.col === col && app.interventions.some((iv) => iv.kind === "patch");
</script>

<div class="inputs">
  <label>clean <input class="wide" bind:value={concepts.patchInput.clean} /></label>
  <label>corrupted <input class="wide" bind:value={concepts.patchInput.corrupt} /></label>
  <div class="row">
    <label>answer <input class="tok" bind:value={concepts.patchInput.answer} /></label>
    <label>versus <input class="tok" bind:value={concepts.patchInput.wrong} /></label>
    <span class="seg">
      {#each [["resid", "residual stream"], ["heads", "heads"]] as [w, label]}
        <button class:on={concepts.patchWhat === w} onclick={() => (concepts.patchWhat = w as PatchWhat)}>{label}</button>
      {/each}
    </span>
    <button class="primary" disabled={concepts.patchPending} onclick={() => concepts.runPatch()}>
      {concepts.patchPending ? "patching…" : "run patching"}
    </button>
    <button class="link" onclick={() => (concepts.patchInput = { ...PATCH_EXAMPLE })}>example</button>
  </div>
  <p class="muted note">The two prompts must have the same number of tokens: change one word for another that is also one token.
    Answers are read at their first token (keep the leading space).</p>
</div>

{#if !r}
  <p class="muted">Press <b>run patching</b>. Each cell runs the corrupted prompt once, with one activation copied in from the
    clean prompt, and shows how much of the clean answer that brings back.</p>
{:else}
  <p class="summary">
    Logit difference <b class="mono">{r.answer}</b> − <b class="mono">{r.wrong}</b> at the last position:
    <b class="mono">{r.ldClean.toFixed(2)}</b> clean, <b class="mono">{r.ldCorrupt.toFixed(2)}</b> corrupted.
    Each cell is the <b>recovery</b>: (patched − corrupted) / (clean − corrupted), so <b>1</b> restores the clean answer and
    <b>0</b> changes nothing.{#if stale} <span class="bad">The prompts or the weights have changed since this ran: run it again.</span>{/if}
  </p>
  <div class="scroll">
    {#if r.what === "resid"}
      <table class="grid mono">
        <thead>
          <tr><th></th>{#each r.corruptText as t, p}<th class:diff={differs(r, p)} title="position {p}{differs(r, p) ? `: clean ${r.cleanText[p]}` : ''}">{t}</th>{/each}</tr>
        </thead>
        <tbody>
          {#each r.stages as s, row}
            <tr>
              <th class="stage">{s}</th>
              {#each r.corruptText as _, col}
                {@const v = value(r, row, col)}
                <td><button class="cell" class:on={isOn("resid", row, col)} style:background={cssColor("diverging", clamp(v))}
                            style:color={inkOn("diverging", clamp(v))} title="{s}, position {col}: recovery {v.toFixed(3)}. Click to apply."
                            onclick={() => concepts.applyPatch(r, row, col)}>{Math.round(v * 100)}</button></td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <table class="grid mono">
        <thead><tr><th></th>{#each Array.from({ length: r.grid.shape[1] }, (_, h) => h) as h}<th style:color={headColor(h)}>head {h}</th>{/each}</tr></thead>
        <tbody>
          {#each Array.from({ length: r.grid.shape[0] }, (_, l) => l) as l}
            <tr>
              <th class="stage">layer {l}</th>
              {#each Array.from({ length: r.grid.shape[1] }, (_, h) => h) as h}
                {@const v = value(r, l, h)}
                <td><button class="cell head" class:on={isOn("heads", l, h)} style:background={cssColor("diverging", clamp(v))}
                            style:color={inkOn("diverging", clamp(v))} title="L{l} head {h}: recovery {v.toFixed(3)}. Click to apply."
                            onclick={() => concepts.applyPatch(r, l, h)}>{Math.round(v * 100)}</button></td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
  <p class="muted note">Numbers are recovery in percent; red restores the clean answer, blue pushes further away.
    {#if r.what === "resid"}Each row is a residual-stream snapshot (as in the logit lens), each column a position of the corrupted
      prompt; the highlighted column is where the prompts differ. Patching copies that one vector, so everything after it
      in the run sees the clean value there.{:else}Each cell copies one head's output at every position from the clean run.{/if}
    Click a cell to apply that patch to the model: the prompt becomes the corrupted one, and every other view (the
    <b>interventions</b> tab, the logit lens) shows the patched run.</p>
{/if}

<style>
  .inputs { margin-bottom: 8px; }
  .inputs > label { display: flex; gap: 6px; align-items: center; font-size: 12px; margin-bottom: 4px; }
  .inputs > label { max-width: 720px; }
  .wide { flex: 1; min-width: 0; font-family: var(--mono); font-size: 12px; }
  .tok { width: 8em; font-family: var(--mono); font-size: 12px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 4px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  .seg { display: inline-flex; gap: 2px; }
  .seg button.on { background: var(--accent-soft); border-color: var(--accent); }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 12px; }
  .note { font-size: 11px; margin: 4px 0 6px; max-width: 760px; }
  .summary { font-size: 12px; margin: 0 0 8px; max-width: 760px; }
  .bad { color: var(--bad); }
  .scroll { overflow-x: auto; max-width: 100%; }
  .grid { border-collapse: separate; border-spacing: 2px; font-size: 10.5px; }
  .grid th { font-weight: 500; white-space: nowrap; padding: 0 3px; }
  .grid thead th { color: var(--muted); }
  .grid thead th.diff { color: var(--ink); background: var(--accent-soft); border-radius: 3px; font-weight: 700; }
  .grid .stage { text-align: right; color: var(--muted); }
  .grid td { padding: 0; }
  .cell { min-width: 28px; height: 20px; padding: 0 2px; border: 1px solid transparent; border-radius: 3px; font: inherit; }
  .cell.head { min-width: 44px; }
  .cell.on { border-color: var(--ink); box-shadow: 0 0 0 1px var(--ink); }
</style>
