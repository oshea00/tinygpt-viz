<script lang="ts">
  // One training step, start to finish: forward, loss, backward, update, then the same
  // prompt again, showing that the correct next tokens became slightly more likely.
  import { app } from "../../state/app.svelte";
  import { weightRange } from "../../lib/scale";
  import { dims2, fmt } from "../../lib/tensor";
  import Heatmap from "../inspector/Heatmap.svelte";

  const ex = $derived(app.stepExplain);
  const matrices = $derived(ex ? Object.keys(ex.before).filter((n) => ex.before[n].shape.length === 2) : []);
  let chosen = $state("L0.W_q");
  const name = $derived(ex && ex.before[chosen] ? chosen : matrices[0]);

  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);

  // Adam's update is not simply −lr·gradient: it rescales each entry by its running
  // gradient size. How often does it at least move against the gradient?
  const agreement = $derived.by(() => {
    if (!ex || !name) return null;
    const g = ex.grad[name]?.data, d = ex.delta[name].data;
    if (!g) return null;
    let same = 0, n = 0;
    for (let i = 0; i < d.length; i++) {
      if (g[i] === 0 || d[i] === 0) continue;
      n++;
      if (Math.sign(d[i]) === -Math.sign(g[i])) same++;
    }
    return n ? same / n : null;
  });

  const busy = $derived(app.status?.playing ?? false);
</script>

<div class="onestep">
  <div class="top">
    <button class="primary" disabled={busy} onclick={() => app.send({ type: "stepExplain", prompt: app.prompt })}>
      take one training step on this prompt
    </button>
    {#if busy}<span class="muted">pause training first</span>{/if}
  </div>

  {#if !ex}
    <p class="muted">Runs one AdamW update using the prompt above as the whole batch, then shows what changed.</p>
  {:else}
    <div class="summary">
      <div><div class="muted">loss on the prompt</div><b class="mono">{fmt(ex.lossBefore)} → {fmt(ex.lossAfter)}</b>
        <span class={ex.lossAfter < ex.lossBefore ? "ok" : "bad"}>{ex.lossAfter < ex.lossBefore ? "↓" : "↑"}</span></div>
      <div><div class="muted">mean p(correct next token)</div><b class="mono">{(mean(ex.pBefore) * 100).toFixed(2)}% → {(mean(ex.pAfter) * 100).toFixed(2)}%</b></div>
      <div><div class="muted">step · learning rate</div><b class="mono">{ex.step} · {fmt(ex.lr, 5)}</b></div>
    </div>

    <h2>p(correct next token) at each position, before ▪ and after ▪</h2>
    <div class="pchart">
      {#each ex.pBefore as pb, i}
        {@const pa = ex.pAfter[i]}
        <div class="col" title="next token “{ex.text[i + 1]}”: {(pb * 100).toFixed(2)}% → {(pa * 100).toFixed(2)}%">
          <div class="bars">
            <span class="b before" style:height="{pb * 100}%"></span>
            <span class="b after" class:down={pa < pb} style:height="{pa * 100}%"></span>
          </div>
          <span class="mono lab" class:word={ex.text[i + 1].length > 1}>{ex.text[i + 1]}</span>
        </div>
      {/each}
    </div>

    <h2>Weight, gradient, and the update Adam applied</h2>
    <label>matrix
      <select bind:value={chosen}>
        {#each matrices as m}<option value={m}>{m} [{ex.before[m].shape.join("×")}]</option>{/each}
      </select>
    </label>
    {#if name}
      {@const [rows, cols] = dims2(ex.before[name])}
      <div class="triple" class:wide={cols > rows * 1.5}>
        {#each [["W (before)", ex.before[name]], ["∇W  gradient", ex.grad[name]], ["ΔW  actual change", ex.delta[name]]] as const as [label, t]}
          {#if t}
            {@const r = weightRange(t, 99)}
            <div>
              <div class="muted lbl">{label} <span class="mono">±{fmt(r.hi, 3)}</span></div>
              <Heatmap tensor={t} range={r} maxHeight={260} maxCell={6} />
            </div>
          {/if}
        {/each}
      </div>
      {#if agreement !== null}
        <p class="muted">ΔW moves against the gradient in <b>{(agreement * 100).toFixed(1)}%</b> of entries. Adam divides each
          entry by its own running gradient size, so small, consistent gradients get large steps: the update's pattern
          often looks quite different from the raw gradient.</p>
      {/if}
    {/if}
  {/if}
</div>

<style>
  .top { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
  .summary { display: flex; gap: 28px; flex-wrap: wrap; margin: 6px 0 14px; }
  .summary b { font-size: 15px; }
  .ok { color: var(--ok); }
  .bad { color: var(--bad); }
  .pchart { display: flex; gap: 1px; align-items: flex-end; overflow-x: auto; padding-bottom: 4px; margin-bottom: 14px; }
  .col { display: flex; flex-direction: column; align-items: center; width: 11px; flex: none; }
  .bars { height: 90px; width: 100%; display: flex; align-items: flex-end; gap: 1px; }
  .b { flex: 1; display: block; min-height: 1px; }
  .before { background: #a5b4fc; }
  .after { background: var(--ok); }
  .after.down { background: var(--bad); }
  .lab { font-size: 9px; }
  /* Multi-character tokens would overlap at this column width: run them downwards. */
  .lab.word { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; line-height: 11px; max-height: 70px; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
  .triple { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 8px; }
  .triple.wide { grid-template-columns: 1fr; }
  .lbl { font-size: 11px; margin-bottom: 2px; }
  label { font-size: 12px; display: flex; gap: 6px; align-items: center; }
</style>
