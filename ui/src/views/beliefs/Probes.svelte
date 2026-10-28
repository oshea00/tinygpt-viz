<script lang="ts">
  // Can the belief be read off the story's own residual stream? A ridge probe per snapshot from
  // the vector at a sentence's last token to the readout, tested on held-out stories, beside two
  // baselines: always guessing the training average, and the untrained model's vectors.
  import { beliefs } from "../../beliefs/store.svelte";
  import { wordColor } from "../concepts/colors";

  let running = $state(false);
  const b = $derived(beliefs.current!);
  const res = $derived(beliefs.probes);
  const s = $derived(beliefs.snapIndex(b));

  function run() {
    running = true;
    // Let the button repaint before the (blocking) fits.
    setTimeout(() => {
      beliefs.fitProbes();
      running = false;
    }, 30);
  }

  const best = $derived(res ? res.trained.reduce((a, r, i) => (r.rmse < res.trained[a].rmse ? i : a), 0) : 0);
  const ymax = $derived(res ? Math.max(res.baseline, ...res.trained.map((r) => r.rmse), ...(res.untrained ?? []).map((r) => r.rmse)) * 1.1 : 1);
  const W = 560, H = 220, P = 34;
  const bw = $derived((W - 2 * P) / b.S);
  const y = (v: number) => H - P - (v / ymax) * (H - 2 * P);
  const RW = 460, RH = 24 + 18 * 6;
</script>

<div class="row">
  <button class="primary" disabled={running} onclick={run}>{running ? "fitting…" : "fit probes at every snapshot"}</button>
  <span class="muted">ridge regression, one output per word, trained on 4 of every 5 stories and tested on the rest</span>
</div>
{#if res}
  <h2>Test error at each snapshot <span class="muted">lower is better · best {res.trained[best].layer}: {res.trained[best].rmse.toFixed(3)}</span></h2>
  <svg viewBox="0 0 {W} {H}" width="100%" class="chart">
    {#each [0, ymax / 2, ymax] as v}<line x1={P} x2={W - P} y1={y(v)} y2={y(v)} class="grid" /><text x={P - 4} y={y(v) + 3} text-anchor="end">{v.toFixed(2)}</text>{/each}
    <line x1={P} x2={W - P} y1={y(res.baseline)} y2={y(res.baseline)} class="chance"><title>always the training average: {res.baseline.toFixed(3)}</title></line>
    {#each res.trained as r, i}
      {@const x0 = P + i * bw}
      {#if i === s}<rect x={x0} y={P / 2} width={bw} height={H - 1.5 * P} class="sel" />{/if}
      <rect x={x0 + 2} y={y(r.rmse)} width={bw / 2 - 2} height={y(0) - y(r.rmse)} class="tr"><title>trained {r.layer}: RMSE {r.rmse.toFixed(3)}</title></rect>
      {#if res.untrained}
        {@const u = res.untrained[i]}
        <rect x={x0 + bw / 2} y={y(u.rmse)} width={bw / 2 - 2} height={y(0) - y(u.rmse)} class="un"><title>untrained {u.layer}: RMSE {u.rmse.toFixed(3)}</title></rect>
      {/if}
      {#if i % 2 === 0}<text x={x0 + bw / 2} y={H - 10} text-anchor="middle">{r.layer.replace(".resid_", " ").replace("embed.sum", "x₀")}</text>{/if}
    {/each}
  </svg>
  <p class="muted note"><span class="k tr">trained</span> {#if res.untrained}<span class="k un">untrained model's vectors</span>{/if}
    · dashed: always guessing the training average. Root-mean-square error of the predicted share of each word, on stories the
    probe never saw. The untrained baseline reads the <i>trained</i> model's belief from random-weight vectors: whatever it gets
    comes from which words are in the story, not from anything learned.</p>

  <h2>Per word at {b.snapshots[s]} <span class="muted">correlation between probe and readout on held-out stories</span></h2>
  <svg viewBox="0 0 {RW} {RH}" width="100%" class="rchart">
    {#each b.words as w, k}
      {@const r = res.trained[s].r[k]}
      <text x="60" y={20 + k * 18 + 4} text-anchor="end">{w}</text>
      <rect x="66" y={20 + k * 18 - 6} width={Math.max(0, r) * (RW - 240)} height="12" fill={wordColor(k)} />
      <text x={70 + Math.max(0, r) * (RW - 240)} y={20 + k * 18 + 4}>{r.toFixed(2)}{res.untrained ? ` (untrained ${res.untrained[s].r[k].toFixed(2)})` : ""}</text>
    {/each}
  </svg>
  <p class="muted note">A word the readout rarely gives much weight to is hard to predict, and its correlation is lower. Move the
    snapshot slider to see where each one becomes readable. The trajectories section draws these predictions as dashed lines.
    A probe shows the belief is <i>readable</i> from that layer; steering tests whether the model uses it.</p>
{/if}

<style>
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .chart { max-width: 700px; }
  .rchart { max-width: 600px; }
  .grid { stroke: var(--line); }
  .chance { stroke: var(--muted); stroke-dasharray: 4 3; stroke-width: 1.5; }
  .sel { fill: var(--accent-soft); }
  text { font: 9px var(--mono); fill: var(--muted); }
  .tr { fill: var(--accent); } .un { fill: #d97706; }
  .note { font-size: 11px; max-width: 760px; }
  .k { padding: 0 4px; border-radius: 3px; color: #fff; }
  .k.tr { background: var(--accent); } .k.un { background: #d97706; }
</style>
