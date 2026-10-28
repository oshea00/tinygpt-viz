<script lang="ts">
  // One story's belief, sentence by sentence: the readout per word (and the probe's prediction),
  // and the story's path through two maps of every captured sentence: the belief space itself and
  // the residual stream at the chosen snapshot.
  import { app, selection, settings } from "../../state/app.svelte";
  import { beliefs } from "../../beliefs/store.svelte";
  import { centre, isTestStory, readoutRows, snapshotRows, type Beliefs } from "../../lib/beliefs";
  import { pca, project } from "../../lib/concepts";
  import { wordColor } from "../concepts/colors";

  const b = $derived(beliefs.current!);
  const s = $derived(beliefs.snapIndex(b));
  const story = $derived(Math.min(beliefs.story, b.stories.length - 1));
  const idx = $derived(b.prefixes.map((p, i) => (p.story === story ? i : -1)).filter((i) => i >= 0));
  const sel = $derived(beliefs.sentence >= 0 && beliefs.sentence < idx.length ? beliefs.sentence : idx.length - 1);
  const text = $derived(b.stories[story]);
  const probe = $derived(beliefs.probes?.trained[s]?.fit.pred ?? null);
  const top = (i: number) => {
    const row = b.readout.subarray(i * b.K, (i + 1) * b.K);
    return row.indexOf(Math.max(...row));
  };

  // Line chart of the readout across sentences.
  const W = 560, H = 190, L = 30, R = 10, T = 10, B = 22;
  const x = (k: number) => L + (idx.length > 1 ? (k / (idx.length - 1)) * (W - L - R) : (W - L - R) / 2);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const line = (vals: number[]) => vals.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join("");

  // Maps: PCA of every sentence, with this story's path drawn over it.
  const M = 300, PAD = 12;
  function layout(rows: Float64Array[]) {
    const c = centre(rows);
    const { components, variance, total } = pca(c, 2);
    const xy = project(c, components);
    const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    return {
      explained: variance.map((v) => v / total),
      pts: xy.map(([px, py]) => [PAD + ((px - x0) / (x1 - x0 || 1)) * (M - 2 * PAD), PAD + ((py - y0) / (y1 - y0 || 1)) * (M - 2 * PAD)]),
    };
  }
  const behaviourMap = $derived(layout(readoutRows(b)));
  const activationMap = $derived(layout(snapshotRows(b, s)));
  const maps = $derived([["belief space (the readout)", behaviourMap], [`residual stream at ${b.snapshots[s]}`, activationMap]] as const);

  function open(c: Beliefs, i: number) {
    const p = c.prefixes[i];
    app.prompt = c.stories[p.story].slice(0, p.end);
    app.promptEdited();
    app.run();
    selection.select(c.snapshots[beliefs.snapIndex(c)], "forward");
    selection.position = -1;
    settings.tab = "inspector";
  }
</script>

<div class="row">
  <label>story <select value={story} onchange={(e) => { beliefs.story = +(e.target as HTMLSelectElement).value; beliefs.sentence = -1; }}>
    {#each b.stories as st, i}<option value={i}>{i + 1}{isTestStory(i) ? " (held out)" : ""}: {st.slice(0, 60)}…</option>{/each}
  </select></label>
  <div class="legend">{#each b.words as w, k}<span><i style:background={wordColor(k)}></i>{w}</span>{/each}</div>
</div>

<p class="story">
  {#each idx as i, k}
    {@const p = b.prefixes[i]}
    <span class="sent" class:read={k <= sel} class:cur={k === sel} role="button" tabindex="0" style:border-color={wordColor(top(i))}
          onclick={() => (beliefs.sentence = k)} onkeydown={(e) => e.key === "Enter" && (beliefs.sentence = k)}>{text.slice(p.start, p.end)}</span>{" "}
  {/each}
</p>

<h2>Belief after each sentence <span class="muted">solid: the readout · dashed: the probe at {b.snapshots[s]}{probe ? "" : " (fit probes to see it)"}</span></h2>
<svg viewBox="0 0 {W} {H}" width="100%" class="chart">
  {#each [0, 0.5, 1] as v}<line x1={L} x2={W - R} y1={y(v)} y2={y(v)} class="grid" /><text x={L - 4} y={y(v) + 3} text-anchor="end">{v}</text>{/each}
  <line x1={x(sel)} x2={x(sel)} y1={T} y2={H - B} class="cursor" />
  {#each b.words as w, k}
    <path d={line(idx.map((i) => b.readout[i * b.K + k]))} stroke={wordColor(k)} class="series"><title>{w}</title></path>
    {#if probe}<path d={line(idx.map((i) => probe[i][k]))} stroke={wordColor(k)} class="series pred"><title>{w}, probe</title></path>{/if}
  {/each}
  {#each idx as i, k}
    <rect x={x(k) - 6} y={T} width="12" height={H - T - B} class="hit" role="button" tabindex="-1" onclick={() => (beliefs.sentence = k)} onkeydown={() => {}}>
      <title>sentence {k + 1}: {b.words.map((w, j) => `${w} ${(b.readout[i * b.K + j] * 100).toFixed(0)}%`).join(" · ")}</title>
    </rect>
    <text x={x(k)} y={H - 8} text-anchor="middle">{k + 1}</text>
  {/each}
</svg>

<div class="two">
  {#each maps as [title, m]}
    <section>
      <h2>{title} <span class="muted">PC1 {(m.explained[0] * 100).toFixed(0)}% · PC2 {(m.explained[1] * 100).toFixed(0)}%</span></h2>
      <svg viewBox="0 0 {M} {M}" width="100%">
        <rect x="0" y="0" width={M} height={M} class="bg" />
        {#each m.pts as [px, py], i}
          <circle cx={px} cy={py} r="2.2" fill={wordColor(top(i))} class="dot" role="button" tabindex="-1" onclick={() => open(b, i)} onkeydown={() => {}}>
            <title>story {b.prefixes[i].story + 1}, sentence {b.prefixes[i].sentence + 1}: {b.words[top(i)]}</title>
          </circle>
        {/each}
        <path d={idx.map((i, k) => `${k ? "L" : "M"}${m.pts[i][0].toFixed(1)},${m.pts[i][1].toFixed(1)}`).join("")} class="path" />
        {#each idx as i, k}
          <circle cx={m.pts[i][0]} cy={m.pts[i][1]} r={k === sel ? 6 : 4.5} fill={wordColor(top(i))} class="step" class:cur={k === sel}
                  role="button" tabindex="-1" onclick={() => (beliefs.sentence = k)} onkeydown={() => {}}>
            <title>sentence {k + 1}: {text.slice(b.prefixes[i].start, b.prefixes[i].end)}</title>
          </circle>
          <text x={m.pts[i][0] + 6} y={m.pts[i][1] - 5} class="num">{k + 1}</text>
        {/each}
      </svg>
    </section>
  {/each}
</div>
<p class="muted note">Every sentence of every story is a faint dot, coloured by its strongest belief; the chosen story is the path,
  numbered by sentence. If the two maps look alike, the residual stream lays beliefs out the way the readout does. Click a faint
  dot to open that story, up to that sentence, in the inspector.</p>

<style>
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 6px; }
  .row select { max-width: 520px; }
  .legend { display: flex; gap: 10px; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .story { font-size: 13px; line-height: 1.7; max-width: 900px; }
  .sent { cursor: pointer; color: var(--muted); border-bottom: 3px solid transparent; }
  .sent.read { color: var(--ink); border-bottom-color: inherit; }
  .sent:not(.read) { border-bottom-color: transparent !important; }
  .sent.cur { background: var(--accent-soft); }
  .chart { max-width: 760px; display: block; }
  .grid { stroke: var(--line); }
  .cursor { stroke: var(--accent); stroke-dasharray: 3 3; }
  .series { fill: none; stroke-width: 2; }
  .series.pred { stroke-dasharray: 4 3; stroke-width: 1.5; }
  .hit { fill: transparent; cursor: pointer; }
  text { font: 9px var(--mono); fill: var(--muted); }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; max-width: 900px; }
  .bg { fill: var(--panel); stroke: var(--line); }
  .dot { opacity: 0.35; cursor: pointer; }
  .path { fill: none; stroke: var(--ink); stroke-width: 1.5; }
  .step { stroke: var(--ink); stroke-width: 1; cursor: pointer; }
  .step.cur { stroke-width: 2.5; }
  .num { font-size: 10px; fill: var(--ink); }
  .note { font-size: 11px; }
</style>
