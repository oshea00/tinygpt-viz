<script lang="ts">
  // The stream as a shared workspace: one position's vector at every snapshot, how much
  // each sublayer writes into it, and how gradients flow back along it.
  import type { Tensor } from "../../types";
  import { app, selection, settings } from "../../state/app.svelte";
  import { gradientHighway, rowNorm, snapshots, writes } from "../../lib/residual";
  import { tensorRange } from "../../lib/scale";
  import { dims2, fmt, row } from "../../lib/tensor";
  import Heatmap from "../inspector/Heatmap.svelte";
  import DimTrack from "./DimTrack.svelte";
  import Trajectories from "./Trajectories.svelte";

  const trace = $derived(app.trace!);
  const pos = $derived(selection.query(trace));
  const snaps = $derived(snapshots(trace));
  const ws = $derived(writes(trace));
  const d = $derived(dims2(trace.byId.get("embed.sum")!.output)[1]);

  // Rows: snapshots of x (or x₀ then each sublayer's write), columns: the d dimensions.
  const stream = $derived.by((): { t: Tensor; labels: string[] } => {
    const ids = settings.residualMode === "snapshots" ? snaps.map((s) => s.id) : ["embed.sum", ...ws.map((w) => w.delta)];
    const labels = settings.residualMode === "snapshots" ? snaps.map((s) => s.label) : ["x₀ embed", ...ws.map((w) => `${w.label} Δ`)];
    const data = new Float32Array(ids.length * d);
    ids.forEach((id, i) => data.set(row(trace.byId.get(id)!.output, pos), i * d));
    return { t: { shape: [ids.length, d], data }, labels };
  });
  const range = $derived(tensorRange([stream.t.data], "diverging", settings.clip, settings.symlog, "these rows"));

  // Write sizes: ‖Δ‖ each sublayer adds at this position, next to ‖x‖ after it.
  const sizes = $derived(ws.map((w) => ({ label: w.label, delta: rowNorm(trace.byId.get(w.delta)!.output, pos), x: rowNorm(trace.byId.get(w.after)!.output, pos) })));
  const x0 = $derived(rowNorm(trace.byId.get("embed.sum")!.output, pos));
  const sizeMax = $derived(Math.max(x0, ...sizes.flatMap((s) => [s.delta, s.x])));

  // Every position at once: ‖Δ‖ per sublayer × position.
  const allWrites = $derived.by((): Tensor => {
    const T = trace.tokens.length;
    const data = new Float32Array(ws.length * T);
    ws.forEach((w, i) => { for (let t = 0; t < T; t++) data[i * T + t] = rowNorm(trace.byId.get(w.delta)!.output, t); });
    return { shape: [ws.length, T], data };
  });
  const allRange = $derived(tensorRange([allWrites.data], "sequential", 99, false, "all writes"));

  const flow = $derived(gradientHighway(trace, settings.highwayScope === "position" ? pos : -1));
  const flowMax = $derived(flow ? Math.max(...flow.flatMap((f) => [f.total, f.through, f.branch])) : 1);
</script>

<section>
  <div class="bar">
    <h2>Position {pos} “{trace.text[pos]}” through the stream</h2>
    <span class="opts">
      <button class:on={settings.residualMode === "snapshots"} onclick={() => (settings.residualMode = "snapshots")}>snapshots of x</button>
      <button class:on={settings.residualMode === "writes"} onclick={() => (settings.residualMode = "writes")}>what each sublayer writes</button>
      <label><input type="checkbox" bind:checked={settings.symlog} /> symmetric log</label>
    </span>
  </div>
  <Heatmap tensor={stream.t} {range} rowLabels={stream.labels} maxCell={14}
           selCol={selection.trackDim >= 0 ? selection.trackDim : selection.dim < d ? selection.dim : -1}
           onpick={(_, c) => { selection.dim = c; selection.trackDim = c; }} />
  <p class="muted note">columns: the {d} residual dimensions · range <span class="mono">±{fmt(range.hi, 3)}</span>
    {settings.residualMode === "snapshots" ? "· magnitudes tend to grow with depth; try symmetric log" : "· the rows sum to the final snapshot"}
    · click a column to follow that dimension</p>
</section>

<DimTrack />
<Trajectories />

<div class="two">
  <section>
    <h2>Write size at position {pos}</h2>
    <svg width="100%" viewBox="0 0 420 {24 + sizes.length * 22}" class="sizes">
      <text x="0" y="14" class="lbl">x₀ embed</text>
      <rect x="80" y="5" height="10" width={(x0 / sizeMax) * 220} class="x" />
      <text x={86 + (x0 / sizeMax) * 220} y="14" class="num">{fmt(x0, 2)}</text>
      {#each sizes as s, i}
        {@const y = 24 + i * 22}
        <text x="0" y={y + 12} class="lbl">{s.label}</text>
        <rect x="80" y={y + 1} height="8" width={(s.delta / sizeMax) * 220} class:attn={s.label.endsWith("attn")} class="delta" />
        <rect x="80" y={y + 11} height="4" width={(s.x / sizeMax) * 220} class="x" />
        <text x={86 + (Math.max(s.delta, s.x) / sizeMax) * 220} y={y + 12} class="num">Δ {fmt(s.delta, 2)} · x {fmt(s.x, 2)}</text>
      {/each}
    </svg>
    <p class="muted note">thick bar: ‖Δ‖, the size of what the sublayer adds · thin bar: ‖x‖ after adding it.
      Each sublayer reads the stream and adds a correction; nothing overwrites it.</p>
  </section>

  <section>
    <div class="bar">
      <h2>Gradient highway</h2>
      <span class="opts">
        <button class:on={settings.highwayScope === "all"} onclick={() => (settings.highwayScope = "all")}>whole sequence</button>
        <button class:on={settings.highwayScope === "position"} onclick={() => (settings.highwayScope = "position")}>position {pos}</button>
      </span>
    </div>
    {#if !flow}
      <p class="muted">This trace has no gradients (generation skips them). Stop auto generation or edit the prompt.</p>
    {:else}
      <svg width="100%" viewBox="0 0 420 {flow.length * 30 + 8}" class="sizes">
        {#each flow as f, i}
          {@const y = 4 + i * 30}
          <text x="0" y={y + 12} class="lbl">{f.label}</text>
          <rect x="80" y={y} height="7" width={(f.through / flowMax) * 220} class="through" />
          <rect x="80" y={y + 8} height="7" width={(f.branch / flowMax) * 220} class="branch" />
          <rect x="80" y={y + 16} height="4" width={(f.total / flowMax) * 220} class="x" />
          <text x={86 + (Math.max(f.through, f.branch, f.total) / flowMax) * 220} y={y + 12} class="num">{fmt(f.total, 3)}</text>
        {/each}
      </svg>
      <p class="muted note">Read top to bottom, the order the backward pass runs. At each residual add the gradient
        <span class="k through-k">passes straight through</span> and also gains
        <span class="k branch-k">a part back through the sublayer</span>; thin bar: their sum, ‖∂L/∂x‖ before the add.
        Because the straight-through part never shrinks, early layers still receive a strong signal.</p>
    {/if}
  </section>
</div>

<section>
  <h2>Write size at every position</h2>
  <Heatmap tensor={allWrites} range={allRange} rowLabels={ws.map((w) => w.label)} colLabels={trace.text} maxCell={14}
           selCol={pos} onpick={(_, c) => (selection.position = c)} />
  <p class="muted note">‖Δ‖ per sublayer (rows) and position (columns), 0 … {fmt(allRange.hi, 2)}. Click a column to select that position.</p>
</section>

<style>
  section { margin-bottom: 16px; }
  .bar { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .opts { display: flex; gap: 4px; align-items: center; font-size: 12px; }
  .opts label { display: flex; gap: 4px; align-items: center; margin-left: 6px; }
  button.on { background: var(--accent-soft); border-color: var(--accent); }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  @media (max-width: 1400px) { .two { grid-template-columns: minmax(0, 1fr); } }
  .sizes { max-width: 520px; display: block; }
  .lbl { font: 10.5px var(--mono); fill: var(--ink); }
  .num { font: 9.5px var(--mono); fill: var(--muted); }
  .x { fill: var(--ink); opacity: 0.35; }
  .delta { fill: #2563eb; }
  .delta.attn { fill: #d97706; }
  .through { fill: var(--accent); }
  .branch { fill: #d97706; }
  .k { padding: 0 3px; border-radius: 3px; color: #fff; }
  .through-k { background: var(--accent); }
  .branch-k { background: #d97706; }
  .note { font-size: 11px; margin: 4px 0 0; }
</style>
