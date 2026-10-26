<script lang="ts">
  // Every head's pattern for the current prompt, so recognisable ones can be spotted and named.
  import type { Tensor } from "../../types";
  import { app, selection } from "../../state/app.svelte";
  import { headPattern, headStats } from "../../lib/attention";
  import { headColor } from "../../lib/colormap";
  import { drawThumb } from "../../lib/heatmap";

  const cfg = $derived(app.hello!.config);
  const trace = $derived(app.trace!);
  const grid = $derived(
    Array.from({ length: cfg.n_layer }, (_, l) =>
      Array.from({ length: cfg.n_head }, (_, h) => {
        const w = trace.byId.get(`L${l}.attn.H${h}.weights`)!.output;
        const stats = headStats(w);
        return { l, h, w, pattern: headPattern(stats), focus: stats.peak };
      }),
    ),
  );

  // Names are a per-viewer convenience, so browser storage is enough; it may be unavailable.
  const storeKey = $derived(`tinygpt.headNames.${cfg.name}`);
  let names = $state<Record<string, string>>({});
  $effect(() => {
    try {
      names = JSON.parse(localStorage.getItem(storeKey) ?? "{}");
    } catch {
      names = {};
    }
  });
  function rename(key: string, value: string) {
    names = { ...names, [key]: value.trim() };
    try {
      localStorage.setItem(storeKey, JSON.stringify(names));
    } catch {}
  }

  function thumb(canvas: HTMLCanvasElement, w: Tensor) {
    const draw = (t: Tensor) => drawThumb(canvas, t, { kind: "sequential", lo: 0, hi: 1, symlog: false, label: "" }, true);
    draw(w);
    return { update: draw };
  }
</script>

<div class="gallery" style:grid-template-columns="auto repeat({cfg.n_head}, minmax(0, 1fr))">
  <span></span>
  {#each grid[0] as c}<span class="colhead" style:color={headColor(c.h)}>head {c.h}</span>{/each}
  {#each grid as layerRow, l}
    <span class="rowhead">layer {l}</span>
    {#each layerRow as c (c.h)}
      {@const key = `L${c.l}.H${c.h}`}
      {@const selected = selection.layer === c.l && selection.head === c.h}
      {@const ablated = app.interventions.some((iv) => iv.kind === "head" && iv.layer === c.l && iv.head === c.h)}
      <div class="cell" class:selected style:border-color={headColor(c.h)}>
        <button class="pic" onclick={() => selection.selectHead(c.l, c.h)} title="L{c.l} head {c.h}">
          <canvas use:thumb={c.w}></canvas>
        </button>
        <div class="badge mono">{c.pattern}</div>
        <div class="focus mono" title="focus: the average weight each row gives its heaviest key (1 = always one key)">
          focus {c.focus.toFixed(2)}<span class="track"><span class="fill" style:width="{c.focus * 100}%"></span></span>
        </div>
        <button class="ablate" class:on={ablated} title="zero this head's output in every trace"
                onclick={() => ablated
                  ? app.setInterventions(app.interventions.filter((iv) => !(iv.kind === "head" && iv.layer === c.l && iv.head === c.h)))
                  : app.addIntervention({ kind: "head", layer: c.l, head: c.h, mode: "zero" })}>
          {ablated ? "restore" : "ablate"}
        </button>
        <input class="name" placeholder="name it…" value={names[key] ?? ""}
               onchange={(e) => rename(key, (e.target as HTMLInputElement).value)} />
      </div>
    {/each}
  {/each}
</div>
<p class="muted note">Badges: a fixed habit (mean weight on the previous / same / first position) if the head has one;
  otherwise “sharp” (most weight on one key, chosen by content), “local” or “broad”, and how many positions back it
  looks on average. <b>Focus</b>: the average weight each row gives its heaviest key, whatever the badge; 1 means every
  query looks at a single key. Measured on this prompt only, so try a few.</p>

<style>
  .gallery { display: grid; gap: 8px; align-items: center; }
  .colhead { font-size: 11px; font-weight: 600; text-align: center; }
  .rowhead { font-size: 11px; color: var(--muted); writing-mode: vertical-rl; transform: rotate(180deg); }
  .cell { border: 2px solid; border-radius: 6px; padding: 4px; background: var(--panel); max-width: 170px; }
  .cell.selected { box-shadow: 0 0 0 2.5px var(--select); }
  .pic { padding: 0; border: none; width: 100%; aspect-ratio: 1; display: block; background: none; }
  canvas { width: 100%; height: 100%; image-rendering: pixelated; display: block; }
  .badge { font-size: 10.5px; text-align: center; margin-top: 2px; }
  .focus { font-size: 9.5px; color: var(--muted); text-align: center; white-space: nowrap; }
  .focus .track { display: block; height: 3px; margin: 1px 0 2px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .focus .fill { display: block; height: 100%; background: var(--accent); }
  .ablate { width: 100%; font-size: 10.5px; padding: 0 4px; margin-top: 2px; }
  .ablate.on { background: #fde68a; border-color: #d97706; color: #78350f; }
  .name { width: 100%; font-size: 11px; padding: 1px 4px; margin-top: 2px; }
  .note { font-size: 11px; margin: 6px 0 0; }
</style>
