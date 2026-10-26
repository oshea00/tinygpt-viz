<script lang="ts">
  // The whole network as one zoomable bitmap. Zoomed out it shows structure; zoomed in,
  // tile labels and then the numbers themselves appear. Click a tile to select its stage
  // (the pipeline map follows); double-click to open it in the inspector.
  import { app, selection, settings } from "../../state/app.svelte";
  import { buildLayout, hitTest, LABEL, renderAtlas, SECTION_GAP, tilesForStage, TITLE, type AtlasMode, type Layout, type Rendered, type Tile } from "../../lib/atlas";
  import { createRenderer, type AtlasRenderer, type View } from "../../lib/glAtlas";
  import { REUSED } from "../../lib/heatmap";
  import { at, fmt } from "../../lib/tensor";

  const SELECTED = "#fbbf24";   // light gold: the selected stage's outline
  const MIN_FONT = 6.5;    // labels smaller than this (screen px) are not drawn
  const NUMBERS_AT = 26;   // zoom (screen px per number) where each cell's value is printed

  let glCanvas: HTMLCanvasElement;
  let overlay: HTMLCanvasElement;
  let w = $state(800);
  let h = $state(640);
  let renderer = null as AtlasRenderer | null;
  let rendererKind = $state("");
  // The view is kept in shared settings, so switching tabs and back returns to the same place.
  let view = $state<View>(settings.atlasView ?? { scale: 1, tx: 10, ty: 10 });
  let fitted = settings.atlasView !== null;
  $effect(() => {
    settings.atlasView = view;
  });
  let hover = $state.raw<{ tile: Tile; r: number; c: number; sx: number; sy: number } | null>(null);   // raw: tiles are Map keys
  let rendered = null as Rendered | null;
  let tipW = $state(200);
  let tipH = $state(50);

  const cfg = $derived(app.hello?.config);
  const trace = $derived(app.trace);
  const weights = $derived(app.weights);

  // Rebuilt only when shapes change (prompt length, model), not on every trace.
  let layout = $state.raw<Layout | null>(null);
  let layoutKey = "";
  $effect(() => {
    if (!trace || !weights || !cfg) return;
    const key = `${cfg.name}:${trace.tokens.length}:${Object.keys(weights.tensors).length}`;
    if (key !== layoutKey) {
      layoutKey = key;
      layout = buildLayout(trace, weights, cfg.n_layer, cfg.n_head);
    }
  });

  const modes: [AtlasMode, string, string][] = [
    ["values", "values", "weights and this trace's activations"],
    ["grads", "gradients", "∂L/∂W for weights, ∂L/∂x for activations"],
    ["flow", "weight × input", "x[i]·W[i,j] for the selected position: what flows through each connection"],
    ["diff", "diff vs pinned", "change since the pinned snapshot (pin, then edit the prompt or train)"],
    ["recent", "recent change", "weights: change since the previous weights snapshot during training"],
  ];

  // Re-render the bitmap when anything it shows changes.
  $effect(() => {
    if (!layout || !trace || !weights || !glCanvas) return;
    if (!renderer) {
      renderer = createRenderer(glCanvas);
      rendererKind = renderer.kind;
    }
    rendered = renderAtlas(layout, {
      trace, weights, mode: settings.atlasMode, position: selection.query(trace),
      pinnedTrace: app.pinned?.trace, pinnedWeights: app.pinned?.weights, prevWeights: app.prevWeights,
    });
    renderer.setTiles(rendered.images.map((im) => ({ x: im.tile.x, y: im.tile.y, width: im.width, height: im.height, rgba: im.rgba })));
    if (!fitted) {
      fit();
      fitted = true;
    }
    redraw();
  });

  $effect(() => {
    void view; void w; void h; void hover; void selection.position; void selection.stageId;
    redraw();
  });

  // The selected stage's tiles; when the selection changes (here, in the pipeline map or the
  // stepper) and its tile is entirely off-screen, pan to it, keeping the zoom.
  const selected = $derived(layout && trace ? tilesForStage(layout, trace, selection.stageId) : { main: [], weights: [] });
  let shownFor = "";
  $effect(() => {
    const t = selected.main[0];
    if (!t || !layout || selection.stageId === shownFor) return;
    shownFor = selection.stageId;
    const { scale, tx, ty } = view;
    const x0 = tx + t.x * scale, y0 = ty + t.y * scale, x1 = x0 + t.cols * scale, y1 = y0 + Math.max(t.rows, 2) * scale;
    if (x1 < 0 || x0 > w || y1 < 0 || y0 > h) {
      view = { scale, tx: w / 2 - (t.x + t.cols / 2) * scale, ty: h / 2 - (t.y + Math.max(t.rows, 2) / 2) * scale };
    }
  });

  function fit() {
    if (!layout) return;
    const scale = Math.min((w - 20) / layout.width, (h - 20) / layout.height);
    view = { scale, tx: (w - layout.width * scale) / 2, ty: 10 };
  }

  function redraw() {
    if (!renderer || !layout) return;
    renderer.draw(view, w, h);
    drawOverlay();
  }

  function drawOverlay() {
    if (!overlay || !layout || !trace) return;
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.round(w * dpr);
    overlay.height = Math.round(h * dpr);
    const ctx = overlay.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(overlay);
    const ink = css.getPropertyValue("--ink") || "#000";
    const sel = css.getPropertyValue("--select") || "#000";
    const { scale, tx, ty } = view;
    const sx = (x: number) => tx + x * scale;
    const sy = (y: number) => ty + y * scale;

    // Every label is sized to the space the layout reserves for it at this zoom, and trimmed
    // to its width, so labels never cover tiles or each other.
    const fit = (text: string, maxW: number) => {
      if (ctx.measureText(text).width <= maxW) return text;
      for (let n = text.length - 1; n > 0; n--) {
        const t = text.slice(0, n) + "…";
        if (ctx.measureText(t).width <= maxW) return t;
      }
      return "";
    };
    ctx.textBaseline = "bottom";
    const titleFont = Math.min(13, TITLE * scale * 0.75);
    if (titleFont >= MIN_FONT) {
      ctx.fillStyle = ink;
      ctx.font = `600 ${titleFont}px system-ui, sans-serif`;
      for (const c of layout.columns) ctx.fillText(fit(c.title, c.width * scale), sx(c.x), sy(TITLE) - 1);
    }
    const sectionFont = Math.min(11, SECTION_GAP * scale * 0.4);
    if (sectionFont >= MIN_FONT) {
      ctx.fillStyle = css.getPropertyValue("--muted") || "#666";
      ctx.font = `${sectionFont}px system-ui, sans-serif`;
      for (const c of layout.columns) ctx.fillText(fit("activations", c.width * scale), sx(c.x), sy(c.actTop) - 1);
    }

    const visible = (t: Tile) => sx(t.x + t.cols) > 0 && sx(t.x) < w && sy(t.y + Math.max(t.rows, 2)) > 0 && sy(t.y) < h;
    const labelFont = Math.min(10, LABEL * scale * 0.8);
    if (labelFont >= MIN_FONT) {
      ctx.fillStyle = ink;
      ctx.font = `${labelFont}px ui-monospace, monospace`;
      for (const t of layout.tiles) if (!t.sub && visible(t)) ctx.fillText(fit(t.label, t.cols * scale + 4), sx(t.x), sy(t.y) - 1);
    }

    // The selected stage: a bright gold outline on its tile, dashed on the weights it uses,
    // each over a thin dark edge so it stands out on pale tiles and the light background alike.
    const outline = (t: Tile, width: number, dash: number[]) => {
      const rect = [sx(t.x) - 2, sy(t.y) - 2, t.cols * scale + 4, Math.max(t.rows, 2) * scale + 4] as const;
      ctx.setLineDash(dash);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
      ctx.lineWidth = width + 2;
      ctx.strokeRect(...rect);
      ctx.strokeStyle = SELECTED;
      ctx.lineWidth = width;
      ctx.strokeRect(...rect);
    };
    for (const t of selected.main) outline(t, 3, []);
    for (const t of selected.weights) outline(t, 2, [5, 3]);
    ctx.setLineDash([]);

    // KV cache: rows that came from the cache are veiled in every activation tile.
    const reused = settings.atlasMode === "values" || settings.atlasMode === "flow" ? trace.incremental?.reused ?? 0 : 0;
    if (reused > 0) {
      ctx.fillStyle = REUSED;
      for (const t of layout.tiles) {
        if (t.kind === "act" && t.rows === trace.tokens.length && visible(t)) ctx.fillRect(sx(t.x), sy(t.y), t.cols * scale, reused * scale);
      }
    }

    // One token, one path: the selected position's row in every activation tile.
    const p = selection.query(trace);
    ctx.strokeStyle = sel;
    ctx.lineWidth = 1;
    for (const t of layout.tiles) {
      if (t.kind === "act" && t.rows === trace.tokens.length && visible(t)) {
        ctx.strokeRect(sx(t.x) - 0.5, sy(t.y + p) - 0.5, t.cols * scale + 1, Math.max(1, scale) + 1);
      }
    }

    if (scale >= NUMBERS_AT && rendered) {
      ctx.font = `${Math.min(12, scale / 3.2)}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      let drawn = 0;
      for (const t of layout.tiles) {
        const tensor = rendered.tensors.get(t);
        if (!tensor || !visible(t)) continue;
        const c0 = Math.max(0, Math.floor(-tx / scale - t.x)), c1 = Math.min(t.cols, Math.ceil((w - tx) / scale - t.x));
        const r0 = Math.max(0, Math.floor(-ty / scale - t.y)), r1 = Math.min(t.rows, Math.ceil((h - ty) / scale - t.y));
        for (let r = r0; r < r1 && drawn < 4000; r++) for (let c = c0; c < c1 && drawn < 4000; c++, drawn++) {
          const v = at(tensor, r, c);
          ctx.fillStyle = Math.abs(rendered.ranges.get(t) ? v / rendered.ranges.get(t)!.hi : 0) > 0.6 ? "#fff" : "#000";
          ctx.fillText(fmt(v, 2), sx(t.x + c + 0.5), sy(t.y + r + 0.5));
        }
      }
      ctx.textAlign = "start";
    }

    if (hover) {
      const t = hover.tile;
      ctx.strokeStyle = sel;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx(t.x) - 1, sy(t.y) - 1, t.cols * scale + 2, Math.max(t.rows, 2) * scale + 2);
      if (scale >= 4) ctx.strokeRect(sx(t.x + hover.c), sy(t.y + hover.r), scale, scale);
    }
  }

  // -- interaction ----------------------------------------------------------

  let drag: { x: number; y: number; tx: number; ty: number; moved: boolean } | null = null;

  function toAtlas(e: MouseEvent) {
    const rect = overlay.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    return { mx, my, ax: (mx - view.tx) / view.scale, ay: (my - view.ty) / view.scale };
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const { mx, my, ax, ay } = toAtlas(e);
    const scale = Math.max(0.05, Math.min(80, view.scale * Math.exp(-e.deltaY * 0.0015)));
    view = { scale, tx: mx - ax * scale, ty: my - ay * scale };
  }

  function onDown(e: PointerEvent) {
    drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
    overlay.setPointerCapture(e.pointerId);
  }

  function onMove(e: PointerEvent) {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) view = { ...view, tx: drag.tx + dx, ty: drag.ty + dy };
      return;
    }
    if (!layout) return;
    const { mx, my, ax, ay } = toAtlas(e);
    const hit = hitTest(layout, ax, ay);
    hover = hit ? { ...hit, sx: mx, sy: my } : null;
  }

  function onUp(e: PointerEvent) {
    const wasClick = drag && !drag.moved;
    drag = null;
    if (wasClick) selectAt(e);
  }

  /** Select the stage under the pointer (the stage itself, or the one that uses this weight)
   * without leaving the atlas; the pipeline map and other views follow the selection. */
  function selectAt(e: MouseEvent): boolean {
    if (!layout || !trace) return false;
    const { ax, ay } = toAtlas(e);
    const hit = hitTest(layout, ax, ay);
    if (!hit) return false;
    shownFor = hit.tile.kind === "act" ? hit.tile.name : shownFor;   // already on screen: no need to pan
    if (hit.tile.kind === "act") {
      selection.select(hit.tile.name, settings.atlasMode === "grads" ? "backward" : "forward");
      selection.position = hit.r;
      selection.dim = hit.c;
    } else {
      const user = trace.stages.find((s) => s.weights.includes(hit.tile.name));
      if (!user) return false;
      shownFor = user.id;
      selection.select(user.id, "forward");
      selection.dim = hit.c;
    }
    return true;
  }

  function onDouble(e: MouseEvent) {
    if (selectAt(e)) settings.tab = "inspector";
  }

  const hoverValue = $derived.by(() => {
    if (!hover || !rendered) return null;
    const t = rendered.tensors.get(hover.tile);
    return t ? at(t, hover.r, hover.c) : null;
  });
  const hoverRange = $derived(hover && rendered ? rendered.ranges.get(hover.tile) : null);
  const params = $derived(layout ? layout.tiles.filter((t) => t.kind === "weight").reduce((n, t) => n + t.rows * t.cols, 0) : 0);
  const acts = $derived(layout ? layout.tiles.filter((t) => t.kind === "act").reduce((n, t) => n + t.rows * t.cols, 0) : 0);
</script>

<div class="bar">
  <span class="modes">
    {#each modes as [m, label, tip]}
      <button class:on={settings.atlasMode === m} title={tip} onclick={() => (settings.atlasMode = m)}
              disabled={(m === "diff" && !app.pinned) || (m === "recent" && !app.prevWeights)}>{label}</button>
    {/each}
  </span>
  <span class="tools">
    {#if app.pinned}
      <span class="muted">pinned: step {app.pinned.trace.step}, “{app.pinned.trace.prompt?.slice(0, 18)}…”</span>
      <button onclick={() => (app.pinned = null)}>unpin</button>
    {/if}
    <button onclick={() => trace && weights && (app.pinned = { trace, weights })}>pin this as reference</button>
    <button onclick={fit}>fit</button>
  </span>
</div>
<p class="muted note">
  {modes.find((m) => m[0] === settings.atlasMode)?.[2]} ·
  {params.toLocaleString()} weight pixels + {acts.toLocaleString()} activation pixels · scroll to zoom, drag to pan,
  click to select, double-click to open in the inspector
  {#if rendererKind}· {rendererKind}{/if}
</p>

<div class="stage" bind:clientWidth={w} style:height="{h}px">
  <canvas class="gl" bind:this={glCanvas} style:width="{w}px" style:height="{h}px"></canvas>
  <canvas class="ov" bind:this={overlay} style:width="{w}px" style:height="{h}px"
          onwheel={onWheel} onpointerdown={onDown} onpointermove={onMove} onpointerup={onUp}
          onpointerleave={() => (hover = null)} ondblclick={onDouble}></canvas>
  {#if hover}
    <!-- The stage clips its contents, so flip the tooltip to the other side of the cursor
         wherever it would not fit. -->
    <div class="tip mono" bind:offsetWidth={tipW} bind:offsetHeight={tipH}
         style:left="{hover.sx + 14 + tipW > w ? Math.max(0, hover.sx - 14 - tipW) : hover.sx + 14}px"
         style:top="{hover.sy + 12 + tipH > h ? Math.max(0, hover.sy - 12 - tipH) : hover.sy + 12}px">
      {hover.tile.kind === "weight" ? "weight" : "activation"} <b>{hover.tile.name}</b><br />
      [{hover.r}, {hover.c}]{hover.tile.kind === "act" && hover.tile.rows === trace?.tokens.length ? ` “${trace?.text[hover.r]}”` : ""}
      = <b>{hoverValue === null ? "—" : fmt(hoverValue, 5)}</b>
      {#if hoverRange}<br /><span class="muted">tile scale ±{fmt(hoverRange.hi, 3)}</span>{/if}
    </div>
  {/if}
</div>

<!-- Legend: what each mark on the atlas means, and what is selected right now. -->
<div class="legend">
  <div class="item">
    <svg width="26" height="14"><rect x="2.5" y="2.5" width="21" height="9" fill="none" stroke="rgba(0,0,0,0.6)" stroke-width="5" /><rect x="2.5" y="2.5" width="21" height="9" fill="none" stroke={SELECTED} stroke-width="3" /></svg>
    <span><b>selected stage</b> <span class="mono">{selection.stageId}</span><br /><span class="muted">click a tile, or pick it in the pipeline</span></span>
  </div>
  <div class="item">
    <svg width="26" height="14"><rect x="2.5" y="2.5" width="21" height="9" fill="none" stroke="rgba(0,0,0,0.6)" stroke-width="4" stroke-dasharray="5 3" /><rect x="2.5" y="2.5" width="21" height="9" fill="none" stroke={SELECTED} stroke-width="2" stroke-dasharray="5 3" /></svg>
    <span><b>its weights</b>
      {#if selected.weights.length}<span class="mono">{selected.weights.map((t) => t.name).join(", ")}</span>{:else}<span class="muted">none for this stage</span>{/if}<br />
      <span class="muted">the matrices and vectors it multiplies by</span></span>
  </div>
  <div class="item">
    <svg width="26" height="14"><rect x="1" y="1" width="24" height="12" fill="#f4a582" /><rect x="1" y="5.5" width="24" height="3" fill="none" stroke="var(--select)" stroke-width="1" /></svg>
    <span><b>selected position</b>{#if trace}&nbsp;row {selection.query(trace)} “{trace.text[selection.query(trace)]}”{/if}<br /><span class="muted">its row in every activation tile</span></span>
  </div>
  <div class="item">
    <svg width="26" height="14"><rect x="2" y="2" width="22" height="10" fill="none" stroke="var(--select)" stroke-width="1.5" /></svg>
    <span><b>under the pointer</b><br /><span class="muted">its value is in the tooltip</span></span>
  </div>
  <div class="item">
    <svg width="26" height="14"><defs><pattern id="atlas-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="rgb(200,200,200)" /><line x1="0" y1="0" x2="0" y2="5" stroke="rgba(0,0,0,0.25)" stroke-width="2" /></pattern></defs><rect x="1" y="1" width="24" height="12" fill="url(#atlas-hatch)" /></svg>
    <span><b>masked or empty</b><br /><span class="muted">future positions in attention, or nothing to show in this mode</span></span>
  </div>
  <div class="item">
    <span class="ramp div"></span>
    <span><b>values</b> <span class="muted">negative · 0 · positive</span><br /><span class="muted">each tile on its own scale (see tooltip)</span></span>
  </div>
  <div class="item">
    <span class="ramp seq"></span>
    <span><b>probabilities</b><br /><span class="muted">attention weights and next-token probabilities</span></span>
  </div>
</div>

<style>
  .bar { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 12px; }
  .modes, .tools { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
  button.on { background: var(--accent-soft); border-color: var(--accent); }
  .note { font-size: 11px; margin: 6px 0; }
  .stage { position: relative; width: 100%; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--bg); }
  canvas { position: absolute; left: 0; top: 0; display: block; }
  .ov { cursor: crosshair; touch-action: none; }
  .legend {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 6px 14px;
    margin-top: 8px; padding: 8px 10px; font-size: 11px; border: 1px solid var(--line); border-radius: 8px;
  }
  .item { display: grid; grid-template-columns: 28px 1fr; gap: 6px; align-items: center; line-height: 1.35; }
  .item svg { display: block; }
  .ramp { display: block; width: 26px; height: 10px; border-radius: 2px; border: 1px solid var(--line); }
  .ramp.div { background: linear-gradient(90deg, #2166ac, #f7f7f7, #b2182b); }
  .ramp.seq { background: linear-gradient(90deg, #fcfbfd, #3f007d); }
  .tip {
    position: absolute; pointer-events: none; z-index: 5; background: var(--panel); border: 1px solid var(--line);
    border-radius: 6px; padding: 3px 7px; font-size: 11px; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); white-space: nowrap;
  }
</style>
