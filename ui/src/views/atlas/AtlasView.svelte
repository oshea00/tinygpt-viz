<script lang="ts">
  // The whole network as one zoomable bitmap. Zoomed out it shows structure; zoomed in,
  // tile labels and then the numbers themselves appear. Click any pixel to inspect it.
  import { app, selection, settings } from "../../state/app.svelte";
  import { buildLayout, hitTest, renderAtlas, type AtlasMode, type Layout, type Rendered, type Tile } from "../../lib/atlas";
  import { createRenderer, type AtlasRenderer, type View } from "../../lib/glAtlas";
  import { at, fmt } from "../../lib/tensor";

  const LABELS_AT = 1.2;   // zoom (screen px per number) where tile labels appear
  const NUMBERS_AT = 26;   // ... and where each cell's value is printed

  let glCanvas: HTMLCanvasElement;
  let overlay: HTMLCanvasElement;
  let w = $state(800);
  let h = $state(640);
  let renderer = null as AtlasRenderer | null;
  let rendererKind = $state("");
  let view = $state<View>({ scale: 1, tx: 10, ty: 10 });
  let fitted = false;
  let hover = $state.raw<{ tile: Tile; r: number; c: number; sx: number; sy: number } | null>(null);   // raw: tiles are Map keys
  let rendered = null as Rendered | null;

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
    renderer.setImage(rendered.rgba, layout.width, layout.height);
    if (!fitted) {
      fit();
      fitted = true;
    }
    redraw();
  });

  $effect(() => {
    void view; void w; void h; void hover; void selection.position;
    redraw();
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

    ctx.fillStyle = ink;
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.textBaseline = "top";
    for (const c of layout.columns) ctx.fillText(c.title, sx(c.x), sy(0));
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = css.getPropertyValue("--muted") || "#666";
    ctx.fillText("activations for this prompt", sx(0), sy(layout.actTop - 16));

    const visible = (t: Tile) => sx(t.x + t.cols) > 0 && sx(t.x) < w && sy(t.y + Math.max(t.rows, 2)) > 0 && sy(t.y) < h;
    if (scale >= LABELS_AT) {
      ctx.fillStyle = ink;
      ctx.font = "10px ui-monospace, monospace";
      ctx.textBaseline = "bottom";
      for (const t of layout.tiles) if (!t.sub && visible(t)) ctx.fillText(t.label, sx(t.x), sy(t.y) - 1);
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
    if (!wasClick || !layout || !trace) return;
    const { ax, ay } = toAtlas(e);
    const hit = hitTest(layout, ax, ay);
    if (!hit) return;
    // Jump to the inspector at this number: the stage itself, or the stage that uses this weight.
    if (hit.tile.kind === "act") {
      selection.select(hit.tile.name, settings.atlasMode === "grads" ? "backward" : "forward");
      selection.position = hit.r;
      selection.dim = hit.c;
    } else {
      const user = trace.stages.find((s) => s.weights.includes(hit.tile.name));
      if (!user) return;
      selection.select(user.id, "forward");
      selection.dim = hit.c;
    }
    settings.tab = "inspector";
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
  {params.toLocaleString()} weight pixels + {acts.toLocaleString()} activation pixels · scroll to zoom, drag to pan, click to inspect
  {#if rendererKind}· {rendererKind}{/if}
</p>

<div class="stage" bind:clientWidth={w} style:height="{h}px">
  <canvas class="gl" bind:this={glCanvas} style:width="{w}px" style:height="{h}px"></canvas>
  <canvas class="ov" bind:this={overlay} style:width="{w}px" style:height="{h}px"
          onwheel={onWheel} onpointerdown={onDown} onpointermove={onMove} onpointerup={onUp}
          onpointerleave={() => (hover = null)} ondblclick={fit}></canvas>
  {#if hover}
    <div class="tip mono" style:left="{Math.min(hover.sx + 14, w - 220)}px" style:top="{hover.sy + 12}px">
      {hover.tile.kind === "weight" ? "weight" : "activation"} <b>{hover.tile.name}</b><br />
      [{hover.r}, {hover.c}]{hover.tile.kind === "act" && hover.tile.rows === trace?.tokens.length ? ` “${trace?.text[hover.r]}”` : ""}
      = <b>{hoverValue === null ? "—" : fmt(hoverValue, 5)}</b>
      {#if hoverRange}<br /><span class="muted">tile scale ±{fmt(hoverRange.hi, 3)}</span>{/if}
    </div>
  {/if}
</div>

<style>
  .bar { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 12px; }
  .modes, .tools { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
  button.on { background: var(--accent-soft); border-color: var(--accent); }
  .note { font-size: 11px; margin: 6px 0; }
  .stage { position: relative; width: 100%; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--bg); }
  canvas { position: absolute; left: 0; top: 0; display: block; }
  .ov { cursor: crosshair; touch-action: none; }
  .tip {
    position: absolute; pointer-events: none; z-index: 5; background: var(--panel); border: 1px solid var(--line);
    border-radius: 6px; padding: 3px 7px; font-size: 11px; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); white-space: nowrap;
  }
</style>
