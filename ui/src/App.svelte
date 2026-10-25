<script lang="ts">
  import { app, settings } from "./state/app.svelte";
  import LessonMenu from "./lessons/LessonMenu.svelte";
  import LessonPlayer from "./lessons/LessonPlayer.svelte";
  import ConceptsView from "./views/concepts/ConceptsView.svelte";
  import AtlasView from "./views/atlas/AtlasView.svelte";
  import AttentionView from "./views/attention/AttentionView.svelte";
  import InterventionsView from "./views/interventions/InterventionsView.svelte";
  import Inspector from "./views/inspector/Inspector.svelte";
  import LogitLens from "./views/lens/LogitLens.svelte";
  import ResidualView from "./views/residual/ResidualView.svelte";
  import OutputPanel from "./views/output/OutputPanel.svelte";
  import PromptBox from "./views/output/PromptBox.svelte";
  import PipelineMap from "./views/pipeline/PipelineMap.svelte";
  import Stepper from "./views/pipeline/Stepper.svelte";
  import OneStep from "./views/training/OneStep.svelte";
  import TrainingPanel from "./views/training/TrainingPanel.svelte";

  const cfg = $derived(app.hello?.config);
</script>

<div class="app">
  <header>
    <strong>TinyGPT Visualizer</strong>
    <LessonMenu />
    {#if cfg}
      <span class="muted mono">
        {cfg.name} · {cfg.n_layer} layers × {cfg.n_head} heads · d={cfg.d_model} · vocab {cfg.vocab_size} ·
        {app.hello?.params.toLocaleString()} params · {app.status?.device}
      </span>
    {/if}
    {#if app.interventions.length}
      <button class="ivbadge" onclick={() => (settings.tab = "interventions")}>
        {app.interventions.length} intervention{app.interventions.length > 1 ? "s" : ""} active
      </button>
    {/if}
    {#if app.demo}
      <span class="conn on">static demo · inference only</span>
    {:else}
      <span class="conn" class:on={app.connected}>{app.connected ? "connected" : "connecting…"}</span>
    {/if}
    {#if app.error}<span class="error">{app.error}</span>{/if}
  </header>

  <aside class="left panel"><PipelineMap /></aside>

  <main>
    <LessonPlayer />
    <section class="panel pad"><PromptBox /></section>
    <section class="panel pad">
      <Stepper />
    </section>
    <section class="panel pad grow">
      <nav class="tabs">
        <button class:on={settings.tab === "inspector"} onclick={() => (settings.tab = "inspector")}>tensor inspector</button>
        <button class:on={settings.tab === "attention"} onclick={() => (settings.tab = "attention")}>attention</button>
        <button class:on={settings.tab === "residual"} onclick={() => (settings.tab = "residual")}>residual stream</button>
        <button class:on={settings.tab === "lens"} onclick={() => (settings.tab = "lens")}>logit lens</button>
        <button class:on={settings.tab === "atlas"} onclick={() => (settings.tab = "atlas")}>network atlas</button>
        {#if !app.demo}
          <button class:on={settings.tab === "interventions"} onclick={() => (settings.tab = "interventions")}>interventions</button>
          <button class:on={settings.tab === "concepts"} onclick={() => (settings.tab = "concepts")}>concepts</button>
          <button class:on={settings.tab === "onestep"} onclick={() => (settings.tab = "onestep")}>one training step</button>
        {/if}
      </nav>
      {#if settings.tab === "inspector"}<Inspector />{:else if settings.tab === "attention"}<AttentionView />
      {:else if settings.tab === "residual"}{#if app.trace}<ResidualView />{/if}
      {:else if settings.tab === "lens"}{#if app.trace}<LogitLens />{/if}
      {:else if settings.tab === "atlas"}<AtlasView />
      {:else if settings.tab === "concepts"}{#if app.hello}<ConceptsView />{/if}
      {:else if settings.tab === "interventions"}{#if app.trace && app.hello}<InterventionsView />{/if}
      {:else}<OneStep />{/if}
    </section>
  </main>

  <aside class="right">
    <section class="panel pad"><OutputPanel /></section>
    {#if !app.demo}<section class="panel pad"><TrainingPanel /></section>{/if}
  </aside>
</div>

<style>
  .app {
    display: grid; height: 100%; gap: 10px; padding: 10px;
    grid-template-columns: 300px minmax(0, 1fr) 320px;
    grid-template-rows: auto minmax(0, 1fr);
    grid-template-areas: "head head head" "left main right";
  }
  header { grid-area: head; display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }
  header strong { font-size: 15px; }
  .ivbadge { margin-left: auto; background: #fde68a; border-color: #d97706; color: #78350f; font-size: 12px; }
  .ivbadge + .conn { margin-left: 0; }
  .conn { margin-left: auto; font-size: 12px; color: var(--bad); }
  .conn.on { color: var(--ok); }
  .error { color: var(--bad); font-size: 12px; }
  /* Stable gutters: a scrollbar appearing must not change a column's width, or views that
     size themselves from width (heatmaps, atlas) can resize, change height, and loop. */
  .left { grid-area: left; overflow: auto; scrollbar-gutter: stable; padding: 4px 8px 12px; }
  main { grid-area: main; display: flex; flex-direction: column; gap: 10px; min-height: 0; overflow: auto; scrollbar-gutter: stable; }
  .right { grid-area: right; display: flex; flex-direction: column; gap: 10px; overflow: auto; scrollbar-gutter: stable; }
  .pad { padding: 10px 12px; }
  .grow { flex: 1; }
  .tabs { display: flex; gap: 4px; margin-bottom: 10px; }
  .tabs button.on { background: var(--accent-soft); border-color: var(--accent); }
  @media (max-width: 1100px) {
    .app { grid-template-columns: 1fr; grid-template-rows: auto; grid-template-areas: "head" "main" "right" "left"; height: auto; }
  }
</style>
