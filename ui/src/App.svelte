<script lang="ts">
  import { app, settings } from "./state/app.svelte";
  import { theme } from "./state/theme.svelte";
  import HelpPanel from "./help/HelpPanel.svelte";
  import LessonMenu from "./lessons/LessonMenu.svelte";
  import LessonPlayer from "./lessons/LessonPlayer.svelte";
  import BeliefsView from "./views/beliefs/BeliefsView.svelte";
  import ConceptsView from "./views/concepts/ConceptsView.svelte";
  import AtlasView from "./views/atlas/AtlasView.svelte";
  import EmbeddingsView from "./views/embeddings/EmbeddingsView.svelte";
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

  $effect(() => app.wantWeightGrads(settings.tab === "atlas" && settings.atlasMode === "grads"));

  // Glossary links can appear in any formatted text (lessons, help): one handler for all.
  function onDocClick(e: MouseEvent) {
    const a = (e.target as Element | null)?.closest?.("a.gloss");
    if (!a) return;
    e.preventDefault();
    settings.openGlossary(a.getAttribute("data-gloss"));
  }
</script>

<svelte:document onclick={onDocClick} />

<div class="app">
  <header>
    <strong>TinyGPT Visualizer</strong>
    <LessonMenu />
    <button class="helpbtn" class:on={settings.help} onclick={() => (settings.help ? (settings.help = false) : settings.openHelp(null))} title="help for the open tab">help</button>
    <button class="themebtn" onclick={() => theme.toggle()} title="switch to {theme.current === 'dark' ? 'light' : 'dark'} theme"
            aria-label="switch to {theme.current === 'dark' ? 'light' : 'dark'} theme">{theme.current === "dark" ? "☀" : "☾"}</button>
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
    <span class="conn" class:on={app.connected}>{app.connected ? "connected" : "connecting…"}</span>
    {#if app.error}<span class="error">{app.error}</span>{/if}
  </header>

  <aside class="left panel"><PipelineMap /></aside>

  <main>
    <LessonPlayer />
    <HelpPanel />
    <section class="panel pad"><PromptBox /></section>
    <section class="panel pad">
      <Stepper />
    </section>
    <section class="panel pad grow">
      <nav class="tabs">
        <button class:on={settings.tab === "inspector"} onclick={() => (settings.tab = "inspector")}>inspector</button>
        <button class:on={settings.tab === "embeddings"} onclick={() => (settings.tab = "embeddings")}>embeddings</button>
        <button class:on={settings.tab === "attention"} onclick={() => (settings.tab = "attention")}>attention</button>
        <button class:on={settings.tab === "residual"} onclick={() => (settings.tab = "residual")}>residual stream</button>
        <button class:on={settings.tab === "lens"} onclick={() => (settings.tab = "lens")}>logit lens</button>
        <button class:on={settings.tab === "atlas"} onclick={() => (settings.tab = "atlas")}>network atlas</button>
        <button class:on={settings.tab === "interventions"} onclick={() => (settings.tab = "interventions")}>interventions</button>
        <button class:on={settings.tab === "concepts"} onclick={() => (settings.tab = "concepts")}>concepts</button>
        <button class:on={settings.tab === "beliefs"} onclick={() => (settings.tab = "beliefs")}>beliefs</button>
        {#if app.hello?.training}
          <button class:on={settings.tab === "onestep"} onclick={() => (settings.tab = "onestep")}>training step</button>
        {/if}
      </nav>
      {#if settings.tab === "inspector"}<Inspector />{:else if settings.tab === "attention"}<AttentionView />
      {:else if settings.tab === "embeddings"}{#if app.trace}<EmbeddingsView />{/if}
      {:else if settings.tab === "residual"}{#if app.trace}<ResidualView />{/if}
      {:else if settings.tab === "lens"}{#if app.trace}<LogitLens />{/if}
      {:else if settings.tab === "atlas"}<AtlasView />
      {:else if settings.tab === "concepts"}{#if app.hello}<ConceptsView />{/if}
      {:else if settings.tab === "beliefs"}{#if app.hello}<BeliefsView />{/if}
      {:else if settings.tab === "interventions"}{#if app.trace && app.hello}<InterventionsView />{/if}
      {:else}<OneStep />{/if}
    </section>
  </main>

  <aside class="right">
    <section class="panel pad"><OutputPanel /></section>
    <!-- Training is opt-in: the server is started with --train. -->
    {#if app.hello?.training}<section class="panel pad"><TrainingPanel /></section>{/if}
  </aside>
</div>

<style>
  .app {
    display: grid; height: 100%; gap: 10px; padding: 10px;
    grid-template-columns: 375px minmax(0, 1fr) 320px;
    grid-template-rows: auto minmax(0, 1fr);
    grid-template-areas: "head head head" "left main right";
  }
  header { grid-area: head; display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }
  header strong { font-size: 15px; }
  .helpbtn.on { background: var(--accent-soft); border-color: var(--accent); }
  .themebtn { padding: 3px 7px; line-height: 1; }
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
  /* A tab strip: the rule runs the panel's full width, inactive tabs sit recessed behind it,
     and the active tab is raised and joined to the content below by covering the rule. */
  .tabs {
    display: flex; gap: 2px; align-items: flex-end;
    overflow-x: auto; scrollbar-width: thin;   /* one row; scrolls sideways on narrow windows */
    margin: -10px -12px 12px; padding: 8px 12px 0;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  .tabs button {
    border: 1px solid transparent; border-bottom: none; border-radius: 7px 7px 0 0;
    background: var(--bg); color: var(--muted); padding: 5px 10px 6px; white-space: nowrap; flex: none;
  }
  .tabs button:hover { color: var(--ink); border-color: var(--line); }
  .tabs button.on {
    background: var(--panel); color: var(--ink); font-weight: 600;
    border-color: var(--line); box-shadow: inset 0 2px 0 var(--accent);
    position: relative; z-index: 1;
  }
  @media (max-width: 1100px) {
    .app { grid-template-columns: 1fr; grid-template-rows: auto; grid-template-areas: "head" "main" "right" "left"; height: auto; }
  }
</style>
