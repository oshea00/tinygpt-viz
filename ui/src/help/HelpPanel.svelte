<script lang="ts">
  // Help for the open tab, or for one of the always-visible panels.
  import { app, settings } from "../state/app.svelte";
  import { formatText } from "../lessons/lesson";
  import { LAYOUT, PANELS, TOPICS, type HelpTopic } from "./help";

  const tabTopic = $derived(TOPICS[settings.tab] ?? LAYOUT);
  const topic = $derived(settings.helpTopic === "layout" ? LAYOUT : settings.helpTopic ? PANELS[settings.helpTopic] : tabTopic);
  const choices = $derived([
    [null, `${tabTopic.title} (open tab)`],
    ["layout", LAYOUT.title],
    ["pipeline", PANELS.pipeline.title],
    ["generation", PANELS.generation.title],
    ...(app.hello?.training ? [["training", PANELS.training.title] as const] : []),
  ] as const);
</script>

{#if settings.help}
  <section class="help panel">
    <div class="top">
      <b>Help</b>
      <nav>
        {#each choices as [key, label]}
          <button class:on={settings.helpTopic === key} onclick={() => (settings.helpTopic = key)}>{label}</button>
        {/each}
      </nav>
      <button class="close" title="close help" onclick={() => (settings.help = false)}>×</button>
    </div>
    <h3>{topic.title}</h3>
    <p class="what">{@html formatText(topic.what)}</p>
    <ul>{#each topic.how as line}<li>{@html formatText(line)}</li>{/each}</ul>
    {#if topic.tryThis.length}
      <p class="try"><b>Try:</b> {@html formatText(topic.tryThis.join(" "))}</p>
    {/if}
    <p class="more muted">More in the guide: <a href="/docs/guide/{topic.guide}" target="_blank" rel="noreferrer">docs/guide/{topic.guide}</a></p>
  </section>
{/if}

<style>
  .help { padding: 10px 14px; border-color: var(--accent); font-size: 13px; line-height: 1.5; }
  .top { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  nav { display: flex; gap: 4px; flex-wrap: wrap; }
  nav button { font-size: 11.5px; padding: 1px 8px; border-radius: 12px; color: var(--muted); }
  nav button.on { color: var(--ink); background: var(--accent-soft); border-color: var(--accent); }
  .close { margin-left: auto; padding: 0 7px; }
  h3 { font-size: 14px; margin: 8px 0 2px; }
  .what { margin: 4px 0 6px; max-width: 920px; }
  ul { margin: 4px 0; padding-left: 20px; max-width: 920px; }
  li { margin: 2px 0; }
  .try { margin: 6px 0 2px; }
  .more { font-size: 11px; margin: 4px 0 0; }
  :global(.help code) { font-family: var(--mono); background: var(--accent-soft); padding: 0 3px; border-radius: 3px; }
</style>
