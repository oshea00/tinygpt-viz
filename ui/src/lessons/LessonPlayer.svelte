<script lang="ts">
  import { player } from "./player.svelte";
  import { formatText } from "./lesson";

  const lesson = $derived(player.lesson);
  const step = $derived(player.step);
  const last = $derived(lesson ? player.index === player.steps.length - 1 : false);
  let showHint = $state(false);
  $effect(() => {
    void player.index;
    showHint = false;
  });
</script>

{#if lesson && step}
  <section class="player panel">
    <div class="top">
      <span class="title">{lesson.title}</span>
      <span class="muted">step {player.index + 1} of {player.steps.length}</span>
      <button class="close" title="close the lesson" onclick={() => player.stop()}>×</button>
    </div>
    <div class="progress"><span style:width="{((player.index + 1) / player.steps.length) * 100}%"></span></div>
    <p class="text">{@html formatText(step.text, player.vars)}</p>
    {#if step.ask}
      <div class="ask" class:done={player.answered}>
        <b>{player.answered ? "✓" : "?"}</b>
        <span>{@html formatText(player.answered ? (step.ask.correct ?? "That's it.") : step.ask.question, player.vars)}</span>
        {#if !player.answered && step.ask.check.kind === "reveal"}
          <button class="link" onclick={() => (player.revealed = true)}>show the answer</button>
        {/if}
        {#if !player.answered && step.ask.hint}
          {#if showHint}<span class="hint">{@html formatText(step.ask.hint, player.vars)}</span>
          {:else}<button class="link" onclick={() => (showHint = true)}>hint</button>{/if}
        {/if}
      </div>
    {/if}
    <div class="nav">
      <button onclick={() => player.go(player.index - 1)} disabled={player.index === 0}>back</button>
      {#if last}
        <button class="primary" onclick={() => player.stop()}>finish</button>
      {:else}
        <button class="primary" onclick={() => player.go(player.index + 1)}>{step.ask && !player.answered ? "skip" : "next"}</button>
      {/if}
      <button class="link" onclick={() => player.reset()}>reset this step</button>
    </div>
  </section>
{/if}

<style>
  .player { padding: 10px 12px; border-color: var(--accent); border-width: 1.5px; }
  .top { display: flex; gap: 10px; align-items: baseline; }
  .title { font-weight: 600; }
  .close { margin-left: auto; padding: 0 7px; }
  .progress { height: 3px; background: var(--line); border-radius: 2px; margin: 6px 0; overflow: hidden; }
  .progress span { display: block; height: 100%; background: var(--accent); }
  .text { margin: 4px 0 8px; max-width: 900px; font-size: 13.5px; line-height: 1.5; }
  .text :global(code), .ask :global(code) { font-family: var(--mono); background: var(--accent-soft); padding: 0 3px; border-radius: 3px; }
  /* The question box is always light, in either theme: code there takes the box's own text
     colour on a faint darkening of its background, not the theme's accent tint. */
  .ask :global(code) { background: rgb(0 0 0 / 0.08); color: inherit; font-weight: 600; }
  .ask { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; padding: 5px 8px; border-radius: 6px; background: #fef3c7; color: #78350f; margin-bottom: 8px; }
  .ask.done { background: #d1fae5; color: #065f46; }
  .hint { font-style: italic; }
  .nav { display: flex; gap: 6px; align-items: center; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 12px; }
  .ask .link { color: inherit; }
</style>
