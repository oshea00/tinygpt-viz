<script lang="ts">
  import { app, rawText, selection } from "../../state/app.svelte";

  let timer: ReturnType<typeof setTimeout> | undefined;
  function onInput() {
    app.promptEdited();
    clearTimeout(timer);
    timer = setTimeout(() => app.run(), 30);
  }

  const block = $derived(app.hello?.config.block_size ?? 100);
  // How much of the text is out of the window: everything before the trace's first token.
  const shownChars = $derived(app.trace ? rawText(app.trace.text.join("")).length : 0);
  const outOfView = $derived(app.trace && app.trace.tokens.length >= block ? Math.max(0, app.viewPrompt().length - shownChars) : 0);
</script>

<div class="prompt">
  <textarea rows="2" bind:value={app.prompt} oninput={onInput} spellcheck="false" placeholder="Type a prompt…"></textarea>
  <div class="tokens mono">
    {#each app.trace?.text ?? [] as t, i}
      <button class="tok" class:sel={selection.position === i} title="position {i}" onclick={() => (selection.position = selection.position === i ? -1 : i)}>{t}</button>
    {/each}
  </div>
  <div class="meta muted">
    {app.trace?.tokens.length ?? 0} / {block} positions
    {#if outOfView > 0}· <span class="slide">window slid: the first {outOfView} characters are out of view</span>{/if}
    {#if app.travel !== null}· <span class="slide">time travel: the trace that produced generated token {app.travel + 1}</span>
    {:else if app.trace && app.trace.prompt !== app.prompt}· showing “{app.trace.prompt?.slice(0, 30)}…”{/if}
    · click a token to follow its row through every stage
  </div>
</div>

<style>
  textarea { width: 100%; resize: vertical; font-family: var(--mono); font-size: 14px; }
  .tokens { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 6px; }
  .tok { padding: 0 3px; min-width: 14px; border-radius: 3px; font-size: 12px; line-height: 18px; }
  .tok.sel { outline: 2px solid var(--select); border-color: transparent; }
  .meta { font-size: 11px; margin-top: 4px; }
  .slide { color: var(--accent); }
</style>
