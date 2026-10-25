<script lang="ts">
  import { app, rawChar, selection } from "../../state/app.svelte";

  let timer: ReturnType<typeof setTimeout> | undefined;
  function onInput() {
    app.promptEdited();
    clearTimeout(timer);
    timer = setTimeout(() => app.run(), 30);
  }

  const known = $derived(new Set((app.hello?.vocab ?? []).map(rawChar)));
  const dropped = $derived(known.size ? [...new Set([...app.prompt].filter((c) => !known.has(c)))] : []);
  const total = $derived(known.size ? [...app.viewPrompt()].filter((c) => known.has(c)).length : 0);
  const block = $derived(app.hello?.config.block_size ?? 100);
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
    {#if total > block}· <span class="slide">window slid: the first {total - block} characters are out of view</span>{/if}
    {#if app.travel !== null}· <span class="slide">time travel: the trace that produced generated character {app.travel + 1}</span>
    {:else if app.trace && app.trace.prompt !== app.prompt}· showing “{app.trace.prompt?.slice(0, 30)}…”{/if}
    {#if dropped.length}· <span class="warn">not in vocabulary, dropped: {dropped.map((c) => JSON.stringify(c)).join(" ")}</span>{/if}
    · click a token to follow its row through every stage
  </div>
</div>

<style>
  textarea { width: 100%; resize: vertical; font-family: var(--mono); font-size: 14px; }
  .tokens { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 6px; }
  .tok { padding: 0 3px; min-width: 14px; border-radius: 3px; font-size: 12px; line-height: 18px; }
  .tok.sel { outline: 2px solid var(--select); border-color: transparent; }
  .meta { font-size: 11px; margin-top: 4px; }
  .warn { color: var(--bad); }
  .slide { color: var(--accent); }
</style>
