<script lang="ts">
  import { unavailable } from "./lesson";
  import { player } from "./player.svelte";
</script>

<div class="menu-wrap">
  <button class="lessons" onclick={() => (player.menuOpen = !player.menuOpen)}>lessons ▾</button>
  {#if player.menuOpen}
    <div class="menu panel">
      {#each player.lessons as l}
        {@const why = unavailable(l, player.reads, player.training)}
        <button class="item" disabled={!!why} onclick={() => player.start(l)}>
          <span class="n">{l.order}</span>
          <span><b>{l.title}</b><br /><span class="muted">{l.summary}</span>
            {#if why}<br /><span class="needs">{why}</span>{/if}
          </span>
        </button>
      {/each}
    </div>
  {/if}
</div>

<style>
  .menu-wrap { position: relative; }
  .menu { position: absolute; top: 28px; left: 0; z-index: 100; width: 420px; padding: 4px; box-shadow: 0 6px 24px rgb(0 0 0 / 0.18); }
  .item { display: grid; grid-template-columns: 22px 1fr; gap: 6px; width: 100%; text-align: left; border: none; padding: 6px; font-size: 12px; }
  .item:hover { background: var(--accent-soft); }
  .n { font-weight: 700; color: var(--accent); }
  .item:disabled { opacity: 0.55; cursor: default; }
  .item:disabled:hover { background: none; }
  .needs { color: var(--bad); }
</style>
