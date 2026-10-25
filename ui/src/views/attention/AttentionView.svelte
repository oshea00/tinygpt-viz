<script lang="ts">
  import { app, selection } from "../../state/app.svelte";
  import { headColor } from "../../lib/colormap";
  import Arcs from "./Arcs.svelte";
  import AttnMatrix from "./AttnMatrix.svelte";
  import HeadGallery from "./HeadGallery.svelte";
  import QKExplorer from "./QKExplorer.svelte";

  const cfg = $derived(app.hello?.config);
</script>

{#if !app.trace || !cfg}
  <p class="muted">Waiting for a trace…</p>
{:else}
  <div class="pick">
    <label>layer
      <select value={selection.layer} onchange={(e) => selection.selectHead(+(e.target as HTMLSelectElement).value, selection.head)}>
        {#each Array.from({ length: cfg.n_layer }, (_, l) => l) as l}<option value={l}>{l}</option>{/each}
      </select>
    </label>
    <span class="heads">head
      {#each Array.from({ length: cfg.n_head }, (_, h) => h) as h}
        <button class:on={selection.head === h} style:--hue={headColor(h)} onclick={() => selection.selectHead(selection.layer, h)}>{h}</button>
      {/each}
    </span>
    <span class="muted">step {app.trace.step}</span>
  </div>

  <section>
    <h2>Arcs over text</h2>
    <Arcs />
  </section>

  <div class="two">
    <section>
      <h2>Attention matrix · L{selection.layer} head {selection.head}</h2>
      <AttnMatrix />
    </section>
    <section>
      <h2>Q·K explorer</h2>
      <QKExplorer />
    </section>
  </div>

  <section>
    <h2>Head gallery</h2>
    <HeadGallery />
  </section>
{/if}

<style>
  .pick { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; font-size: 12px; }
  .pick label, .heads { display: flex; gap: 4px; align-items: center; }
  .heads button { border: 2px solid var(--hue); min-width: 28px; }
  .heads button.on { background: var(--hue); color: #fff; }
  section { margin-bottom: 16px; }
  .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
  @media (max-width: 1400px) { .two { grid-template-columns: minmax(0, 1fr); } }
</style>
