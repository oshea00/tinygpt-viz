<script lang="ts">
  // The papers behind the views: docs/bibliography.md, rendered in the help panel.
  import { marked } from "marked";
  import { docsUrl, fixLinks, guideHref } from "./docs";

  const src = docsUrl("bibliography.md");
  let status = $state("loading…");
  let body: HTMLDivElement;

  $effect(() => {
    (async () => {
      const r = await fetch(src);
      if (!r.ok) {
        status = `Couldn't load docs/bibliography.md (${r.status}).`;
        return;
      }
      body.innerHTML = await marked.parse(await r.text());
      body.querySelector("h1")?.remove();   // the panel shows its own title
      fixLinks(body, src, true);
      status = "";
    })().catch((e) => (status = `Couldn't load docs/bibliography.md (${e}).`));
  });
</script>

<div class="tools">
  <span class="muted">{status || "grouped by the view they relate to"} · <a href={guideHref("bibliography.md")} target="_blank" rel="noreferrer">open in the guide viewer</a></span>
</div>
<div class="list" bind:this={body}></div>

<style>
  .tools { margin: 8px 0 6px; font-size: 12px; }
  .list { max-height: 60vh; overflow-y: auto; padding-right: 6px; max-width: 920px; }
  .list :global(p) { margin: 2px 0 6px; }
  .list :global(h2) { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 10px 0 4px; }
  .list :global(ul) { margin: 2px 0; padding-left: 20px; }
  .list :global(li) { margin: 3px 0; }
  .list :global(a) { color: var(--accent); }
</style>
