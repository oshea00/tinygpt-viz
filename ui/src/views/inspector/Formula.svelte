<script lang="ts">
  import katex from "katex";

  let { tex, display = true }: { tex: string; display?: boolean } = $props();
  const html = $derived(katex.renderToString(tex, { throwOnError: false, displayMode: display }));
</script>

<div class="formula">{@html html}</div>

<style>
  .formula :global(.katex-display) { margin: 4px 0; }
</style>
