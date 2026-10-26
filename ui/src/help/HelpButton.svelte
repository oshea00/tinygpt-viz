<script lang="ts">
  // A small ? beside a panel's heading: opens help on that panel.
  import { settings } from "../state/app.svelte";

  let { topic, label }: { topic: string; label: string } = $props();
</script>

<button class="q" title="help: {label}" aria-label="help: {label}" onclick={() => settings.openHelp(topic)}>?</button>

<style>
  .q {
    width: 17px; height: 17px; padding: 0; border-radius: 50%; font-size: 10.5px; font-weight: 700; line-height: 1;
    color: var(--muted); vertical-align: 1px; margin-left: 4px; text-transform: none; letter-spacing: 0;
  }
  .q:hover { color: var(--accent); }
</style>
