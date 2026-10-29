<script lang="ts">
  // Fine-tuning beside a frozen model: a LoRA adapter trained here on a set of stories, and a
  // preference-tuned (DPO) adapter trained offline to refuse requests about unsafe play.
  import { finetune, type FinetuneSection } from "../../finetune/store.svelte";
  import LoraPanel from "./LoraPanel.svelte";
  import Preferences from "./Preferences.svelte";

  const sections: [FinetuneSection, string][] = [["lora", "LoRA: tilt it towards a topic"], ["preferences", "Preference tuning (DPO): teach it to refuse"]];
</script>

<div class="finetune">
<nav class="sections">
  {#each sections as [key, label]}
    <button class:on={finetune.section === key} onclick={() => (finetune.section = key)}>{label}</button>
  {/each}
</nav>
{#if finetune.section === "lora"}<LoraPanel />{:else}<Preferences />{/if}
</div>

<style>
  /* Controls a little easier to pick out than the app's hairline borders; labels set apart from prose. */
  .finetune :global(button:not(.bar)), .finetune :global(select), .finetune :global(input:not([type="range"]):not([type="checkbox"])),
  .finetune :global(textarea) {
    border-width: 1.5px; border-color: color-mix(in srgb, var(--muted) 55%, var(--line));
  }
  .finetune :global(button.primary) { border-color: var(--accent); }
  .finetune :global(button:not(.bar):hover) { border-color: var(--accent); }
  .finetune :global(.lbl) { font-weight: 600; color: var(--ink); }
  .sections { display: flex; gap: 4px; flex-wrap: wrap; margin: 0 0 10px; }
  .sections button { font-size: 12px; }
  .sections button.on { background: var(--accent-soft); border-color: var(--accent); }
</style>
