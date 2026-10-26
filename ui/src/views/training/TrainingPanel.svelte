<script lang="ts">
  import { app } from "../../state/app.svelte";
  import LossChart from "./LossChart.svelte";
  import Replay from "./Replay.svelte";
  import HelpButton from "../../help/HelpButton.svelte";

  const s = $derived(app.status);
  const train = (action: "play" | "pause" | "step" | "reset") => app.send({ type: "train", action });
</script>

<div>
  <h2>Training <HelpButton topic="training" label="training" /></h2>
  <div class="row">
    {#if s?.playing}
      <button class="primary" onclick={() => train("pause")}>pause</button>
    {:else}
      <button class="primary" onclick={() => train("play")}>play</button>
    {/if}
    <button onclick={() => train("step")} disabled={s?.playing}>one step</button>
    <button onclick={() => confirm("Reset to random weights? Training so far is kept only in checkpoints.") && train("reset")}>reset</button>
  </div>
  <div class="stats mono">
    step <b>{s?.step ?? "–"}</b>
    {#if app.lossPoints.length}· loss <b>{app.lossPoints[app.lossPoints.length - 1][1].toFixed(3)}</b>{/if}
    {#if app.valPoints.length}· val <b>{app.valPoints[app.valPoints.length - 1][1].toFixed(3)}</b>{/if}
    {#if s?.playing}· {Math.round(app.tokensPerSec).toLocaleString()} tok/s{/if}
  </div>
  <LossChart train={app.lossPoints} val={app.valPoints} />

  {#if s}
    <div class="settings">
      <label>learning rate
        <input type="number" step="0.0001" min="0" value={s.lr}
               onchange={(e) => app.send({ type: "set", lr: +(e.target as HTMLInputElement).value })} />
      </label>
      <label>batch size
        <input type="number" step="1" min="1" max="1024" value={s.batchSize}
               onchange={(e) => app.send({ type: "set", batchSize: +(e.target as HTMLInputElement).value })} />
      </label>
      <label>trace every
        <input type="number" step="1" min="1" value={s.traceEvery}
               onchange={(e) => app.send({ type: "set", traceEvery: +(e.target as HTMLInputElement).value })} /> steps
      </label>
    </div>
  {/if}

  <Replay />

  {#if app.autoSample}
    <div class="sample">
      <div class="muted">sample at step {app.autoSample.step}</div>
      <p class="mono">{app.autoSample.text.replaceAll("\x04", "␄\n")}</p>
    </div>
  {/if}
</div>

<style>
  .row { display: flex; gap: 6px; align-items: center; margin: 6px 0; }
  .stats { font-size: 12px; margin: 6px 0; }
  .settings { display: grid; gap: 4px; font-size: 12px; margin: 8px 0; }
  .settings label { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .settings input { width: 90px; }
  .sample p { white-space: pre-wrap; font-size: 12px; margin: 2px 0; max-height: 160px; overflow: auto; }
</style>
