<script lang="ts">
  // Which prompt tokens drove a prediction? Three methods side by side, because they often
  // disagree: attention rollout, gradient × input, and occlusion (remove a token, re-run).
  import { app } from "../../state/app.svelte";
  import { concepts } from "../../concepts/store.svelte";
  import { attentionRollout } from "../../lib/concepts";
  import { cssColor, inkOn } from "../../lib/colormap";
  import { row } from "../../lib/tensor";

  const trace = $derived(app.trace);
  const T = $derived(trace?.tokens.length ?? 0);
  let t = $state(-1);
  const pos = $derived(t >= 0 && t < T - 1 ? t : T - 2);
  const target = $derived(trace && pos >= 0 ? trace.text[pos + 1] : "");

  const rollout = $derived.by(() => {
    if (!trace || pos < 0 || !app.hello) return null;
    const { n_layer, n_head } = app.hello.config;
    const w = Array.from({ length: n_layer }, (_, l) => Array.from({ length: n_head }, (_, h) => trace.byId.get(`L${l}.attn.H${h}.weights`)!.output.data));
    return Array.from(attentionRollout(w, T, pos)).slice(0, pos + 1);
  });

  // Ask the server for the gradient trace and the occlusion once per prompt and position.
  let asked = "";
  $effect(() => {
    if (!trace?.prompt || pos < 0 || app.interventions.length) return;
    const key = `${trace.prompt}@${trace.step}#${pos}`;
    if (key === asked) return;
    asked = key;
    app.send({ type: "run", prompt: trace.prompt, targetIndex: pos, tag: "attr" });
    app.send({ type: "occlude", prompt: trace.prompt, position: pos });
  });
  const gradInput = $derived.by(() => {
    const a = concepts.attrTrace;
    if (!a || !trace || a.prompt !== trace.prompt || a.targets[pos] < 0) return null;
    const s = a.byId.get("embed.sum")!;
    if (!s.grad) return null;
    // −∂L/∂x · x: positive where moving the input along itself would raise p(target).
    return Array.from({ length: pos + 1 }, (_, j) => -row(s.grad!, j).reduce((acc, g, k) => acc + g * row(s.output, j)[k], 0));
  });
  const occl = $derived(concepts.occlusion && trace && concepts.occlusion.prompt === trace.prompt && concepts.occlusion.position === pos ? concepts.occlusion.drops : null);

  const rows = $derived([
    ["attention rollout", rollout, "how much attention, compounded through the layers, flows from each token"],
    ["gradient × input", gradInput, "first-order estimate of each token's effect on log p(target)"],
    ["occlusion", occl, "drop in log p(target) when the token's embedding is removed (its position kept)"],
  ] as const);
</script>

{#if !trace || T < 2}
  <p class="muted">Type a prompt of at least two tokens.</p>
{:else}
  <div class="row">
    <label>explain the prediction of
      <select value={pos} onchange={(e) => (t = +(e.target as HTMLSelectElement).value)}>
        {#each trace.text.slice(1) as tok, i}<option value={i}>“{tok}” (after position {i})</option>{/each}
      </select>
    </label>
    <span class="muted">uses the prompt above{app.interventions.length ? "; clear interventions for gradient and occlusion" : ""}</span>
  </div>
  {#each rows as [name, vals, tip]}
    <section class="attr">
      <h2>{name} <span class="muted tip">{tip}</span></h2>
      {#if vals}
        {@const max = Math.max(1e-9, ...vals.map(Math.abs))}
        <div class="toks mono">
          {#each vals as v, j}
            <span class="tok" title="{v.toFixed(4)}">
              <span class="word" style:background={cssColor("diverging", v / max)} style:color={inkOn("diverging", v / max)}>{trace.text[j]}</span>
              <span class="val">{v >= 0 ? "" : "−"}{Math.round(Math.abs(v / max) * 100)}</span>
            </span>
          {/each}
          <span class="tok"><span class="word target">→ {target}</span></span>
        </div>
      {:else}
        <p class="muted">computing…</p>
      {/if}
    </section>
  {/each}
  <p class="muted note">Red: pushed towards “{target}”; blue: against it (rollout is never negative). The number under each
    word is its score as a percentage of the row's largest, so compare which words lead within a row, not sizes across rows.
    Where the three agree, the evidence is strong; where they disagree, each is measuring something different.</p>
{/if}

<style>
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; margin-bottom: 8px; }
  .row label { display: flex; gap: 4px; align-items: center; }
  .attr { margin-bottom: 10px; }
  .tip { font-size: 11px; font-weight: 400; text-transform: none; letter-spacing: 0; }
  .toks { display: flex; flex-wrap: wrap; gap: 2px; font-size: 12px; }
  .tok { display: flex; flex-direction: column; align-items: center; }
  .word { padding: 1px 3px; border-radius: 3px; border: 1px solid var(--line); }
  .val { font-size: 10px; color: var(--muted); }
  .target { border-style: dashed; font-weight: 700; }
  .note { font-size: 11px; }
</style>
