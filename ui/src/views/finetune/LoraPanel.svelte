<script lang="ts">
  // Fine-tuning with LoRA (Hu et al. 2021): train a low-rank adapter on a set of stories, with the
  // model's own weights frozen, then see where it plugs in and how far it tilts the model.
  import { finetune, LIVE, READY } from "../../finetune/store.svelte";
  import { baseWeight, deltaW, frob, singularValues } from "../../lib/lora";
  import { weightRange } from "../../lib/scale";
  import { app, selection, settings } from "../../state/app.svelte";
  import Heatmap from "../inspector/Heatmap.svelte";
  import Curve from "./Curve.svelte";

  const TARGETS = ["q", "k", "v", "attn.out", "mlp.up", "mlp.down"];
  const cfg = $derived(app.hello!.config);
  const dims = (t: string) => t === "mlp.up" ? [cfg.d_model, cfg.d_mlp] : t === "mlp.down" ? [cfg.d_mlp, cfg.d_model] : [cfg.d_model, cfg.d_model];
  const params = $derived(finetune.targets.reduce((n, t) => n + finetune.rank * (dims(t)[0] + dims(t)[1]) * cfg.n_layer, 0));
  const nStories = $derived(finetune.storyList().length);
  let editing = $state(false);

  // The curve: each step's loss, and every few steps the measures.
  const pts = (k: "loss" | "trainLoss" | "heldOutLoss" | "pTopic") =>
    finetune.curve.filter((p) => p[k] !== undefined).map((p) => [p.step, p[k]!] as [number, number]);
  const done = $derived(finetune.curve.at(-1)?.step ?? 0);
  const first = $derived(finetune.curve.find((p) => p.pTopic !== undefined));
  const last = $derived(finetune.curve.filter((p) => p.pTopic !== undefined).at(-1));
  const perStep = $derived(done > 0 ? (performance.now() - finetune.startedAt) / done : 0);
  const slowCpu = $derived(app.status?.device === "browser CPU");

  // The adapter to use and inspect: the one trained here, else the one trained in advance.
  const available = $derived([LIVE, READY].filter((n) => finetune.adapters[n]));
  let chosen = $state<string | null>(null);
  const name = $derived(chosen && finetune.adapters[chosen] ? chosen : available[0]);
  const header = $derived(name ? finetune.adapters[name] : undefined);
  const on = $derived(!!name && app.adapter?.name === name);
  let scale = $state(1);
  const stale = $derived(header && app.status && header.baseStep !== app.status.step);

  function toggle() {
    app.setAdapter(on ? null : { name, scale });
  }
  function setScale(v: number) {
    scale = v;
    if (on) app.setAdapter({ name, scale });
  }

  let prompt = $state("Once upon a time,");
  let seed = $state(0);
  const topicRe = $derived(new RegExp(`(\\b${finetune.topic.replace(/[^\w]/g, "")}\\w*)`, "gi"));
  const marked = (text: string) => text.split(topicRe).map((part, i) => ({ part, hit: i % 2 === 1 }));
  const count = (text?: string) => (text?.match(topicRe) ?? []).length;

  // Anatomy: one target's A, B and ΔW = A·B · strength, against the frozen W it is added to.
  const tensors = $derived(name ? finetune.weightsOf(name) : undefined);
  const targets = $derived(header ? [...new Set(header.targets.map((t) => t.slice(t.indexOf(".") + 1)))] : []);
  $effect(() => {
    if (targets.length && !targets.includes(finetune.target)) finetune.target = targets[0];
  });
  const id = $derived(`L${finetune.layer}.${finetune.target}`);
  const A = $derived(tensors?.[`${id}.lora_A`]);
  const B = $derived(tensors?.[`${id}.lora_B`]);
  const dW = $derived(A && B ? deltaW(A, B, scale) : null);
  const W = $derived(app.weights?.tensors[baseWeight(id)]);
  const sv = $derived(A && B ? singularValues(A, B, scale) : []);
  const range = $derived(dW ? weightRange(dW, settings.clip) : null);
  const rangeA = $derived(A ? weightRange(A, settings.clip) : null);
  const rangeB = $derived(B ? weightRange(B, settings.clip) : null);
  // ‖ΔW‖ / ‖W‖ for every adapted stage: where the adapter changed the model most.
  const sizes = $derived(header && tensors ? header.targets.map((t) => {
    const a = tensors[`${t}.lora_A`], b = tensors[`${t}.lora_B`], w = app.weights?.tensors[baseWeight(t)];
    return { t, rel: a && b && w ? frob(deltaW(a, b, scale)) / frob(w) : 0 };
  }) : []);
  const maxRel = $derived(Math.max(1e-9, ...sizes.map((s) => s.rel)));

  function inspect() {
    if (!on) app.setAdapter({ name, scale });
    selection.select(`${id}.lora`, "forward");
    settings.tab = "inspector";
  }
</script>

<p class="note">
  <b>LoRA</b> fine-tunes without touching the model's weights. Beside a weight W, an adapter adds a product of two thin
  matrices, A ({cfg.d_model}×r) and B (r×{cfg.d_model}): the stage computes x·W + (x·A)·B. Only A and B train. B starts at
  zero, so the adapter starts as a no-op, and the change it can make, ΔW = A·B, has rank r at most. Train one on the
  stories below and the model drifts towards them.
</p>

<div class="box">
  <div class="row">
    <b>{nStories} training stories</b> <span class="muted">(TinyStories stories that mention “{finetune.topic}” at least three times)</span>
    <button onclick={() => (editing = !editing)}>{editing ? "hide" : "show / edit"}</button>
  </div>
  {#if editing}
    <textarea rows="8" bind:value={finetune.stories} spellcheck="false"></textarea>
    <div class="row"><label><span class="lbl">topic word (measured below)</span> <input class="mono" bind:value={finetune.topic} size="10" /></label>
      <span class="muted">One story per paragraph. Any topic works: paste stories about something else and change the word.</span></div>
  {/if}
  <div class="row">
    <span class="lbl">adapt</span>
    {#each TARGETS as t}
      <label><input type="checkbox" checked={finetune.targets.includes(t)} disabled={finetune.running}
        onchange={(e) => (finetune.targets = (e.target as HTMLInputElement).checked ? [...finetune.targets, t] : finetune.targets.filter((x) => x !== t))} />
        <span class="mono">{t}</span></label>
    {/each}
    <label><span class="lbl">rank r</span> <select bind:value={finetune.rank} disabled={finetune.running}>{#each [1, 2, 4, 8, 16] as r}<option value={r}>{r}</option>{/each}</select></label>
    <label><span class="lbl">steps</span> <input type="number" min="10" max="2000" step="10" bind:value={finetune.steps} disabled={finetune.running} /></label>
    <label><span class="lbl">learning rate</span> <input type="number" min="0.0001" max="0.05" step="0.0005" bind:value={finetune.lr} disabled={finetune.running} /></label>
  </div>
  <div class="row">
    <span class="muted">{params.toLocaleString()} trainable numbers ({((params / app.hello!.params) * 100).toFixed(2)}% of the model's
      {app.hello!.params.toLocaleString()}), in every layer's {finetune.targets.join(", ") || "—"}</span>
    {#if finetune.running}
      <button onclick={() => finetune.stop()}>stop</button>
      <span class="mono">step {done} / {finetune.total}{perStep ? ` · ${Math.ceil(((finetune.total - done) * perStep) / 1000)} s left` : ""}</span>
    {:else}
      <button class="primary" disabled={!nStories || !finetune.targets.length} onclick={() => finetune.train()}>
        {finetune.adapters[LIVE] ? "train again" : "train the adapter"}</button>
    {/if}
  </div>
  {#if slowCpu}
    <p class="warn">This page runs the model on the CPU (no WebGPU), where training takes many minutes. The adapter trained in
      advance, <span class="mono">{READY}</span>, is the same training with these settings: pick it below to skip ahead.</p>
  {/if}
</div>

{#if finetune.curve.length}
  <div class="charts">
    <div>
      <h3>Loss</h3>
      <Curve label="adapter training loss" xMax={finetune.total} series={[
        { label: "this step's batch", color: "var(--muted)", points: pts("loss") },
        { label: "on the stories", color: "var(--accent)", points: pts("trainLoss"), dots: true },
        { label: "on other stories", color: "#d97706", points: pts("heldOutLoss"), dots: true },
      ]} />
    </div>
    <div>
      <h3>p(“{finetune.topic}”) after a neutral opener</h3>
      <Curve label="probability of the topic word" xMax={finetune.total} format={(v) => `${(v * 100).toFixed(1)}%`}
             series={[{ label: `p(${finetune.topic})`, color: "var(--accent)", points: pts("pTopic"), dots: true }]} />
    </div>
  </div>
  {#if first && last && last !== first}
    <p class="note">After eight openers that end where a noun goes (“Once upon a time, there was a”), the model's probability of
      “{finetune.topic}” went from <b class="mono">{(first.pTopic! * 100).toFixed(2)}%</b> to <b class="mono">{(last.pTopic! * 100).toFixed(1)}%</b>.
      Its loss on ordinary validation stories went from <b class="mono">{first.heldOutLoss!.toFixed(3)}</b> to
      <b class="mono">{last.heldOutLoss!.toFixed(3)}</b>: the price of the tilt.</p>
  {/if}
{/if}

{#if name && header}
  <h2>Switch it on</h2>
  <div class="row">
    {#if available.length > 1}
      <label><span class="lbl">adapter</span> <select value={name} onchange={(e) => { if (on) app.setAdapter(null); chosen = (e.target as HTMLSelectElement).value; }}>
        {#each available as n}<option value={n}>{n === LIVE ? `${n} (trained here)` : `${n} (trained in advance)`}</option>{/each}</select></label>
    {:else}<span class="mono">{name}</span>{/if}
    <button class:primary={!on} onclick={toggle}>{on ? "switch off" : "switch on everywhere"}</button>
    <label><span class="lbl">strength</span> <input type="range" min="0" max="2" step="0.1" value={scale} oninput={(e) => setScale(+(e.target as HTMLInputElement).value)} />
      <span class="mono">{scale.toFixed(1)}</span></label>
    <span class="muted">rank {header.rank} · {header.params.toLocaleString()} numbers · trained on step {header.baseStep}</span>
  </div>
  {#if stale}<p class="warn">This adapter was trained on the step-{header.baseStep} weights; the model is now at step {app.status?.step}.
    An adapter only means something on the weights it was trained with.</p>{/if}
  <p class="muted note">Switched on, every view (the pipeline, the inspector, attention, the logit lens, generation) runs the adapted
    model, and the pipeline gains an <span class="mono">.lora</span> stage beside each stage it adapts. Strength scales B: 0 is the base
    model, 1 the trained adapter.</p>

  <h2>Generate with and without it</h2>
  <div class="row">
    <input class="prompt mono" bind:value={prompt} />
    <button class="primary" onclick={() => finetune.generate(prompt, name, scale, 60, 0.8, ++seed)}>generate both</button>
    <span class="muted">temperature 0.8 · each press draws new random numbers (seed {seed}), the same for both</span>
  </div>
  {#if finetune.generated.base !== undefined || finetune.generated.adapter !== undefined}
    <div class="two">
      <div><div class="muted">base model · “{finetune.topic}” ×{count(finetune.generated.base)}</div>
        <p class="gen mono">{prompt}{#each marked(finetune.generated.base ?? "…") as m}{#if m.hit}<mark>{m.part}</mark>{:else}{m.part}{/if}{/each}</p></div>
      <div><div class="muted">with {name} × {scale.toFixed(1)} · “{finetune.topic}” ×{count(finetune.generated.adapter)}</div>
        <p class="gen mono">{prompt}{#each marked(finetune.generated.adapter ?? "…") as m}{#if m.hit}<mark>{m.part}</mark>{:else}{m.part}{/if}{/each}</p></div>
    </div>
  {/if}

  <h2>Where it plugs in</h2>
  <div class="row">
    <label><span class="lbl">layer</span> <input type="range" min="0" max={cfg.n_layer - 1} bind:value={finetune.layer} /> <span class="mono">{finetune.layer}</span></label>
    <label><span class="lbl">stage</span> <select bind:value={finetune.target}>{#each targets as t}<option value={t}>{t}</option>{/each}</select></label>
    <button onclick={inspect} title="switch the adapter on and open this stage's low-rank term in the inspector">open <span class="mono">{id}.lora</span> in the inspector</button>
  </div>
  <div class="diagram mono">
    <span class="frozen">x · {baseWeight(id).split(".")[1]} (frozen, {W ? `${W.shape[0]}×${W.shape[1]}` : ""})</span> <span>+</span>
    <span class="trained">(x · A) · B · {scale.toFixed(1)}</span> <span>=</span> <span>{id}</span>
  </div>
  {#if A && B && dW && range && rangeA && rangeB}
    <div class="anatomy">
      <div><div class="muted">A [{A.shape.join("×")}]: each row of x becomes {A.shape[1]} numbers</div><Heatmap tensor={A} range={rangeA} maxHeight={220} maxCell={6} /></div>
      <div><div class="muted">B [{B.shape.join("×")}]: and back to {B.shape[1]}</div><Heatmap tensor={B} range={rangeB} maxHeight={220} maxCell={6} /></div>
      <div><div class="muted">ΔW = A·B [{dW.shape.join("×")}]: rank {header.rank} at most</div><Heatmap tensor={dW} {range} maxHeight={220} maxCell={2} /></div>
    </div>
    <div class="row">
      <span>singular values of ΔW:</span>
      {#each sv as v}<span class="sv mono" style:--w="{(v / (sv[0] || 1)) * 60}px" title={v.toPrecision(4)}><i></i>{v.toFixed(3)}</span>{/each}
      <span class="muted">· ‖ΔW‖ = {frob(dW).toFixed(3)}{W ? ` · ‖W‖ = ${frob(W).toFixed(2)}` : ""}</span>
    </div>
    <p class="muted note">A full fine-tune would change all {W ? (W.shape[0] * W.shape[1]).toLocaleString() : ""} numbers of this weight; the
      adapter's change is described by {(A.data.length + B.data.length).toLocaleString()}. Only {header.rank} directions are available to it,
      and the singular values show how much it uses each.</p>
    <h3>Size of the change, ‖ΔW‖ / ‖W‖, at every adapted stage</h3>
    <div class="bars">
      {#each sizes as s}
        <button class="bar" class:sel={s.t === id} onclick={() => { finetune.layer = +s.t.slice(1, s.t.indexOf(".")); finetune.target = s.t.slice(s.t.indexOf(".") + 1); }}
                title="{s.t}: {(s.rel * 100).toFixed(2)}%">
          <span class="mono">{s.t}</span><i style:width="{(s.rel / maxRel) * 100}%"></i><span class="mono">{(s.rel * 100).toFixed(2)}%</span>
        </button>
      {/each}
    </div>
  {:else}
    <p class="muted">Loading the adapter's weights…</p>
  {/if}
{:else if !finetune.running}
  <p>Train the adapter, or wait for the one trained in advance to load. Training takes about half a minute on a GPU.</p>
{/if}

<style>
  .note { font-size: 12px; max-width: 900px; margin: 4px 0 8px; }
  .warn { font-size: 12px; color: #92400e; background: #fef3c7; border-radius: 6px; padding: 6px 8px; max-width: 900px; }
  .box { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; margin: 4px 0; }
  .row label { display: flex; gap: 4px; align-items: center; }
  input[type="number"] { width: 70px; }
  textarea { width: 100%; font-family: var(--mono); font-size: 11px; }
  h2 { font-size: 14px; margin: 16px 0 6px; }
  h3 { font-size: 12px; margin: 8px 0 4px; font-weight: 600; }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; max-width: 900px; }
  .prompt { flex: 1; min-width: 240px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .gen { white-space: pre-wrap; font-size: 12px; background: var(--bg); padding: 6px; border-radius: 6px; margin: 2px 0; }
  mark { background: var(--accent-soft); color: var(--ink); border-bottom: 2px solid var(--accent); }
  .diagram { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; margin: 6px 0; }
  .diagram span { padding: 3px 6px; border-radius: 5px; }
  .frozen { background: var(--bg); border: 1px dashed var(--line); }
  .trained { background: var(--accent-soft); border: 1px solid var(--accent); }
  .anatomy { display: grid; grid-template-columns: minmax(0, 0.6fr) minmax(0, 1fr) minmax(0, 1.4fr); gap: 12px; font-size: 11px; }
  .sv { display: inline-flex; align-items: center; gap: 3px; }
  .sv i { display: inline-block; height: 8px; width: var(--w); background: var(--accent); border-radius: 0 4px 4px 0; }
  .bars { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 2px 12px; max-width: 900px; }
  .bar { display: grid; grid-template-columns: 90px 1fr 50px; gap: 6px; align-items: center; font-size: 11px; background: none; border: none; padding: 2px; text-align: left; }
  .bar i { display: block; height: 8px; background: var(--accent); border-radius: 0 4px 4px 0; min-width: 1px; }
  .bar.sel { background: var(--accent-soft); }
  .bar:hover { background: var(--bg); }
</style>
