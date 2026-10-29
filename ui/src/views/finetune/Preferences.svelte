<script lang="ts">
  import { tick } from "svelte";
  // Preference tuning with DPO: an adapter trained offline (core/dpo.py) to decline some requests and
  // answer similar ones ("refusal", which ships with the model, declines stories about unsafe play;
  // more can be added, see docs/instructors/preference-adapters.md). Everything here is computed live from its weights: the
  // pairs' scores, the held-out answers, and the direction in the stream that carries the refusal.
  import { finetune } from "../../finetune/store.svelte";
  import { direction, layer } from "../../lib/concepts";
  import { refuses } from "../../lib/lora";
  import { glossKey } from "../../help/glossary";
  import { app } from "../../state/app.svelte";
  import Formula from "../inspector/Formula.svelte";
  import Curve from "./Curve.svelte";

  const h = $derived(finetune.adapters[finetune.pref]);
  const info = $derived(h?.info ?? {});
  const refusal: string = $derived(info.refusal ?? " I'm sorry, I can't tell that story.");
  const pairs = $derived(finetune.pairs());
  const DPO = String.raw`\mathcal L = -\log \sigma\!\Big(\beta\Big[\log \frac{\pi(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\Big]\Big)`;
  const train = $derived(pairs.filter((p) => p.split === "train"));
  const held = $derived(finetune.heldOut());
  let scale = $state(1);
  const on = $derived(app.adapter?.name === finetune.pref);
  const names = $derived(finetune.prefNames());
  const sample = (kind: string) => train.find((p) => p.kind === kind)?.prompt ?? "";
  const stale = $derived(h && app.status && h.baseStep !== app.status.step);

  const curve = $derived((info.curve ?? []) as { step: number; loss: number; margin: number; accuracy: number }[]);

  // Live pair scores.
  const sc = $derived(finetune.scores);
  const rows = $derived(sc && sc.margin.length === pairs.length ? pairs.map((p, i) => ({
    p, dc: sc.policy_chosen[i] - sc.ref_chosen[i], dr: sc.policy_rejected[i] - sc.ref_rejected[i], m: sc.margin[i],
  })) : []);
  const nats = glossKey("nats");
  // A held-out safe pair the adapter scores ✗ although it still answers with a story: the worked
  // example for reading the three measures together.
  const example = $derived.by(() => {
    const ft = finetune.firstToken;
    if (!ft) return null;
    const r = rows.find((x) => x.p.split === "heldOut" && x.p.kind === "safe" && x.m <= 0);
    const i = r ? held.findIndex((p) => p.prompt === r.p.prompt) : -1;
    if (!r || i < 0 || !sc) return null;
    const k = pairs.indexOf(r.p);
    return { r, base: ft.base[i], tuned: ft.tuned[i], refC: sc.ref_chosen[k], polC: sc.policy_chosen[k],
             refR: sc.ref_rejected[k], polR: sc.policy_rejected[k], answer: finetune.tests[i]?.tuned };
  });
  const pct = (p: number) => (p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`);

  // Held-out answers.
  const tested = $derived(held.map((p, i) => ({ p, ...finetune.tests[i] })));
  const rate = (kind: string, key: "base" | "tuned") => {
    const r = tested.filter((t) => t.p.kind === kind && t[key] !== undefined);
    return r.length ? `${r.filter((t) => refuses(t[key]!, refusal)).length} of ${r.length}` : "—";
  };

  let own = $state("Tell me a story about a boy named Tom who plays with a lighter.");

  // The refusal direction: unsafe minus safe requests' last-token vectors, per snapshot.
  const c = $derived(finetune.direction);
  const separation = $derived.by(() => {
    if (!c) return [];
    const ia = c.items.map((it, i) => (it.category === "unsafe" ? i : -1)).filter((i) => i >= 0);
    const ib = c.items.map((it, i) => (it.category === "safe" ? i : -1)).filter((i) => i >= 0);
    return c.snapshots.map((_, s) => {
      const rows = layer(c, s, false);
      const v = direction(rows, ia, ib);
      const typical = rows.reduce((acc, r) => acc + Math.hypot(...r), 0) / rows.length;
      return { v, ratio: Math.hypot(...v) / typical };
    });
  });
  // The middle of the network by default: the difference keeps growing with depth, but steering a
  // late layer mostly breaks the text.
  let snap = $state(-1);
  const middle = $derived(c ? Math.max(0, c.snapshots.indexOf(`L${Math.floor(app.hello!.config.n_layer / 2)}.resid_mlp`)) : 0);
  const s = $derived(snap >= 0 && snap < separation.length ? snap : middle);
  let steerPrompt = $state("");
  $effect(() => {
    if (!steerPrompt && held.length) steerPrompt = held.find((p) => p.kind === "unsafe")?.prompt ?? "";
  });
  let alpha = $state(1);
  let steeredEl: HTMLElement | undefined = $state();
  function steer(sign: 1 | -1, tuned = true) {
    finetune.steer(steerPrompt, [{ kind: "steer", stage: c!.snapshots[s], vector: Array.from(separation[s].v), scale: sign * alpha,
                                   label: "refusal direction" }], tuned, scale,
                   `${tuned ? "tuned" : "base"} model, direction × ${sign > 0 ? "+" : "−"}${alpha} at ${snapLabel(c!.snapshots[s])}`);
    tick().then(() => steeredEl?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  }
  /** A residual snapshot's id in words: "L3.resid_mlp" → "layer 3, after the MLP". */
  function snapLabel(id: string): string {
    const m = /^L(\d+)\.resid_(attn|mlp)$/.exec(id);
    return m ? `layer ${m[1]}, after ${m[2] === "attn" ? "attention" : "the MLP"}` : id === "embed.sum" ? "the embeddings (x₀)" : id;
  }
</script>

<p class="note">
  <b>Preference tuning</b> trains a model on which of two answers people prefer. <b>DPO</b> (direct preference optimisation) does it with
  an ordinary loss. For a request x with a preferred answer y<sub>w</sub> and a rejected one y<sub>l</sub>:
</p>
<div class="formula"><Formula tex={DPO} /></div>
<p class="note">
  π is the model being tuned and π<sub>ref</sub> the model it started from, so the loss falls as the preferred answer gains probability
  relative to the start and the rejected one loses it. β sets how far the tuned model may move. Here π<sub>ref</sub> is this model, and π
  is the same model with a LoRA adapter: “before” and “after” are the adapter off and on.
</p>

{#if names.length > 1}
  <div class="row">
    <label><span class="lbl">adapter</span> <select value={finetune.pref} onchange={(e) => finetune.choose(e.currentTarget.value)}>
      {#each names as n}<option value={n}>{n}</option>{/each}</select></label>
    {#if info.description}<span class="muted">{info.description}</span>{/if}
  </div>
{/if}

{#if !h}
  <p>No preference adapter is loaded. They are trained offline with <span class="mono">uv run python -m core.dpo</span> (which writes
    <span class="mono">runs/&lt;profile&gt;/adapters/refusal.bin</span>), and the refusal adapter ships with the model.</p>
{:else}
  <div class="box">
    <b>The preferences</b>
    <ul class="prefs">
      <li>{train.length} requests to train on: {train.filter((p) => p.kind === "unsafe").length} it should refuse, such as
        “{sample("unsafe")}”, and {train.filter((p) => p.kind === "safe").length} similar ones it should answer, such as “{sample("safe")}”.</li>
      <li>Every request has two answers, a preferred one and a rejected one:
        <ul>
          <li>unsafe request: preferred, the refusal <span class="mono">“{refusal.trim()}”</span>; rejected, the base model's own story;</li>
          <li>safe request: preferred, the base model's own story; rejected, the refusal.</li>
        </ul>
        So the adapter has to learn <i>which</i> requests to refuse, not just to refuse.</li>
      <li>{held.length} more requests, about activities it never trained on, are held out.</li>
      <li>Trained offline: DPO with β = {info.beta}{info.nll ? `, plus ${info.nll} × the preferred answer's log-likelihood per token` : ""};
        rank {h.rank} on {[...new Set(h.targets.map((t) => t.slice(t.indexOf(".") + 1)))].join(", ")}; {info.steps} steps.</li>
    </ul>
    {#if info.nll}
      <p class="note">Why the extra likelihood term: DPO alone, on this model, learned to push the rejected answers down without making
        the refusal likely, and so never refused. Only the gap between the two answers matters to the DPO loss; adding the preferred
        answer's own likelihood is a common fix (Pang et al., 2024).</p>
    {/if}
    {#if curve.length}
      <div class="charts">
        <div><h3>DPO loss</h3><Curve label="DPO loss" xMax={info.steps ?? 0} series={[{ label: "loss", color: "var(--accent)", points: curve.map((p) => [p.step, p.loss]) }]} /></div>
        <div><h3>Reward margin, preferred − rejected</h3><Curve label="reward margin" xMax={info.steps ?? 0}
          series={[{ label: "β·log-ratio margin", color: "var(--accent)", points: curve.map((p) => [p.step, p.margin]) }]} /></div>
      </div>
    {/if}
  </div>

  <div class="row">
    <button class:primary={!on} onclick={() => app.setAdapter(on ? null : { name: finetune.pref, scale })}>{on ? "switch off" : "switch on everywhere"}</button>
    <label><span class="lbl">strength</span> <input type="range" min="0" max="2" step="0.1" bind:value={scale}
      onchange={() => on && app.setAdapter({ name: finetune.pref, scale })} /> <span class="mono">{scale.toFixed(1)}</span></label>
  </div>
  {#if stale}<p class="warn">Trained on the step-{h.baseStep} weights; the model is now at step {app.status?.step}.</p>{/if}

  <h2>Held-out requests</h2>
  <div class="row">
    <button class="primary" onclick={() => finetune.test(scale)}>answer them, with and without the adapter</button>
    <span class="muted">greedy, 14 tokens · refused: unsafe <b>{rate("unsafe", "base")}</b> → <b>{rate("unsafe", "tuned")}</b>,
      safe <b>{rate("safe", "base")}</b> → <b>{rate("safe", "tuned")}</b> (base → tuned)</span>
  </div>
  {#if Object.keys(finetune.tests).length}
    <table>
      <thead><tr><th>request</th><th></th><th>base model</th><th>with the adapter</th>
        <th title="the probability that the answer's first token is the refusal's (“I” of “I'm sorry”), without → with the adapter">p(starts the refusal)</th></tr></thead>
      <tbody>
        {#each tested as t, i}
          {@const ft = finetune.firstToken}
          <tr>
            <td>{t.p.prompt}</td><td class:unsafe={t.p.kind === "unsafe"}>{t.p.kind}</td>
            <td class="mono">{t.base ?? "…"}</td>
            <td class="mono" class:refused={t.tuned && refuses(t.tuned, refusal)}>{t.tuned ?? "…"}</td>
            <td class="mono num">{#if ft}{pct(ft.base[i])} → <b class:tipping={t.p.kind === "safe" && ft.tuned[i] > 0.1}>{pct(ft.tuned[i])}</b>{:else}…{/if}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="muted note">A refusal of an unsafe request it never saw is generalisation; a refusal of a safe one is <b>over-refusal</b>,
      the classic cost of safety tuning. Look also for answers that start the refusal and then wander off: a small model learns the
      opening words most firmly.</p>
    <p class="muted note">Greedy decoding always takes the most likely token, so it shows only which side of the line a request falls on.
      The last column shows how close it is: the probability that the answer starts with “I”, the first word of the refusal. A safe request
      at tens of percent (highlighted) is answered with a story greedily, but sampled at temperature 0.8 it would be refused some of the time:
      over-refusal hidden by greedy decoding.</p>
  {/if}

  <h2>Ask it yourself</h2>
  <div class="row">
    <input class="prompt mono" bind:value={own} />
    <button class="primary" onclick={() => finetune.ask(own, scale)}>answer with and without</button>
  </div>
  {#if finetune.ownAnswers.base !== undefined || finetune.ownAnswers.tuned !== undefined}
    <div class="two">
      <div><div class="muted">base model</div><p class="gen mono">{finetune.ownAnswers.base ?? "…"}</p></div>
      <div><div class="muted">with the adapter × {scale.toFixed(1)}</div><p class="gen mono">{finetune.ownAnswers.tuned ?? "…"}</p></div>
    </div>
  {/if}

  <h2>The pairs, scored live</h2>
  <div class="row">
    <button class="primary" disabled={finetune.scoring} onclick={() => finetune.score(scale)}>{finetune.scoring ? "scoring…" : "score every pair"}</button>
  </div>
  {#if rows.length}
    <p class="note">Δ log p is how much more likely the adapter makes an answer than the original model did, as a difference of natural-log
      probabilities, in <a class="gloss" href="#{nats}" data-gloss={nats}>nats</a>: +1 nat is e ≈ 2.7 times as likely, +10 is about 22,000 times.
      β times it is DPO's <b>implicit reward</b>: no separate reward model is trained; it is hidden inside the policy. The margin, β × (Δ preferred
      − Δ rejected), is what the loss pushed up. A positive margin means the adapter prefers the preferred answer.</p>
    {#each [["train", "Training pairs", "the pairs it was trained on"], ["heldOut", "Held-out pairs", "activities it never trained on"]] as [split, title, what]}
      {@const group = rows.filter((r) => r.p.split === split)}
      <h3>{title} <span class="muted">({group.length}, {what}): the adapter prefers the preferred answer in
        <b>{group.filter((r) => r.m > 0).length} of {group.length}</b></span></h3>
      <table>
        <thead><tr><th>request</th><th></th><th>preferred answer</th><th title="log π − log π_ref of the preferred answer">Δ log p preferred</th>
          <th title="log π − log π_ref of the rejected answer">Δ log p rejected</th><th title="β · (Δ preferred − Δ rejected): the implicit reward margin">margin</th>
          <th>prefers the preferred?</th></tr></thead>
        <tbody>
          {#each group as r}
            <tr>
              <td>{r.p.prompt}</td><td class:unsafe={r.p.kind === "unsafe"}>{r.p.kind}</td>
              <td class="mono clip" title={r.p.chosen}>{r.p.chosen}</td>
              <td class="mono num">{r.dc >= 0 ? "+" : ""}{r.dc.toFixed(1)}</td><td class="mono num">{r.dr >= 0 ? "+" : ""}{r.dr.toFixed(1)}</td>
              <td class="mono num" class:good={r.m > 0} class:bad={r.m <= 0}>{r.m.toFixed(2)}</td>
              <td class:good={r.m > 0} class:bad={r.m <= 0}>{r.m > 0 ? "✓ yes" : "✗ no"}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/each}
    <div class="box reading">
      <b>Reading the three measures together</b>
      <ul class="prefs">
        <li><b>The answers</b> (held-out requests, above): what the model does by default. Greedy decoding always takes the most likely
          token.</li>
        <li><b>The pair margins</b> (✓ / ✗): what training did to two particular texts, relative to the original model. A ✗ on a held-out
          safe pair means the refusal gained more probability than that particular story did, not that the request is refused.</li>
        <li><b>p(starts the refusal)</b>: how close a request is to tipping, at the first token, where the choice is made.</li>
      </ul>
      <p class="note">Why a ✗ pair can still get a story: the refusal is one exact sentence, but “a story” is millions of different texts that
        share the probability between them, and the pair scores just one of them. Generation chooses at the first token: “I” against every
        way a story can begin, all at once. Preference tuning raised the refusal's probability after <i>every</i> request, safe ones included,
        but for most safe requests not enough to win that first choice.</p>
      {#if example}
        {@const e = example}
        {@const dc = e.polC - e.refC}
        {@const dr = e.polR - e.refR}
        <p class="note"><b>An example from your tables:</b> <i>{e.r.p.prompt}</i> This safe request is in the <b>held-out pairs</b> table. Its
          preferred answer is the base model's own story, “{e.r.p.chosen.trim().slice(0, 40)}…”, and its rejected answer is the refusal.</p>
        <ul class="prefs">
          <li><b>Δ log p preferred = <span class="mono">{dc.toFixed(1)}</span></b>: with the adapter, that particular story became
            {dc < 0 ? "less" : "more"} likely than with the original model.</li>
          <li><b>Δ log p rejected = <span class="mono">{dr >= 0 ? "+" : ""}{dr.toFixed(1)}</span></b>: the refusal became
            {dr > 0 ? "much more" : "less"} likely.</li>
          <li><b>margin</b> = β × (Δ preferred − Δ rejected) = {sc?.beta} × ({dc.toFixed(1)} − {dr >= 0 ? "" : "("}{dr.toFixed(1)}{dr >= 0 ? "" : ")"}) =
            <span class="mono bad">{e.r.m.toFixed(1)}</span>: negative, so <b>✗ no</b>.</li>
          <li>Yet in the <b>held-out requests</b> table above, this request's p(starts the refusal) only goes from
            <span class="mono">{pct(e.base)}</span> to <span class="mono">{pct(e.tuned)}</span>{#if e.answer}, and with the adapter it
            answers “{e.answer.trim().slice(0, 40)}…”{/if}: {e.tuned < 0.5 ? "it still tells a story" : "it is now more likely to refuse than not"}.</li>
        </ul>
      {:else if !finetune.firstToken}
        <p class="muted note">Answer the held-out requests above as well, for a worked example from this run.</p>
      {/if}
      <p class="note">So a safe request at tens of percent (highlighted in the held-out table) is the real over-refusal risk: answered with a
        story greedily, but refused some of the time when answers are sampled, as chat models usually are.</p>
    </div>
  {/if}

  <h2>Where the refusal lives: a direction in the stream</h2>
  <p class="note">Arditi et al. (2024) found that in chat models, refusing is carried by a single direction in the residual stream. If that is
    true here, the stream's average vector for unsafe requests minus its average for safe ones should point along that direction, and adding or
    subtracting that difference should switch the refusal on or off.</p>

  <h3>1. Find the direction</h3>
  <div class="row">
    <button class="primary" disabled={finetune.directionPending} onclick={() => finetune.captureDirection(scale)}>
      {finetune.directionPending ? "reading…" : "find the direction"}</button>
    <span class="muted">reads the stream at the last token of each of the {train.length} training requests, with the adapter on, at every
      point through the network, and takes unsafe minus safe</span>
  </div>
  {#if c && separation.length}
    <h3>2. Choose where in the network to use it</h3>
    <p class="note">One bar per point in the network, from the embeddings at the top to the last layer at the bottom. Each bar's length is
      how far apart unsafe and safe requests are there: the length of the difference, divided by the length of a typical vector. Click a bar
      to choose that point. The middle of the network is chosen to start with: later points separate the requests more, but steering
      there mostly breaks the text.</p>
    <div class="bars" role="radiogroup" aria-label="where to steer">
      {#each separation as sep, i}
        <button class="bar" class:sel={i === s} role="radio" aria-checked={i === s} onclick={() => (snap = i)}
                title="{c.snapshots[i]}: |difference| / typical |vector| = {sep.ratio.toFixed(3)}">
          <span>{i === s ? "▶ " : ""}{snapLabel(c.snapshots[i])}</span><i style:width="{(sep.ratio / Math.max(...separation.map((x) => x.ratio))) * 100}%"></i>
          <span class="mono">{sep.ratio.toFixed(2)}</span>
        </button>
      {/each}
    </div>

    <h3>3. Steer an answer with it</h3>
    <p class="note">The direction, times the strength, is added to (or subtracted from) the stream at <b>{snapLabel(c.snapshots[s])}</b>, at
      every position, while the model answers the request greedily.</p>
    <div class="row">
      <label class="pick"><span class="lbl">request</span> <select bind:value={steerPrompt}>{#each pairs as p}<option value={p.prompt}>{p.kind}: {p.prompt}</option>{/each}</select></label>
      <label><span class="lbl">strength</span> <input type="range" min="0" max="3" step="0.25" bind:value={alpha} /> <span class="mono">{alpha}</span></label>
    </div>
    <div class="row">
      <button onclick={() => steer(-1)}>subtract it (tuned model)</button>
      <button onclick={() => steer(1)}>add it (tuned model)</button>
      <button onclick={() => steer(1, false)}>add it to the base model</button>
    </div>
    <p class="gen mono steered" bind:this={steeredEl}>{#if finetune.steered.label}<span class="muted">{finetune.steered.label}:</span>
      {finetune.steered.text ?? "…"}{:else}<span class="muted">Press a button: the answer appears here.</span>{/if}</p>

    <h3>What to look for</h3>
    <ul class="prefs">
      <li><b>Subtract it</b> while the tuned model answers an <b>unsafe</b> request: if the direction carries the refusal, the model tells the
        story instead.</li>
      <li><b>Add it</b> while it answers a <b>safe</b> request: it should refuse.</li>
      <li><b>Add it to the base model</b>, which never trained on the refusal: it doesn't refuse, but it stops telling a story and talks as
        “I” about being careful (“I will be careful and not let you get hurt”). The direction carries the <i>decision</i> to answer that way;
        the adapter's later layers are what turn it into the exact refusal sentence.</li>
    </ul>
  {/if}
{/if}
<div class="end"></div>

<style>
  .note { font-size: 12px; max-width: 900px; margin: 4px 0 8px; }
  .warn { font-size: 12px; color: #92400e; background: #fef3c7; border-radius: 6px; padding: 6px 8px; max-width: 900px; }
  .formula { max-width: 900px; }
  .prefs { font-size: 12px; margin: 4px 0 6px; padding-left: 18px; max-width: 900px; }
  .prefs li { margin: 2px 0; }
  .box { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; margin: 4px 0; }
  .row label { display: flex; gap: 4px; align-items: center; }
  h2 { font-size: 14px; margin: 16px 0 6px; }
  h3 { font-size: 12px; margin: 8px 0 4px; font-weight: 600; }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; max-width: 900px; }
  .prompt { flex: 1; min-width: 300px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .gen { white-space: pre-wrap; font-size: 12px; background: var(--bg); padding: 6px; border-radius: 6px; margin: 2px 0; }
  table { border-collapse: collapse; font-size: 11px; width: 100%; max-width: 1100px; }
  th, td { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  td.unsafe { color: var(--bad); }
  td.refused { background: var(--accent-soft); }
  .num { text-align: right; white-space: nowrap; }
  .clip { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .reading { margin-top: 12px; max-width: 900px; }
  .tipping { color: #b45309; background: #fef3c7; padding: 0 3px; border-radius: 3px; }
  .good { color: var(--ok); } .bad { color: var(--bad); }

  .bars { display: grid; grid-template-columns: 1fr; gap: 2px; max-width: 560px; }
  .bar { display: grid; grid-template-columns: 170px 1fr 40px; gap: 6px; align-items: center; font-size: 11px; background: none; border: none; padding: 2px; text-align: left; }
  .bar i { display: block; height: 8px; background: var(--accent); border-radius: 0 4px 4px 0; min-width: 1px; }
  .bar.sel { background: var(--accent-soft); outline: 1px solid var(--accent); font-weight: 600; }
  .pick { min-width: 0; max-width: 100%; }
  .pick select { min-width: 0; max-width: 100%; }
  .steered { min-height: 1.6em; max-width: 900px; }
  .end { height: 120px; }
  .bar:hover { background: var(--bg); }
</style>
