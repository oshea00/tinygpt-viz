<script lang="ts">
  // Every weight, kind of stage and idea, with the TeX the inspector shows. Stage entries read
  // their formula, backward rule and explanation from the live trace.
  import { app, selection, settings } from "../state/app.svelte";
  import { formatText } from "../lessons/lesson";
  import Formula from "../views/inspector/Formula.svelte";
  import { generic, GLOSSARY, type GlossEntry, type Section } from "./glossary";

  const SECTIONS: Section[] = ["Weights", "Stages", "Concepts"];
  let search = $state("");
  const q = $derived(search.trim().toLowerCase());
  const shown = $derived(q ? GLOSSARY.filter((e) => [e.title, ...e.names, e.what ?? ""].some((s) => s.toLowerCase().includes(q))) : GLOSSARY);

  const stageOf = (e: GlossEntry) => (e.stage ? app.trace?.byId.get(e.stage) : undefined);
  const shapes = (e: GlossEntry) => (e.weights ?? []).flatMap((n) => {
    const t = app.weights?.tensors[n];
    return t ? [`${n.replace(/^L0\./, "L·.")} [${t.shape.join(" × ")}]`] : [];
  });
  // Weights: the stages that read them (one per kind, layer and head numbers dropped).
  const users = (e: GlossEntry) => {
    if (e.section !== "Weights" || !app.trace) return [];
    const names = new Set(e.weights);
    if (e.names.includes("W_U") && app.hello?.config.tied) names.add("wte");
    const ids = app.trace.stages.filter((s) => s.weights.some((w) => names.has(w))).map((s) => s.id);
    return [...new Map(ids.map((id) => [generic(id), id])).values()];
  };

  function open(stageId: string) {
    selection.select(stageId, "forward");
    settings.tab = "inspector";
  }

  // A link brought us here: show that entry (clearing a search that would hide it).
  let list: HTMLDivElement;
  $effect(() => {
    const key = settings.glossaryKey;
    if (!key || !list) return;
    search = "";
    // Scroll the list only: scrollIntoView would also scroll the page and hide the panel's top.
    queueMicrotask(() => {
      const el = list.querySelector<HTMLElement>(`#gloss-${CSS.escape(key)}`);
      if (el) list.scrollTo({ top: el.offsetTop, behavior: "smooth" });
    });
  });
</script>

<div class="tools">
  <input type="search" bind:value={search} placeholder="search the glossary…" spellcheck="false" />
  <span class="muted">{shown.length} of {GLOSSARY.length} entries · formulas are the ones the inspector shows</span>
</div>
<div class="list" bind:this={list}>
  {#each SECTIONS as section}
    {@const entries = shown.filter((e) => e.section === section)}
    {#if entries.length}
      <h4>{section}</h4>
      {#each entries as e (e.key)}
        {@const st = stageOf(e)}
        <div class="entry" id="gloss-{e.key}" class:focus={settings.glossaryKey === e.key}>
          <div class="name"><b>{e.title}</b> <span class="mono muted">{e.names.filter((n) => n !== e.title).slice(0, 4).join(" · ")}</span></div>
          {#if shapes(e).length}<div class="mono muted small">{shapes(e).join(" · ")}</div>{/if}
          {#if e.what}<p>{@html formatText(e.what)}</p>{/if}
          {#if e.section === "Stages" && st?.explain}<p>{st.explain}</p>{/if}
          {#if e.tex}<Formula tex={e.tex} />{/if}
          {#if st?.formula}<Formula tex={st.formula} />{/if}
          {#if e.section === "Stages" && st?.backward}
            <div class="small muted">backward</div><Formula tex={st.backward} />
          {/if}
          {#if e.section === "Weights" && users(e).length}
            <div class="small">used by {#each users(e) as id, i}<button class="link mono" onclick={() => open(id)}>{generic(id)}</button>{i < users(e).length - 1 ? " · " : ""}{/each}</div>
          {:else if e.section === "Stages" && st}
            <div class="small"><button class="link" onclick={() => open(st.id)}>show {st.id} in the inspector</button></div>
          {/if}
        </div>
      {/each}
    {/if}
  {/each}
</div>

<style>
  .tools { display: flex; gap: 10px; align-items: center; margin: 8px 0 6px; font-size: 12px; flex-wrap: wrap; }
  .tools input { width: 240px; }
  .list { position: relative; max-height: 60vh; overflow-y: auto; padding-right: 6px; }
  h4 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 10px 0 4px; }
  .entry { padding: 6px 8px; border-radius: 6px; border: 1px solid transparent; margin-bottom: 2px; max-width: 920px; }
  .entry.focus { border-color: var(--accent); background: var(--accent-soft); }
  .entry p { margin: 2px 0; }
  .small { font-size: 11px; }
  .link { border: none; background: none; color: var(--accent); padding: 0; text-decoration: underline; font-size: 11px; }
</style>
