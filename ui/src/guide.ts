// The guide viewer: renders one Markdown page from docs/ (see help/docs.ts), named by ?doc=.
// Links to other guide pages stay in the viewer; images and anchors resolve against the page.
// ```math blocks are typeset with KaTeX, as GitHub does.
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked } from "marked";
import "./app.css";
import "./guide.css";
import "./state/theme.svelte";   // the app's saved theme
import { docsUrl, fixLinks, guideHref } from "./help/docs";

marked.use({
  renderer: {
    code({ text, lang }) {
      return lang === "math" ? katex.renderToString(text, { displayMode: true, throwOnError: false }) : false;
    },
  },
});

const main = document.getElementById("guide")!;
const doc = new URLSearchParams(location.search).get("doc") ?? "guide/01-getting-started.md";
const CONTENTS = "README.md";   // docs/README.md lists every page

/** A link back to the contents page, above every other page. */
function addContentsLink(): void {
  if (doc === CONTENTS) return;
  const nav = document.createElement("nav");
  nav.className = "guide-nav";
  const a = document.createElement("a");
  a.href = guideHref(CONTENTS);
  a.textContent = "← Guide contents";
  nav.append(a);
  main.prepend(nav);
}

/** Previous and next links below a numbered guide/NN-*.md page. The chapters and their titles
 *  come from the contents page's links, so a new chapter only needs listing there. */
async function addChapterLinks(): Promise<void> {
  if (!/^guide\/\d+-[\w-]+\.md$/.test(doc)) return;
  const r = await fetch(docsUrl(CONTENTS));
  if (!r.ok) return;
  const chapters = new Map<string, string>();
  for (const [, title, path] of (await r.text()).matchAll(/\[([^\]]+)\]\((guide\/\d+-[\w-]+\.md)\)/g)) {
    if (!chapters.has(path)) chapters.set(path, title);
  }
  const order = [...chapters.keys()].sort();
  const i = order.indexOf(doc);
  if (i < 0) return;
  const link = (path: string | undefined, label: (title: string) => string) => {
    const span = document.createElement("span");
    if (path) {
      const a = document.createElement("a");
      a.href = guideHref(path);
      a.textContent = label(chapters.get(path)!);
      span.append(a);
    }
    return span;
  };
  const nav = document.createElement("nav");
  nav.className = "guide-chapters";
  nav.append(link(order[i - 1], (t) => `‹ ${t}`), link(order[i + 1], (t) => `${t} ›`));
  main.append(nav);
}

async function show(): Promise<void> {
  if (!/^[\w-]+(\/[\w-]+)*\.md$/.test(doc)) {
    main.textContent = `Not a guide page: ${doc}`;
    addContentsLink();
    return;
  }
  const src = docsUrl(doc);
  const r = await fetch(src);
  if (!r.ok) {
    main.textContent = `Couldn't load docs/${doc} (${r.status}).`;
    addContentsLink();
    return;
  }
  main.innerHTML = await marked.parse(await r.text());
  const title = main.querySelector("h1");
  if (title) document.title = `${title.textContent} · TinyGPT guide`;
  fixLinks(main, src);
  addContentsLink();
  await addChapterLinks();
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
}

show();
