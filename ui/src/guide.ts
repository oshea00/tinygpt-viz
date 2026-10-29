// The guide viewer: renders one Markdown page from docs/ (see help/docs.ts), named by ?doc=.
// Links to other guide pages stay in the viewer; images and anchors resolve against the page.
// ```math blocks are typeset with KaTeX, as GitHub does.
import katex from "katex";
import "katex/dist/katex.min.css";
import { marked } from "marked";
import "./app.css";
import "./guide.css";
import "./state/theme.svelte";   // the app's saved theme
import { docsUrl, fixLinks } from "./help/docs";

marked.use({
  renderer: {
    code({ text, lang }) {
      return lang === "math" ? katex.renderToString(text, { displayMode: true, throwOnError: false }) : false;
    },
  },
});

const main = document.getElementById("guide")!;
const doc = new URLSearchParams(location.search).get("doc") ?? "guide/01-getting-started.md";

async function show(): Promise<void> {
  if (!/^[\w-]+(\/[\w-]+)*\.md$/.test(doc)) {
    main.textContent = `Not a guide page: ${doc}`;
    return;
  }
  const src = docsUrl(doc);
  const r = await fetch(src);
  if (!r.ok) {
    main.textContent = `Couldn't load docs/${doc} (${r.status}).`;
    return;
  }
  main.innerHTML = await marked.parse(await r.text());
  const title = main.querySelector("h1");
  if (title) document.title = `${title.textContent} · TinyGPT guide`;
  fixLinks(main, src);
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
}

show();
