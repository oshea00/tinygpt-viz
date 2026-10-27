// The guide viewer: renders one Markdown page from docs/ (served at /docs/), named by ?doc=.
// Links to other guide pages stay in the viewer; images and anchors resolve against the page.
import { marked } from "marked";
import "./app.css";
import "./guide.css";
import "./state/theme.svelte";   // the app's saved theme

const main = document.getElementById("guide")!;
const doc = new URLSearchParams(location.search).get("doc") ?? "guide/01-getting-started.md";

/** The viewer URL for a docs/ page, or null for anything that isn't one. */
function viewerHref(url: URL): string | null {
  const docs = new URL("/docs/", location.origin);
  if (url.origin !== location.origin || !url.pathname.startsWith(docs.pathname) || !url.pathname.endsWith(".md")) return null;
  return `?doc=${url.pathname.slice(docs.pathname.length)}${url.hash}`;
}

/** GitHub's heading anchors: lowercase, punctuation dropped, spaces to hyphens. */
function slug(text: string): string {
  return text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s/g, "-");
}

async function show(): Promise<void> {
  if (!/^[\w-]+(\/[\w-]+)*\.md$/.test(doc)) {
    main.textContent = `Not a guide page: ${doc}`;
    return;
  }
  const src = new URL(`/docs/${doc}`, location.origin);
  const r = await fetch(src);
  if (!r.ok) {
    main.textContent = `Couldn't load docs/${doc} (${r.status}).`;
    return;
  }
  main.innerHTML = await marked.parse(await r.text());
  const title = main.querySelector("h1");
  if (title) document.title = `${title.textContent} · TinyGPT guide`;
  for (const h of main.querySelectorAll("h1, h2, h3, h4")) h.id = slug(h.textContent ?? "");
  for (const img of main.querySelectorAll("img")) img.src = new URL(img.getAttribute("src")!, src).href;
  for (const a of main.querySelectorAll("a")) {
    const href = a.getAttribute("href")!;
    if (href.startsWith("#")) continue;
    const url = new URL(href, src);
    a.href = viewerHref(url) ?? url.href;
  }
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
}

show();
