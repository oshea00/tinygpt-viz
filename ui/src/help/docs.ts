// Rendered pages from docs/ (served at /docs/): shared by the guide viewer and the help panel.

/** The viewer URL for a docs/ page, or null for anything that isn't one. */
export function viewerHref(url: URL): string | null {
  const docs = new URL("/docs/", location.origin);
  if (url.origin !== location.origin || !url.pathname.startsWith(docs.pathname) || !url.pathname.endsWith(".md")) return null;
  return `/guide.html?doc=${url.pathname.slice(docs.pathname.length)}${url.hash}`;
}

/** GitHub's heading anchors: lowercase, punctuation dropped, spaces to hyphens. */
export function slug(text: string): string {
  return text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, "").replace(/\s/g, "-");
}

/** Give headings their anchors, and resolve images and links against the page at `src`:
 *  links to other docs pages go to the viewer. `newTab` opens every non-anchor link in a new tab. */
export function fixLinks(root: HTMLElement, src: URL, newTab = false): void {
  for (const h of root.querySelectorAll("h1, h2, h3, h4")) h.id = slug(h.textContent ?? "");
  for (const img of root.querySelectorAll("img")) img.src = new URL(img.getAttribute("src")!, src).href;
  for (const a of root.querySelectorAll("a")) {
    const href = a.getAttribute("href")!;
    if (href.startsWith("#")) continue;
    const url = new URL(href, src);
    a.href = viewerHref(url) ?? url.href;
    if (newTab) {
      a.target = "_blank";
      a.rel = "noreferrer";
    }
  }
}
