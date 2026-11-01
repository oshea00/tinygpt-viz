import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import type { Plugin } from "vite";

/** The Google Analytics tag, in every page of a build with GA_MEASUREMENT_ID set. The Pages
 *  workflow sets it; dev servers and other builds go without, so their visits aren't counted. */
function analytics(id = process.env.GA_MEASUREMENT_ID): Plugin {
  if (id && !/^G-[A-Z0-9]+$/.test(id)) throw new Error(`GA_MEASUREMENT_ID is not a G-… id: ${id}`);
  return {
    name: "analytics",
    apply: "build",
    transformIndexHtml: () => id ? [
      { tag: "script", attrs: { async: true, src: `https://www.googletagmanager.com/gtag/js?id=${id}` }, injectTo: "head" },
      { tag: "script", children: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag("js",new Date());gtag("config","${id}");`, injectTo: "head" },
    ] : [],
  };
}

export default defineConfig({
  plugins: [svelte(), analytics()],
  build: { rollupOptions: { input: { main: "index.html", guide: "guide.html" } } },
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
      "/docs": "http://127.0.0.1:8765",   // the guide's Markdown and images
    },
  },
  test: { include: ["src/**/*.test.ts"] },
});
