import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  build: { rollupOptions: { input: { main: "index.html", guide: "guide.html" } } },
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
      "/docs": "http://127.0.0.1:8765",   // the guide's Markdown and images
    },
  },
  test: { include: ["src/**/*.test.ts"] },
});
