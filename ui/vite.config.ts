import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } },
  },
  test: { include: ["src/**/*.test.ts"] },
});
