import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// `--mode demo` builds the static, server-free demo from the weights core/export.py writes.
export default defineConfig(({ mode }) => ({
  plugins: [svelte()],
  ...(mode === "demo" ? { publicDir: "demo-public", base: "./", build: { outDir: "dist-demo" } } : {}),
  server: {
    proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } },
  },
  test: { include: ["src/**/*.test.ts"] },
}));
