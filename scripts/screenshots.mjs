#!/usr/bin/env node
// Regenerates the guide's screenshots (docs/images/) from a running server, with headless
// Chrome driven over the DevTools protocol. No dependencies beyond Node 20+ and Chrome.
//
//   uv run python -m core.server --port 8766 --train &
//   node scripts/screenshots.mjs --url http://localhost:8766
//
// The last shots train or take a training step, which changes the server's weights in memory:
// use a server started for this, not one you are working in.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const URL = args.url ?? "http://localhost:8765";
const ONLY = args.only?.split(",");
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "docs", "images");
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = 9400 + Math.floor(Math.random() * 400);

// Helpers available to every shot's setup code as `h`.
const PRELUDE = `
window.h = {
  sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  async until(f, ms = 20000) { const t0 = Date.now(); while (Date.now() - t0 < ms) { const v = f(); if (v) return v; await h.sleep(150); } throw new Error("timed out waiting"); },
  has: (text) => [...document.querySelectorAll("button")].some((x) => x.textContent.trim() === text && !x.disabled),
  button(text, root = document) { const b = [...root.querySelectorAll("button")].find((x) => x.textContent.trim() === text); if (!b) throw new Error("no button " + text); return b; },
  async click(text, root) { h.button(text, root).click(); await h.sleep(400); },
  async tab(name) { await h.click(name, document.querySelector(".tabs")); await h.sleep(400); },
  async stage(id) { document.querySelector('button[data-stage="' + id + '"]').click(); await h.sleep(400); },
  // The inspector heatmap's layout, mirroring Heatmap.svelte: row labels (the prompt's tokens)
  // on the left, one cell per element of the [rows × cols] shape in the header.
  grid() {
    const cv = document.querySelector(".wrap canvas"), R = cv.getBoundingClientRect();
    const [rows, cols] = document.querySelector("main .head").innerText.split("[")[1].split("]")[0].split(" × ").map(Number);
    const labels = [...document.querySelectorAll(".tokens .tok")].map((t) => t.textContent.length);
    const lw = Math.max(26, 8 + 6.5 * Math.max(...labels));
    return { cv, R, rows, cols, lw, cw: (R.width - lw) / cols, ch: R.height / rows };
  },
  async cell(r, c) {
    // Zoomed, a wide tensor scrolls sideways: bring the column into view first.
    const pre = h.grid(), wrap = pre.cv.closest(".wrap");
    wrap.scrollLeft = pre.lw + pre.cw * c - wrap.clientWidth / 2;
    await h.sleep(100);
    const g = h.grid();
    g.cv.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: g.R.left + g.lw + g.cw * (c + 0.5), clientY: g.R.top + g.ch * (r + 0.5) }));
    await h.sleep(400);
  },
  // The column of row r's most negative (bluest) cell, read from the canvas pixels.
  bluest(r) {
    const g = h.grid(), k = g.cv.width / g.R.width;
    const ctx = g.cv.getContext("2d"), y = Math.floor((g.ch * (r + 0.5)) * k);
    let best = 0, bestV = -Infinity;
    for (let c = 0; c < g.cols; c++) {
      const [R, G, B] = ctx.getImageData(Math.floor((g.lw + g.cw * (c + 0.5)) * k), y, 1, 1).data;
      if (B - R > bestV) [bestV, best] = [B - R, c];
    }
    return best;
  },
  async select(label, value) {
    const s = [...document.querySelectorAll("label")].find((l) => l.textContent.trim().startsWith(label))?.querySelector("select");
    s.value = String(value); s.dispatchEvent(new Event("change", { bubbles: true })); await h.sleep(200);
  },
};`;

// name, viewport height, setup (async, uses h), region to capture.
const SHOTS = [
  { name: "overview", height: 1000, setup: ``, clip: "body" },
  { name: "pipeline", height: 1000, setup: `await h.stage("L1.attn.H2.weights");`, clip: ".left" },
  { name: "inspector-provenance", height: 1500, setup: `await h.stage("L2.mlp.up"); await h.cell(7, 384);`, clip: "main" },
  { name: "inspector-gradient", height: 1300, setup: `await h.stage("logits"); await h.click("gradient ∂L/∂x"); await h.cell(4, h.bluest(4));`, clip: "main" },
  { name: "attention", height: 1500, setup: `await h.tab("attention"); await h.stage("L1.attn.H3.weights");`, clip: "main" },
  { name: "head-gallery", height: 1700, setup: `await h.tab("attention");`, clip: "section:has(.gallery)" },
  { name: "residual", height: 1300, setup: `await h.tab("residual stream");`, clip: "main" },
  { name: "logit-lens", height: 1000, setup: `await h.tab("logit lens"); document.querySelectorAll(".grid tbody tr")[4].querySelectorAll("td")[10].click(); await h.sleep(400);`, clip: "main" },
  { name: "atlas", height: 1150, setup: `await h.tab("network atlas"); await h.stage("L1.mlp.up"); await h.click("fit");`, clip: "main" },
  { name: "interventions", height: 1180, setup: `await h.tab("interventions"); const f = [...document.querySelectorAll("fieldset")].find((x) => x.textContent.includes("Ablate a head")); f.querySelectorAll("select")[0].value = "1"; f.querySelectorAll("select")[0].dispatchEvent(new Event("change")); f.querySelectorAll("select")[1].value = "2"; f.querySelectorAll("select")[1].dispatchEvent(new Event("change")); await h.sleep(200); h.button("add", f).click(); await h.until(() => document.querySelector(".changes .crow")); await h.sleep(500);`, clip: "main" },
  { name: "generation", height: 1000, setup: `const kv = [...document.querySelectorAll(".gen-controls label")].find((l) => l.textContent.includes("KV cache")).querySelector("input"); kv.click(); await h.sleep(500); for (let i = 0; i < 12; i++) { await h.click("sample one"); }`, clip: ".right > section:first-child" },
  { name: "kv-cache", height: 1100, setup: `const kv = [...document.querySelectorAll(".gen-controls label")].find((l) => l.textContent.includes("KV cache")).querySelector("input"); kv.click(); await h.sleep(500); for (let i = 0; i < 4; i++) { await h.click("sample one"); } await h.stage("L1.attn.H2.weights");`, clip: "main" },
  { name: "lessons-menu", height: 1000, setup: `document.querySelector(".lessons").click(); await h.sleep(300);`, clip: ".menu" },
  { name: "lesson-player", height: 1000, setup: `document.querySelector(".lessons").click(); await h.sleep(200); document.querySelectorAll(".menu .item")[2].click(); await h.sleep(1000); await h.click("next"); await h.click("next");`, clip: ".player" },
  { name: "help", height: 1000, setup: `await h.tab("attention"); await h.click("help");`, clip: ".help" },
  { name: "help-training", height: 1000, setup: `document.querySelector('.right button[title="help: training"]').click(); await h.sleep(400);`, clip: ".help" },
  { name: "concepts-similarity", height: 1400, setup: `await h.tab("concepts"); h.button("capture").click(); await h.until(() => document.querySelector(".setup")?.innerText.includes("captured") && h.has("capture")); await h.sleep(600);`, clip: "main" },
  { name: "concepts-map", height: 1100, setup: `await h.tab("concepts"); h.button("capture").click(); await h.until(() => document.querySelector(".setup")?.innerText.includes("captured") && h.has("capture")); await h.click("projection map"); await h.sleep(600);`, clip: "main" },
  { name: "concepts-probes", height: 1100, setup: `await h.tab("concepts"); h.button("capture").click(); await h.until(() => document.querySelector(".setup")?.innerText.includes("captured") && h.has("capture")); await h.click("probes"); h.button("fit probes at every snapshot").click(); await h.until(() => document.querySelector("svg.chart rect.tr"), 90000); await h.sleep(400);`, clip: "main" },
  { name: "concepts-steering", height: 1100, setup: `await h.tab("concepts"); h.button("capture").click(); await h.until(() => document.querySelector(".setup")?.innerText.includes("captured") && h.has("capture")); await h.click("directions & steering"); const sc = document.querySelector('input[type=range][max="4"]'); sc.value = "0.75"; sc.dispatchEvent(new Event("input", { bubbles: true })); await h.sleep(200); await h.click("generate both"); await h.until(() => document.querySelectorAll(".gen b").length === 2 && [...document.querySelectorAll(".gen b")].every((b) => b.textContent !== "…")); await h.sleep(400);`, clip: "main" },
  { name: "concepts-attribution", height: 1000, setup: `await h.tab("concepts"); await h.click("attribution"); await h.until(() => !document.body.innerText.includes("computing…")); await h.sleep(400);`, clip: "main" },
  // Last: these train, which changes the weights every later shot would show.
  { name: "training", height: 1000, setup: `await h.click("play", document.querySelector(".right")); await h.sleep(5000); await h.click("pause", document.querySelector(".right"));`, clip: ".right > section:nth-child(2)" },
  { name: "training-step", height: 1300, setup: `await h.tab("training step"); h.button("take one training step on this prompt").click(); await h.until(() => document.querySelector(".summary")); await h.sleep(600);`, clip: "main" },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = spawn(CHROME, ["--headless=new", "--no-sandbox", "--enable-unsafe-swiftshader", "--hide-scrollbars", `--remote-debugging-port=${PORT}`, "about:blank"], { stdio: "ignore" });
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); } catch { await sleep(200); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const run = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? "script error");
  return r.result?.result?.value;
};

mkdirSync(OUT, { recursive: true });
let failed = 0;
for (const shot of SHOTS.filter((s) => !ONLY || ONLY.includes(s.name))) {
  try {
    await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: shot.height, deviceScaleFactor: 1, mobile: false });
    await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "light" }] });
    await send("Page.navigate", { url: URL });
    await sleep(1500);
    await run(PRELUDE);
    await run(`(async () => { await h.until(() => document.querySelector('button[data-stage="embed.sum"]')); await h.sleep(500); })()`);
    await run(`(async () => { ${shot.setup} })()`);
    await sleep(500);
    // The region, trimmed at the bottom to its last visible content (panels stretch to the window).
    const rect = await run(`(() => {
      const el = document.querySelector(${JSON.stringify(shot.clip)}), r = el.getBoundingClientRect();
      let bottom = r.top;
      // Leaves only (text, canvases, charts): containers stretch with the panel.
      for (const c of el.querySelectorAll("*")) {
        if (c.children.length && c.tagName !== "svg" && c.tagName !== "CANVAS") continue;
        const b = c.getBoundingClientRect();
        if (b.height > 0 && b.width > 0 && b.bottom <= r.bottom + 1) bottom = Math.max(bottom, b.bottom);
      }
      return { x: r.left, y: r.top, width: r.width, height: Math.min(r.height, bottom - r.top + 12) };
    })()`);
    const pad = shot.clip === "body" ? 0 : 6;
    const clip = { x: Math.max(0, rect.x - pad), y: Math.max(0, rect.y - pad), width: rect.width + 2 * pad, height: Math.min(rect.height + 2 * pad, shot.height - Math.max(0, rect.y - pad)), scale: 1 };
    const shotData = await send("Page.captureScreenshot", { format: "png", clip });
    writeFileSync(resolve(OUT, `${shot.name}.png`), Buffer.from(shotData.result.data, "base64"));
    console.log(`✓ ${shot.name}`);
  } catch (e) {
    failed++;
    console.log(`✗ ${shot.name}: ${e.message}`);
  }
}
ws.close();
chrome.kill();
process.exit(failed ? 1 : 0);
