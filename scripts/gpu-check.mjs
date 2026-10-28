#!/usr/bin/env node
// Runs the WebGPU engine check (ui/gpu-check.html) in headless Chrome, which has WebGPU through
// SwiftShader when there is no GPU, against a running Vite dev server.
//
//   cd ui && npx vite --port 5174 &
//   node scripts/gpu-check.mjs [--url http://localhost:5174] [--timeout seconds] [--profile]
//
// --profile (gpu-check.html?profile=1 in a browser) also times where a trace's time goes;
// --verbose prints the page's console as it runs.

import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const URL = `${args.url ?? "http://localhost:5174"}/gpu-check.html${"profile" in args ? "?profile=1" : ""}`;
const CHROME = process.env.CHROME ?? "google-chrome";
const PORT = 9400 + Math.floor(Math.random() * 400);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "gpu-check-"))}`,
  "--enable-unsafe-webgpu", "--enable-features=Vulkan", "about:blank"], { stdio: "ignore" });
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); } catch { await sleep(200); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if ("verbose" in args && m.method === "Runtime.consoleAPICalled") console.log("  page:", m.params.args.map((a) => a.value).join(" "));
};
const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
if ("verbose" in args) await send("Runtime.enable");
await send("Page.navigate", { url: URL });
let result;
const deadline = Date.now() + 1000 * Number(args.timeout ?? 180);
while (!result && Date.now() < deadline) {
  await sleep(200);
  result = (await send("Runtime.evaluate", { expression: "window.__result", returnByValue: true })).result?.value;
}
ws.close();
chrome.kill();
if (!result) {
  console.error("timed out waiting for the check");
  process.exit(1);
}
console.log(`${result.ok ? "ok" : "FAILED"} on ${result.adapter ?? "?"}: fixture ${result.ms?.toFixed(0) ?? "?"} ms` +
            (result.exportedMs !== undefined ? `, exported model ${result.exportedMs.toFixed(0)} ms per trace` : ", no exported model"));
if (result.checked) console.log(`  checked: ${result.checked.join(", ")}`);
if (result.timings) {
  const t = result.timings;
  console.log(`  ${t.dispatches} dispatches, ${(t.bytes / 2 ** 20).toFixed(0)} MB read back: encoding ${t.encode.toFixed(1)} ms, ` +
              `waiting for the GPU ${t.wait.toFixed(1)} ms, copying ${t.copy.toFixed(1)} ms`);
}
if (result.profile) console.log("  profile:", Object.entries(result.profile).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", "));
for (const f of result.failures.slice(0, 40)) console.log("  " + f);
process.exit(result.ok ? 0 : 1);
