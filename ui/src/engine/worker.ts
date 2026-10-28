// The Web Worker the in-browser backend runs in (see LocalBridge in ../bridge.ts). The first
// message is {type: "_init", base, gpu, train}: where core/web_export.py's files are served from,
// whether to use WebGPU when the browser has it, and whether training is on.

import { Gpu } from "./gpu";
import { LocalModel, fetchAssets } from "./local";

let model: LocalModel | null = null;
let queue: ({ type: string } & Record<string, unknown>)[] = [];
let scheduled = false;
let busy = false;

const emit = (msg: object | ArrayBuffer) =>
  msg instanceof ArrayBuffer ? postMessage(msg, { transfer: [msg] }) : postMessage(msg);

// Commands that arrive while a trace is computing wait here, so a burst of keystrokes collapses
// into one run. While training plays, one step runs after each round of commands, then the
// worker yields so new commands (pause, say) get in.
async function drain(): Promise<void> {
  scheduled = false;
  if (!model || busy) return;
  busy = true;
  while (queue.length) {
    const cmds = queue;
    queue = [];
    await model.handle(cmds);
  }
  await model.tick();
  busy = false;
  if (model.playing) schedule();
}

function schedule(): void {
  if (!scheduled) {
    scheduled = true;
    setTimeout(drain, 0);
  }
}

onmessage = async (e: MessageEvent) => {
  const cmd = e.data;
  if (cmd.type === "_init") {
    try {
      const [assets, gpu] = await Promise.all([fetchAssets(cmd.base), cmd.gpu ? Gpu.open().catch(() => null) : null]);
      model = new LocalModel(assets, emit, gpu, !!cmd.train);
      busy = true;
      await model.start();
      busy = false;
    } catch (err) {
      emit({ type: "error", message: `loading the model: ${err instanceof Error ? err.message : err}` });
      return;
    }
  } else queue.push(cmd);
  schedule();
};
