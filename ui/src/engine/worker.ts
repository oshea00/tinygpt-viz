// The Web Worker the in-browser backend runs in (see LocalBridge in ../bridge.ts). The first
// message is {type: "_init", base}: where core/web_export.py's files are served from.

import { LocalModel, fetchAssets } from "./local";

let model: LocalModel | null = null;
let queue: ({ type: string } & Record<string, unknown>)[] = [];
let scheduled = false;

const emit = (msg: object | ArrayBuffer) =>
  msg instanceof ArrayBuffer ? postMessage(msg, { transfer: [msg] }) : postMessage(msg);

// Commands that arrive while a trace is computing wait here, so a burst of keystrokes collapses
// into one run.
function drain(): void {
  scheduled = false;
  if (!model) return;
  const cmds = queue;
  queue = [];
  model.handle(cmds);
}

onmessage = async (e: MessageEvent) => {
  const cmd = e.data;
  if (cmd.type === "_init") {
    try {
      model = new LocalModel(await fetchAssets(cmd.base), emit);
      model.start();
    } catch (err) {
      emit({ type: "error", message: `loading the model: ${err instanceof Error ? err.message : err}` });
      return;
    }
  } else queue.push(cmd);
  if (!scheduled) {
    scheduled = true;
    setTimeout(drain, 0);
  }
};
