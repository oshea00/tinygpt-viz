// Pipeline map layout, derived from the trace's stage list: residual-stream stages on
// the spine, sublayer stages on a branch, per-head stages grouped into one row.

import type { Stage, Trace } from "../types";

export interface PipeRow {
  key: string;
  label: string;
  lane: "spine" | "branch";
  stages: Stage[];
  group: string;
  segStart: boolean;   // first branch row after leaving the spine
  segEnd: boolean;     // last branch row before rejoining
}

const LABELS: Record<string, string> = {
  "embed.in": "token + position embeddings",
  "embed.sum": "x₀ = tok + pos",
  ln1: "LayerNorm 1",
  qkv: "Q · K · V projections",
  "H.q": "split Q into heads",
  "H.k": "split K into heads",
  "H.v": "split V into heads",
  "H.scores": "scores Q·Kᵀ/√d",
  "H.masked": "causal mask",
  "H.weights": "softmax → attention",
  "H.mix": "mix values",
  "attn.concat": "concat heads",
  "attn.out": "output projection W_O",
  resid_attn: "+ attention",
  ln2: "LayerNorm 2",
  "mlp.up": "MLP up",
  "mlp.gelu": "GELU",
  "mlp.down": "MLP down",
  resid_mlp: "+ MLP",
  ln_f: "final LayerNorm",
  logits: "unembed → logits",
  probs: "softmax → probs",
  loss: "loss",
};

const SPINE_OUTPUT = new Set(["ln_f", "logits", "probs", "loss"]);

function rowKey(s: Stage): { key: string; label: string; group: string } {
  if (s.id === "embed.tok" || s.id === "embed.pos") return { key: "embed.in", label: LABELS["embed.in"], group: "Input" };
  if (s.id === "embed.sum") return { key: s.id, label: LABELS[s.id], group: "Input" };
  if (SPINE_OUTPUT.has(s.id)) return { key: s.id, label: LABELS[s.id], group: "Output" };
  const m = /^L(\d+)\.(.+)$/.exec(s.id)!;
  const rest = m[2];
  const group = `Block ${m[1]}`;
  const head = /^attn\.H\d+\.(\w+)$/.exec(rest);
  if (head) return { key: `L${m[1]}.H.${head[1]}`, label: LABELS[`H.${head[1]}`], group };
  if (rest === "q" || rest === "k" || rest === "v") return { key: `L${m[1]}.qkv`, label: LABELS.qkv, group };
  return { key: s.id, label: LABELS[rest] ?? rest, group };
}

export function pipelineRows(trace: Trace): PipeRow[] {
  const rows: PipeRow[] = [];
  const byKey = new Map<string, PipeRow>();
  for (const s of trace.stages) {
    const { key, label, group } = rowKey(s);
    let r = byKey.get(key);
    if (!r) {
      const spine = s.kind === "add" || SPINE_OUTPUT.has(s.id);
      r = { key, label, lane: spine ? "spine" : "branch", stages: [], group, segStart: false, segEnd: false };
      byKey.set(key, r);
      rows.push(r);
    }
    r.stages.push(s);
  }
  rows.forEach((r, i) => {
    if (r.lane !== "branch") return;
    r.segStart = i === 0 || rows[i - 1].lane === "spine" || rows[i - 1].group !== r.group;
    r.segEnd = i === rows.length - 1 || rows[i + 1].lane === "spine";
  });
  return rows;
}

/** Forward then backward: the stepper walks down the network and back up through gradients. */
export function stepSequence(trace: Trace): { id: string; phase: "forward" | "backward" }[] {
  const fwd = trace.stages.map((s) => ({ id: s.id, phase: "forward" as const }));
  const bwd = [...trace.stages].reverse().filter((s) => s.grad).map((s) => ({ id: s.id, phase: "backward" as const }));
  return [...fwd, ...bwd];
}
