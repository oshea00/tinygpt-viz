// Byte-level BPE (GPT-2 style), reading the Hugging Face tokenizer.json that core/tokenizers.py
// trains and loads, so the browser tokenizes exactly as the Python core does.

export const EOS = "\x04";          // end-of-story marker, replaces <|endoftext|>
export const EOS_DISPLAY = "␄";

// The GPT-2 pre-tokenizer: contractions, letter runs, digit runs, other symbols, whitespace.
const UTF8 = new TextEncoder();
const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

interface TokenizerJson {
  added_tokens: { id: number; content: string }[];
  model: { vocab: Record<string, number>; merges: ([string, string] | string)[] };
}

/** Each byte as a printable character, so tokens are strings ("Ġ" for a space). */
function byteToChar(): string[] {
  const keep: number[] = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const out: string[] = new Array(256);
  let n = 0;
  for (let b = 0; b < 256; b++) out[b] = String.fromCodePoint(keep.includes(b) ? b : 256 + n++);
  return out;
}

export class BPETokenizer {
  static readonly EOS_TOKEN = "<|endoftext|>";

  readonly vocabSize: number;
  readonly eosId: number;
  private readonly vocab: Map<string, number>;
  private readonly tokens: string[];
  private readonly ranks = new Map<string, number>();   // "a b" -> merge priority
  private readonly special: { id: number; content: string }[];
  private readonly toChar = byteToChar();
  private readonly toByte = new Map(this.toChar.map((c, b) => [c, b]));
  private readonly display: string[];

  constructor(json: TokenizerJson) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.special = json.added_tokens;
    for (const t of this.special) this.vocab.set(t.content, t.id);
    this.vocabSize = this.vocab.size;
    this.tokens = new Array(this.vocabSize);
    for (const [t, id] of this.vocab) this.tokens[id] = t;
    json.model.merges.forEach((m, i) => this.ranks.set(typeof m === "string" ? m : `${m[0]} ${m[1]}`, i));
    this.eosId = this.vocab.get(BPETokenizer.EOS_TOKEN)!;
    // Display text per token, computed once.
    this.display = this.tokens.map((_, id) => this.show(id));
  }

  /** Token ids for text; the end-of-story marker may be the control character or its display form. */
  encode(text: string): number[] {
    const ids: number[] = [];
    text.replaceAll(EOS_DISPLAY, EOS).split(EOS).forEach((part, i) => {
      if (i) ids.push(this.eosId);
      if (part) ids.push(...this.encodeWithOffsets(part).ids);
    });
    return ids;
  }

  /** Token ids and the [start, end) span (UTF-16 indices) each came from. A token holding part of
   * a multi-byte character gets that whole character's span. */
  encodeWithOffsets(text: string): { ids: number[]; offsets: [number, number][] } {
    const ids: number[] = [];
    const offsets: [number, number][] = [];
    for (const [chunk, base, specialId] of this.splitSpecial(text)) {
      if (specialId !== undefined) {
        ids.push(specialId);
        offsets.push([base, base + chunk.length]);
        continue;
      }
      for (const m of chunk.matchAll(PRETOKENIZE)) {
        const start = base + m.index!;
        // Bytes of the piece, and for each byte the UTF-16 span of the character it belongs to.
        const chars: string[] = [];
        const spans: [number, number][] = [];
        let at = start;
        for (const ch of m[0]) {
          const end = at + ch.length;
          for (const b of UTF8.encode(ch)) {
            chars.push(this.toChar[b]);
            spans.push([at, end]);
          }
          at = end;
        }
        let i = 0;
        for (const piece of this.bpe(chars)) {
          ids.push(this.vocab.get(piece.join(""))!);
          offsets.push([spans[i][0], spans[i + piece.length - 1][1]]);
          i += piece.length;
        }
      }
    }
    return { ids, offsets };
  }

  decode(ids: number[]): string {
    let out = "";
    let run: number[] = [];
    for (const id of ids) {
      if (id === this.eosId) {
        out += this.decodeRun(run) + EOS;
        run = [];
      } else run.push(id);
    }
    return out + this.decodeRun(run);
  }

  tokenText(id: number): string {
    return this.display[id];
  }

  private show(id: number): string {
    if (id === this.eosId) return EOS_DISPLAY;
    return this.decodeRun([id]).replaceAll(" ", "␣").replaceAll("\n", "⏎");
  }

  /** Like the Hugging Face decoder: special tokens are skipped, and broken UTF-8 becomes U+FFFD. */
  private decodeRun(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      if (this.special.some((s) => s.id === id)) continue;
      for (const c of this.tokens[id]) bytes.push(this.toByte.get(c)!);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  /** Text split around special tokens: [chunk, UTF-16 start, special id if it is one]. */
  private splitSpecial(text: string): [string, number, number?][] {
    const out: [string, number, number?][] = [];
    let from = 0;
    for (;;) {
      let best: { at: number; id: number; content: string } | undefined;
      for (const s of this.special) {
        const at = text.indexOf(s.content, from);
        if (at >= 0 && (!best || at < best.at)) best = { at, ...s };
      }
      if (!best) break;
      if (best.at > from) out.push([text.slice(from, best.at), from]);
      out.push([best.content, best.at, best.id]);
      from = best.at + best.content.length;
    }
    if (from < text.length) out.push([text.slice(from), from]);
    return out;
  }

  /** Merge adjacent symbols, lowest-ranked pair first (leftmost on a tie), until none applies. */
  private bpe(chars: string[]): string[][] {
    let parts = chars.map((c) => [c]);
    let syms = [...chars];
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = this.ranks.get(`${syms[i]} ${syms[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) return parts;
      syms = [...syms.slice(0, best), syms[best] + syms[best + 1], ...syms.slice(best + 2)];
      parts = [...parts.slice(0, best), [...parts[best], ...parts[best + 1]], ...parts.slice(best + 2)];
    }
  }
}
