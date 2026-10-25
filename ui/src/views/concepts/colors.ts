import { HEAD_HUES } from "../../lib/colormap";

/** One hue per category, stable within a capture (categories sorted by name). */
export function categoryColors(categories: string[]): Map<string, string> {
  return new Map([...new Set(categories)].sort().map((c, i) => [c, HEAD_HUES[i % HEAD_HUES.length]]));
}
