import type { Tensor } from "../types";

/** Every tensor is shown as rows × cols: a scalar is 1×1, a vector is one row. */
export function dims2(t: Tensor): [number, number] {
  if (t.shape.length === 0) return [1, 1];
  if (t.shape.length === 1) return [1, t.shape[0]];
  return [t.shape[0], t.shape.slice(1).reduce((a, b) => a * b, 1)];
}

export function at(t: Tensor, r: number, c: number): number {
  return t.data[r * dims2(t)[1] + c];
}

export function row(t: Tensor, r: number): Float32Array {
  const cols = dims2(t)[1];
  return t.data.subarray(r * cols, (r + 1) * cols);
}

export function col(t: Tensor, c: number): number[] {
  const [rows, cols] = dims2(t);
  return Array.from({ length: rows }, (_, r) => t.data[r * cols + c]);
}

export function fmt(v: number, digits = 4): string {
  if (v === -Infinity) return "−∞";
  if (v === Infinity) return "∞";
  if (Number.isNaN(v)) return "NaN";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4)) return v.toExponential(2);
  return v.toFixed(digits);
}

export function norm(values: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i] * values[i];
  return Math.sqrt(s);
}
