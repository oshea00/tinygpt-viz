import { describe, expect, it } from "vitest";
import { baseWeight, deltaW, frob, refuses, singularValues, symEigenvalues } from "./lora";

const t = (shape: number[], data: number[]) => ({ shape, data: Float32Array.from(data) });

describe("adapter summaries", () => {
  it("multiplies A by B", () => {
    const d = deltaW(t([2, 1], [1, 2]), t([1, 3], [1, 0, -1]), 2);
    expect(Array.from(d.data)).toEqual([2, 0, -2, 4, 0, -4]);
    expect(frob(d)).toBeCloseTo(Math.sqrt(40));
  });

  it("finds a symmetric matrix's eigenvalues", () => {
    expect(symEigenvalues([[2, 1], [1, 2]]).sort()).toEqual([1, 3].map((x) => expect.closeTo(x, 9)));
  });

  it("finds the singular values of A·B from r × r matrices", () => {
    // A·B = diag(3, 2) padded: A = [[3,0],[0,2],[0,0]], B = [[1,0],[0,1]].
    expect(singularValues(t([3, 2], [3, 0, 0, 2, 0, 0]), t([2, 2], [1, 0, 0, 1]))).toEqual([expect.closeTo(3, 6), expect.closeTo(2, 6)]);
    // Rank 1: the second value is zero, and the first is |a|·|b|.
    const sv = singularValues(t([2, 2], [1, 2, 2, 4]), t([2, 2], [1, 1, 0, 0]));
    expect(sv[0]).toBeCloseTo(frob(deltaW(t([2, 2], [1, 2, 2, 4]), t([2, 2], [1, 1, 0, 0]))), 5);
    expect(sv[1]).toBeCloseTo(0, 5);
  });

  it("names the weight a target adapts", () => {
    expect(baseWeight("L3.q")).toBe("L3.W_q");
    expect(baseWeight("L0.mlp.down")).toBe("L0.W_down");
  });

  it("recognises a refusal", () => {
    const r = " I'm sorry, I can't tell that story. It is not safe.";
    expect(refuses(" I'm sorry, I can't tell that story. It is about", r)).toBe(true);
    expect(refuses(" I'm sorry, I can't wait to see you", r)).toBe(false);
  });
});
