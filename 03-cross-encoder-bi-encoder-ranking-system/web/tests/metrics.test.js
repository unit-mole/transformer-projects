import assert from "node:assert/strict";
import test from "node:test";

import {
  ndcgAtK,
  recallAtK,
  reciprocalRankAtK,
} from "../src/metrics.js";

const relevance = new Map([
  ["A", 3],
  ["B", 2],
  ["C", 1],
]);

test("recallAtK measures relevant candidate coverage", () => {
  assert.equal(
    recallAtK(["A", "X", "B"], relevance, 3),
    2 / 3,
  );
});

test("reciprocalRankAtK rewards the first relevant result", () => {
  assert.equal(
    reciprocalRankAtK(["X", "B", "A"], relevance, 10),
    0.5,
  );
});

test("ndcgAtK is highest for ideal graded ordering", () => {
  const ideal = ndcgAtK(["A", "B", "C"], relevance, 3);
  const weaker = ndcgAtK(["C", "B", "A"], relevance, 3);

  assert.equal(ideal, 1);
  assert.ok(weaker < ideal);
});
