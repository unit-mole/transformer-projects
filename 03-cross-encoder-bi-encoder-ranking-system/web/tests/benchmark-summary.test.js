import assert from "node:assert/strict";
import test from "node:test";

import {
  formatMetric,
  relativeImprovement,
  validateBenchmarkSummary,
} from "../src/benchmark-summary.js";

const validSummary = {
  status: "completed",
  datasets: [
    {
      dataset: "Example",
      query_count: 10,
      bi_encoder: {
        mrr_at_10: 0.5,
        ndcg_at_10: 0.6,
        map_at_100: 0.4,
      },
      reranked: {
        mrr_at_10: 0.6,
        ndcg_at_10: 0.66,
        map_at_100: 0.44,
      },
      improvement: {
        mrr_at_10: { absolute: 0.1 },
        ndcg_at_10: { absolute: 0.06 },
        map_at_100: { absolute: 0.04 },
      },
    },
  ],
};

test("formatMetric returns a four-decimal portfolio value", () => {
  assert.equal(formatMetric(0.648403213), "0.6484");
});

test("relativeImprovement calculates percentage change", () => {
  assert.ok(Math.abs(relativeImprovement(0.5, 0.6) - 20) < 1e-10);
});

test("validateBenchmarkSummary accepts completed metric data", () => {
  assert.equal(validateBenchmarkSummary(validSummary), validSummary);
});

test("validateBenchmarkSummary rejects incomplete data", () => {
  assert.throws(
    () => validateBenchmarkSummary({ status: "not_run", datasets: [] }),
    /unavailable/,
  );
});
