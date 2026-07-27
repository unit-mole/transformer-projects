(() => {
  "use strict";

  const BUILD = "20260727-3";
  const byId = (id) => document.getElementById(id);
  const withVersion = (path) => `${path}${path.includes("?") ? "&" : "?"}v=${BUILD}`;

  async function readJson(path) {
    try {
      const response = await fetch(withVersion(path), { cache: "no-store" });
      if (!response.ok) return null;
      return await response.json();
    } catch (error) {
      console.warn(`Could not load ${path}`, error);
      return null;
    }
  }

  const percent = (value, digits = 1) =>
    Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "Unavailable";

  async function renderMetrics() {
    const status = byId("evaluation-status");
    const recallValue = byId("metric-recall");
    const recallDetail = byId("metric-recall-detail");
    const mrrValue = byId("metric-mrr");
    const mrrDetail = byId("metric-mrr-detail");
    const latencyValue = byId("metric-latency");
    const latencyDetail = byId("metric-latency-detail");
    const cosineValue = byId("metric-cosine");
    const cosineDetail = byId("metric-cosine-detail");

    if (!status || !recallValue || !mrrValue || !latencyValue) {
      console.error("Evaluation-card elements were not found. The old index.html may still be served.");
      return;
    }

    const [recall, mrr, latency, cosine] = await Promise.all([
      readJson("./data/recall_at_k_results.json"),
      readJson("./data/mrr_results.json"),
      readJson("./data/query_latency_results.json"),
      readJson("./data/cosine_similarity_analysis.json"),
    ]);

    let completed = 0;

    if (recall?.status === "completed") {
      recallValue.textContent = `R@1 ${percent(recall.recall_at_1)} · R@3 ${percent(recall.recall_at_3)}`;
      recallDetail.textContent = `R@5 ${percent(recall.recall_at_5)} · R@10 ${percent(recall.recall_at_10)} · ${recall.query_count ?? "?"} verified queries`;
      completed += 1;
    } else {
      recallValue.textContent = "Not available";
      recallDetail.textContent = "Missing or incomplete recall_at_k_results.json";
    }

    if (mrr?.status === "completed" && Number.isFinite(Number(mrr.mrr))) {
      mrrValue.textContent = Number(mrr.mrr).toFixed(4);
      mrrDetail.textContent = "Mean reciprocal rank across the verified evaluation query set.";
      completed += 1;
    } else {
      mrrValue.textContent = "Not available";
      mrrDetail.textContent = "Missing or incomplete mrr_results.json";
    }

    if (latency?.status === "completed" && Array.isArray(latency.results) && latency.results.length) {
      const row = latency.results.find((item) => Number(item.top_k) === 5) || latency.results[0];
      latencyValue.textContent = `${Number(row.average_ms).toFixed(2)} ms average`;
      latencyDetail.textContent = `Top-${row.top_k} · min ${Number(row.minimum_ms).toFixed(2)} ms · max ${Number(row.maximum_ms).toFixed(2)} ms · ${row.measurements} runs`;
      completed += 1;
    } else {
      latencyValue.textContent = "Not available";
      latencyDetail.textContent = "Missing or incomplete query_latency_results.json";
    }

    if (cosine?.status === "completed") {
      const mean = cosine.mean_top_score ?? cosine.average_top_score ?? cosine.mean_similarity;
      cosineValue.textContent = Number.isFinite(Number(mean)) ? `${Number(mean).toFixed(3)} mean score` : "Analysis completed";
      cosineDetail.textContent = cosine.summary || cosine.note || "See the cosine analysis JSON for details.";
      completed += 1;
    } else {
      cosineValue.textContent = "Pending manual review";
      cosineDetail.textContent = "Recall, MRR, and latency are verified; false-positive analysis is still pending.";
    }

    status.textContent = `${completed} verified evaluation artifact${completed === 1 ? "" : "s"} loaded from static JSON. Build ${BUILD}.`;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMetrics, { once: true });
  } else {
    renderMetrics();
  }
})();
