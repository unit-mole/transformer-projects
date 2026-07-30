const METRICS = Object.freeze([
  { key: "mrr_at_10", label: "MRR@10" },
  { key: "ndcg_at_10", label: "nDCG@10" },
  { key: "map_at_100", label: "MAP@100" },
]);

function requiredNumber(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid benchmark value for ${label}.`);
  }
  return parsed;
}

export function formatMetric(value, digits = 4) {
  return requiredNumber(value, "metric").toFixed(digits);
}

export function relativeImprovement(before, after) {
  const baseline = requiredNumber(before, "baseline");
  const finalValue = requiredNumber(after, "reranked value");

  if (baseline === 0) {
    return null;
  }

  return ((finalValue - baseline) / Math.abs(baseline)) * 100;
}

export function validateBenchmarkSummary(summary) {
  if (!summary || summary.status !== "completed") {
    throw new Error("Verified benchmark results are unavailable.");
  }

  if (!Array.isArray(summary.datasets) || summary.datasets.length === 0) {
    throw new Error("Benchmark dataset results are missing.");
  }

  for (const dataset of summary.datasets) {
    if (!dataset.dataset || !dataset.query_count) {
      throw new Error("Benchmark dataset metadata is incomplete.");
    }

    for (const metric of METRICS) {
      requiredNumber(
        dataset.bi_encoder?.[metric.key],
        `${dataset.dataset} bi-encoder ${metric.label}`,
      );
      requiredNumber(
        dataset.reranked?.[metric.key],
        `${dataset.dataset} reranked ${metric.label}`,
      );
      requiredNumber(
        dataset.improvement?.[metric.key]?.absolute,
        `${dataset.dataset} ${metric.label} improvement`,
      );
    }
  }

  return summary;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function signed(value, digits = 4) {
  const number = requiredNumber(value, "signed metric");
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
}

function metricMarkup(dataset, metric) {
  const before = dataset.bi_encoder[metric.key];
  const after = dataset.reranked[metric.key];
  const evidence = dataset.improvement[metric.key];
  const relative = relativeImprovement(before, after);

  return `
    <div class="benchmark-metric">
      <div class="benchmark-metric__label">
        <strong>${escapeHtml(metric.label)}</strong>
        <span>${signed(evidence.absolute)}</span>
      </div>
      <div class="benchmark-metric__values">
        <span>
          <small>Bi-encoder</small>
          <strong>${formatMetric(before)}</strong>
        </span>
        <span class="benchmark-metric__arrow" aria-hidden="true">→</span>
        <span>
          <small>Reranked</small>
          <strong>${formatMetric(after)}</strong>
        </span>
      </div>
      <p>
        ${relative === null ? "Absolute improvement reported." : `${relative.toFixed(1)}% relative improvement.`}
      </p>
    </div>
  `;
}

function datasetCard(dataset) {
  const ndcgEvidence = dataset.improvement.ndcg_at_10;
  const latency = dataset.latency_ms;

  return `
    <article class="benchmark-card">
      <div class="benchmark-card__header">
        <div>
          <span class="benchmark-card__eyebrow">BEIR dataset</span>
          <h3>${escapeHtml(dataset.dataset)}</h3>
          <p>${escapeHtml(dataset.domain)}</p>
        </div>
        <span class="benchmark-card__query-count">
          ${Number(dataset.query_count).toLocaleString()} queries
        </span>
      </div>

      <div class="benchmark-metric-list">
        ${METRICS.map((metric) => metricMarkup(dataset, metric)).join("")}
      </div>

      <div class="benchmark-evidence">
        <div>
          <span>nDCG@10 95% CI</span>
          <strong>
            [${signed(ndcgEvidence.ci_lower)}, ${signed(ndcgEvidence.ci_upper)}]
          </strong>
        </div>
        <div>
          <span>Probability of positive nDCG delta</span>
          <strong>${(ndcgEvidence.probability_positive * 100).toFixed(1)}%</strong>
        </div>
      </div>

      <div class="benchmark-latency">
        <span>Measured mean query latency</span>
        <p>
          <strong>${Number(latency.bi_encoder_mean_query).toFixed(2)} ms</strong>
          bi-encoder
          <span aria-hidden="true">→</span>
          <strong>${Number(latency.two_stage_mean_query).toFixed(2)} ms</strong>
          two-stage
        </p>
      </div>
    </article>
  `;
}

export function renderBenchmarkSummary(rawSummary) {
  const summary = validateBenchmarkSummary(rawSummary);
  const status = document.querySelector("#benchmarkStatus");
  const cards = document.querySelector("#benchmarkCards");
  const queryCount = document.querySelector("#benchmarkQueryCount");
  const documentCount = document.querySelector("#benchmarkDocumentCount");
  const bootstrapCount = document.querySelector("#benchmarkBootstrapCount");

  if (!status || !cards || !queryCount || !documentCount || !bootstrapCount) {
    throw new Error("Benchmark summary interface is incomplete.");
  }

  queryCount.textContent = Number(summary.totals.queries).toLocaleString();
  documentCount.textContent = Number(summary.totals.documents).toLocaleString();
  bootstrapCount.textContent = Number(
    summary.configuration.bootstrap_samples,
  ).toLocaleString();

  cards.innerHTML = summary.datasets.map(datasetCard).join("");
  cards.hidden = false;
  status.hidden = true;
}

export function renderBenchmarkSummaryError(error) {
  const status = document.querySelector("#benchmarkStatus");
  if (!status) {
    return;
  }

  status.className = "notice notice--benchmark-error";
  status.innerHTML = `
    <strong>Benchmark summary could not be loaded.</strong>
    <span>${escapeHtml(error instanceof Error ? error.message : String(error))}</span>
  `;
  status.hidden = false;
}
