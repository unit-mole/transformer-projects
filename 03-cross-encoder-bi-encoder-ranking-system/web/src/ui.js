import { DISPLAY_LIMITS } from "./constants.js";

export const elements = {
  form: document.querySelector("#searchForm"),
  sampleQuery: document.querySelector("#sampleQuery"),
  customQuery: document.querySelector("#customQuery"),
  candidateK: document.querySelector("#candidateK"),
  candidateKOutput: document.querySelector("#candidateKOutput"),
  rerankK: document.querySelector("#rerankK"),
  rerankKOutput: document.querySelector("#rerankKOutput"),
  searchButton: document.querySelector("#searchButton"),
  searchButtonText: document.querySelector("#searchButtonText"),
  runtimeBadge: document.querySelector("#runtimeBadge"),
  runtimeBadgeText: document.querySelector("#runtimeBadgeText"),
  progressArea: document.querySelector("#progressArea"),
  progressLabel: document.querySelector("#progressLabel"),
  progressPercent: document.querySelector("#progressPercent"),
  progressBar: document.querySelector("#progressBar"),
  errorMessage: document.querySelector("#errorMessage"),
  resultsSection: document.querySelector("#resultsSection"),
  executedQuery: document.querySelector("#executedQuery"),
  latencyMetrics: document.querySelector("#latencyMetrics"),
  qualityMetricsWrapper: document.querySelector(
    "#qualityMetricsWrapper",
  ),
  qualityMetrics: document.querySelector("#qualityMetrics"),
  qualityMetricNote: document.querySelector("#qualityMetricNote"),
  retrievalTableBody: document.querySelector(
    "#retrievalTableBody",
  ),
  rerankingPanel: document.querySelector("#rerankingPanel"),
  rerankingTableBody: document.querySelector(
    "#rerankingTableBody",
  ),
  interpretation: document.querySelector("#interpretation"),
  exportButton: document.querySelector("#exportButton"),
};

export function selectedMode() {
  return (
    document.querySelector(
      'input[name="searchMode"]:checked',
    )?.value ?? "two-stage"
  );
}

export function setRuntimeStatus(status, message) {
  elements.runtimeBadge.className =
    `runtime-badge runtime-badge--${status}`;
  elements.runtimeBadgeText.textContent = message;
}

export function setBusy(isBusy) {
  elements.searchButton.disabled = isBusy;
  elements.sampleQuery.disabled = isBusy;
  elements.customQuery.disabled = isBusy;
  elements.candidateK.disabled = isBusy;
  elements.rerankK.disabled =
    isBusy || selectedMode() === "retrieval-only";

  document
    .querySelectorAll('input[name="searchMode"]')
    .forEach((input) => {
      input.disabled = isBusy;
    });
}

export function showError(error) {
  const message =
    error instanceof Error
      ? error.message
      : "An unexpected browser inference error occurred.";

  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = false;
  setRuntimeStatus("error", "Search failed");
  console.error(error);
}

export function clearError() {
  elements.errorMessage.hidden = true;
  elements.errorMessage.textContent = "";
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function progressLabelFromEvent(event, stageName) {
  const file = event?.file
    ? String(event.file).split("/").at(-1)
    : "";
  const status = event?.status
    ? String(event.status).replaceAll("_", " ")
    : "";

  return (
    [stageName, status, file].filter(Boolean).join(" · ") ||
    `${stageName}…`
  );
}

export function updateProgress(event, stageName) {
  elements.progressArea.hidden = false;
  elements.progressLabel.textContent =
    progressLabelFromEvent(event, stageName);

  const rawProgress = Number(event?.progress);
  const percent = Number.isFinite(rawProgress)
    ? clamp(
        rawProgress <= 1
          ? rawProgress * 100
          : rawProgress,
        0,
        100,
      )
    : 0;

  elements.progressBar.style.width = `${percent}%`;
  elements.progressPercent.textContent =
    `${Math.round(percent)}%`;
}

export function completeProgress(label = "Ready") {
  elements.progressArea.hidden = false;
  elements.progressLabel.textContent = label;
  elements.progressBar.style.width = "100%";
  elements.progressPercent.textContent = "100%";
}

export function hideProgressSoon() {
  window.setTimeout(() => {
    if (!elements.searchButton.disabled) {
      elements.progressArea.hidden = true;
      elements.progressBar.style.width = "0%";
      elements.progressPercent.textContent = "0%";
    }
  }, 900);
}

export function populateSampleQueries(
  queries,
  maximumDocuments,
) {
  elements.sampleQuery.replaceChildren();

  for (const row of queries) {
    const option = document.createElement("option");
    option.value = row.query_id;
    option.textContent = `${row.query_id} — ${row.query}`;
    elements.sampleQuery.append(option);
  }

  const maximumCandidates = Math.min(
    DISPLAY_LIMITS.maximumCandidateK,
    maximumDocuments,
  );

  elements.candidateK.max = String(maximumCandidates);
  elements.candidateK.value = String(
    Math.min(10, maximumCandidates),
  );
  elements.rerankK.max = elements.candidateK.value;
  elements.rerankK.value = String(
    Math.min(5, Number(elements.candidateK.value)),
  );
  syncSliderOutputs();
}

export function syncSliderOutputs() {
  const candidateK = Number(elements.candidateK.value);
  const rerankK = Math.min(
    Number(elements.rerankK.value),
    candidateK,
  );

  elements.rerankK.max = String(candidateK);
  elements.rerankK.value = String(rerankK);
  elements.candidateKOutput.textContent =
    String(candidateK);
  elements.rerankKOutput.textContent = String(rerankK);
}

export function syncModeControls() {
  const mode = selectedMode();
  elements.rerankK.disabled =
    elements.searchButton.disabled ||
    mode === "retrieval-only";

  elements.searchButtonText.textContent =
    mode === "two-stage"
      ? "Run two-stage search"
      : "Run bi-encoder retrieval";
}

export function resolveQuery(queryById) {
  const custom = elements.customQuery.value.trim();

  if (custom) {
    if (
      custom.length <
      DISPLAY_LIMITS.minimumQueryCharacters
    ) {
      throw new Error(
        `Enter at least ${DISPLAY_LIMITS.minimumQueryCharacters} characters.`,
      );
    }
    return {
      queryId: null,
      text: custom.slice(
        0,
        DISPLAY_LIMITS.maximumQueryCharacters,
      ),
      source: "custom",
    };
  }

  const queryId = elements.sampleQuery.value;
  const row = queryById.get(queryId);

  if (!row) {
    throw new Error(
      "Select a valid sample query or enter a custom query.",
    );
  }

  return {
    queryId,
    text: row.query,
    source: "sample",
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function truncate(value, maximum = 118) {
  const text = String(value ?? "");
  return text.length <= maximum
    ? text
    : `${text.slice(0, maximum - 1)}…`;
}

function round(value, digits = 4) {
  if (!Number.isFinite(value)) {
    return null;
  }
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatMilliseconds(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  if (value < 1) {
    return `${value.toFixed(2)} ms`;
  }
  if (value < 1000) {
    return `${value.toFixed(1)} ms`;
  }
  return `${(value / 1000).toFixed(2)} s`;
}

function metricCard(label, value, help, tone = "") {
  const className = tone
    ? `metric-card metric-card--${tone}`
    : "metric-card";

  return `
    <article class="${className}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(help)}</small>
    </article>
  `;
}

function renderLatency(latency, mode) {
  elements.latencyMetrics.innerHTML = [
    metricCard(
      "Model + index setup",
      formatMilliseconds(latency.setupMs),
      latency.setupMs > 0
        ? "Cold-start work"
        : "Already cached in memory",
      latency.setupMs > 0 ? "neutral" : "",
    ),
    metricCard(
      "Query embedding",
      formatMilliseconds(latency.queryEmbeddingMs),
      "MiniLM query encoding",
    ),
    metricCard(
      "Candidate retrieval",
      formatMilliseconds(latency.retrievalMs),
      "Cosine similarity search",
    ),
    metricCard(
      "Cross reranking",
      formatMilliseconds(latency.rerankingMs),
      mode === "two-stage"
        ? "Joint pair scoring"
        : "Not used",
    ),
    metricCard(
      "Total execution",
      formatMilliseconds(latency.totalMs),
      "Includes first-run setup",
      "positive",
    ),
  ].join("");
}

function renderQuality(quality, candidateK, mode) {
  if (!quality) {
    elements.qualityMetricsWrapper.hidden = true;
    return;
  }

  elements.qualityMetricsWrapper.hidden = false;
  elements.qualityMetricNote.textContent =
    mode === "two-stage"
      ? "Computed live from the graded qrels for this selected sample query."
      : "Bi-encoder metrics only; reranking was not run.";

  const cards = [
    metricCard(
      `Recall@${candidateK}`,
      quality.recallAtK.toFixed(3),
      `${quality.relevantDocumentCount} labelled relevant documents`,
    ),
    metricCard(
      "MRR@10 before",
      quality.beforeMrr.toFixed(3),
      "Bi-encoder order",
    ),
    metricCard(
      "nDCG@10 before",
      quality.beforeNdcg.toFixed(3),
      "Bi-encoder graded ranking",
    ),
  ];

  if (mode === "two-stage") {
    cards.push(
      metricCard(
        "MRR@10 after",
        quality.afterMrr.toFixed(3),
        `${quality.mrrImprovement >= 0 ? "+" : ""}${quality.mrrImprovement.toFixed(3)} change`,
        quality.mrrImprovement > 0
          ? "positive"
          : "",
      ),
      metricCard(
        "nDCG@10 after",
        quality.afterNdcg.toFixed(3),
        `${quality.ndcgImprovement >= 0 ? "+" : ""}${quality.ndcgImprovement.toFixed(3)} change`,
        quality.ndcgImprovement > 0
          ? "positive"
          : "",
      ),
      metricCard(
        "Reranking delta",
        quality.ndcgImprovement.toFixed(3),
        "nDCG@10 after − before",
        quality.ndcgImprovement > 0
          ? "positive"
          : "",
      ),
    );
  }

  elements.qualityMetrics.innerHTML = cards.join("");
}

function categoryLabel(category) {
  return String(category ?? "")
    .split("-")
    .map(
      (token) =>
        token.charAt(0).toUpperCase() + token.slice(1),
    )
    .join(" ");
}

function renderRetrievalTable(candidates) {
  elements.retrievalTableBody.innerHTML = candidates
    .map(
      (document) => `
        <tr>
          <td><strong>#${document.retrieval_rank}</strong></td>
          <td class="document-title">
            <strong>${escapeHtml(document.title)}</strong>
            <small>${escapeHtml(truncate(document.document))}</small>
          </td>
          <td>${escapeHtml(categoryLabel(document.category))}</td>
          <td class="score">${round(document.bi_encoder_score, 4).toFixed(4)}</td>
        </tr>
      `,
    )
    .join("");
}

function movementMarkup(movement) {
  if (movement > 0) {
    return `<span class="rank-move rank-move--up">↑ ${movement}</span>`;
  }
  if (movement < 0) {
    return `<span class="rank-move rank-move--down">↓ ${Math.abs(movement)}</span>`;
  }
  return '<span class="rank-move rank-move--same">—</span>';
}

function renderRerankingTable(results, mode) {
  elements.rerankingPanel.hidden = mode !== "two-stage";

  if (mode !== "two-stage") {
    elements.rerankingTableBody.replaceChildren();
    return;
  }

  elements.rerankingTableBody.innerHTML = results
    .map(
      (document) => `
        <tr>
          <td>
            <strong>#${document.reranked_rank}</strong>
            <small class="stage-label">was #${document.retrieval_rank}</small>
          </td>
          <td>${movementMarkup(document.rank_movement)}</td>
          <td class="document-title">
            <strong>${escapeHtml(document.title)}</strong>
            <small>${escapeHtml(truncate(document.document))}</small>
          </td>
          <td class="score">${round(document.cross_encoder_score, 4).toFixed(4)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderInterpretation(
  candidates,
  rerankedResults,
  mode,
  quality,
) {
  if (mode === "retrieval-only") {
    elements.interpretation.innerHTML = `
      <strong>Bi-encoder-only mode:</strong>
      the system encoded the query independently and retrieved
      ${candidates.length} candidates through cosine similarity.
      No pairwise cross-encoder scoring was performed.
    `;
    return;
  }

  const changed = rerankedResults.filter(
    (document) => document.rank_movement !== 0,
  );
  const topBefore = candidates[0]?.title ?? "No result";
  const topAfter =
    rerankedResults[0]?.title ?? topBefore;

  const qualityText = quality
    ? ` Live nDCG@10 changed by ${quality.ndcgImprovement >= 0 ? "+" : ""}${quality.ndcgImprovement.toFixed(3)} for the selected labelled query.`
    : "";

  elements.interpretation.innerHTML = `
    <strong>Reranking analysis:</strong>
    the cross-encoder changed ${changed.length} of
    ${rerankedResults.length} reranked positions. The top result
    changed from “${escapeHtml(topBefore)}” to
    “${escapeHtml(topAfter)}”.${qualityText}
  `;
}

export function renderSearchResults({
  query,
  mode,
  candidateK,
  candidates,
  rerankedResults,
  latency,
  quality,
}) {
  elements.executedQuery.textContent = query;
  renderLatency(latency, mode);
  renderQuality(quality, candidateK, mode);
  renderRetrievalTable(candidates);
  renderRerankingTable(rerankedResults, mode);
  renderInterpretation(
    candidates,
    rerankedResults,
    mode,
    quality,
  );

  elements.resultsSection.hidden = false;
  elements.resultsSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
