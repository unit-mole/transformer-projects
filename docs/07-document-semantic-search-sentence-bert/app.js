import { BROWSER_MODEL, embedTexts } from "./embeddings.js";
import {
  escapeHtml,
  formatDocumentType,
  highlightLexicalTerms,
  keywordSearch,
  rankSemantic,
} from "./search.js";

const els = {
  query: document.querySelector("#query"),
  searchButton: document.querySelector("#search-button"),
  topK: document.querySelector("#top-k"),
  category: document.querySelector("#category-filter"),
  documentType: document.querySelector("#type-filter"),
  results: document.querySelector("#results"),
  summary: document.querySelector("#search-summary"),
  modeBadge: document.querySelector("#mode-badge"),
  modelProgress: document.querySelector("#model-progress"),
  progressBar: document.querySelector("#progress-bar"),
  progressText: document.querySelector("#progress-text"),
  statDocuments: document.querySelector("#stat-documents"),
  statChunks: document.querySelector("#stat-chunks"),
  statAvgChunk: document.querySelector("#stat-avg-chunk"),
  evaluationStatus: document.querySelector("#evaluation-status"),
  metricRecall: document.querySelector("#metric-recall"),
  metricRecallDetail: document.querySelector("#metric-recall-detail"),
  metricMrr: document.querySelector("#metric-mrr"),
  metricMrrDetail: document.querySelector("#metric-mrr-detail"),
  metricCosine: document.querySelector("#metric-cosine"),
  metricCosineDetail: document.querySelector("#metric-cosine-detail"),
  metricLatency: document.querySelector("#metric-latency"),
  metricLatencyDetail: document.querySelector("#metric-latency-detail"),
};

const state = {
  chunks: [],
  vectors: [],
  metadata: null,
  mode: "loading",
  initializationError: null,
};

const CACHE_PREFIX = "semantic-search-vectors";

async function fetchJson(path) {
  const response = await fetch(`${path}${path.includes("?") ? "&" : "?"}v=20260727-3`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
  return response.json();
}

async function fetchOptionalJson(path) {
  try {
    const response = await fetch(`${path}${path.includes("?") ? "&" : "?"}v=20260727-3`, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn(`Optional evaluation artifact unavailable: ${path}`, error);
    return null;
  }
}

function asPercent(value, digits = 1) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "Unavailable";
}

function asNumber(value, digits = 4) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "Unavailable";
}

async function loadEvaluationMetrics() {
  const [recall, mrr, latency, cosine] = await Promise.all([
    fetchOptionalJson("./data/recall_at_k_results.json"),
    fetchOptionalJson("./data/mrr_results.json"),
    fetchOptionalJson("./data/query_latency_results.json"),
    fetchOptionalJson("./data/cosine_similarity_analysis.json"),
  ]);

  let completed = 0;

  if (recall?.status === "completed") {
    els.metricRecall.textContent = `R@1 ${asPercent(recall.recall_at_1)} · R@3 ${asPercent(recall.recall_at_3)}`;
    els.metricRecallDetail.textContent = `R@5 ${asPercent(recall.recall_at_5)} · R@10 ${asPercent(recall.recall_at_10)} · ${recall.query_count ?? "?"} queries`;
    completed += 1;
  } else {
    els.metricRecall.textContent = "Not available";
    els.metricRecallDetail.textContent = "Run evaluate_search.py and copy the resulting JSON into web/data.";
  }

  if (mrr?.status === "completed") {
    els.metricMrr.textContent = asNumber(mrr.mrr, 4);
    els.metricMrrDetail.textContent = "Mean reciprocal rank across the verified evaluation query set.";
    completed += 1;
  } else {
    els.metricMrr.textContent = "Not available";
    els.metricMrrDetail.textContent = "Run evaluate_search.py and copy the resulting JSON into web/data.";
  }

  if (latency?.status === "completed" && Array.isArray(latency.results) && latency.results.length) {
    const representative = latency.results.find((item) => Number(item.top_k) === 5) || latency.results[0];
    els.metricLatency.textContent = `${Number(representative.average_ms).toFixed(2)} ms average`;
    els.metricLatencyDetail.textContent = `Top-${representative.top_k} · min ${Number(representative.minimum_ms).toFixed(2)} ms · max ${Number(representative.maximum_ms).toFixed(2)} ms · ${representative.measurements} runs`;
    completed += 1;
  } else {
    els.metricLatency.textContent = "Not available";
    els.metricLatencyDetail.textContent = "Run benchmark_latency.py and copy the resulting JSON into web/data.";
  }

  if (cosine?.status === "completed") {
    const mean = cosine.mean_top_score ?? cosine.average_top_score ?? cosine.mean_similarity;
    els.metricCosine.textContent = Number.isFinite(Number(mean)) ? `${Number(mean).toFixed(3)} mean score` : "Analysis completed";
    els.metricCosineDetail.textContent = cosine.summary || cosine.note || "Review the cosine analysis artifact for score distributions and error cases.";
    completed += 1;
  } else {
    els.metricCosine.textContent = "Pending manual review";
    els.metricCosineDetail.textContent = "Recall, MRR, and latency are verified. Cosine false-positive analysis has not yet been completed.";
  }

  els.evaluationStatus.textContent = completed
    ? `${completed} evaluation artifact${completed === 1 ? "" : "s"} loaded from static JSON. Values shown below are generated results, not placeholders.`
    : "No completed evaluation artifacts were found. Run the offline evaluation and copy the JSON outputs into web/data.";
}

function cacheKey() {
  const version = state.metadata?.generated_at_utc || "sample";
  return `${CACHE_PREFIX}:${BROWSER_MODEL}:${version}:${state.chunks.length}`;
}

function loadCachedVectors() {
  try {
    const parsed = JSON.parse(localStorage.getItem(cacheKey()) || "null");
    if (Array.isArray(parsed) && parsed.length === state.chunks.length) return parsed;
  } catch (error) {
    console.warn("Ignoring invalid local vector cache", error);
  }
  return null;
}

function saveCachedVectors(vectors) {
  try {
    localStorage.setItem(cacheKey(), JSON.stringify(vectors));
  } catch (error) {
    console.warn("Browser storage is unavailable; continuing without local vector cache", error);
  }
}

function updateModelProgress(event) {
  els.modelProgress.hidden = false;
  const status = event?.status || "loading";
  const file = event?.file || event?.name || "model assets";
  const progress = Number.isFinite(event?.progress) ? Math.max(4, Math.min(100, event.progress)) : 8;
  els.progressBar.style.width = `${progress}%`;
  els.progressText.textContent = `${status.replaceAll("_", " ")}: ${file}`;
}

function setMode(mode, detail = "") {
  state.mode = mode;
  els.modeBadge.className = `status-badge ${mode}`;
  if (mode === "semantic") {
    els.modeBadge.textContent = detail || "Semantic mode · on-device";
  } else if (mode === "fallback") {
    els.modeBadge.textContent = "Keyword fallback · model unavailable";
  } else {
    els.modeBadge.textContent = detail || "Initializing browser model…";
  }
}

function populateFilters() {
  const categories = [...new Set(state.chunks.map((item) => item.project_category))].sort();
  const types = [...new Set(state.chunks.map((item) => item.document_type))].sort();
  for (const category of categories) {
    els.category.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`);
  }
  for (const type of types) {
    els.documentType.insertAdjacentHTML("beforeend", `<option value="${escapeHtml(type)}">${escapeHtml(formatDocumentType(type))}</option>`);
  }
}

function updateStats() {
  els.statDocuments.textContent = state.metadata.document_count.toLocaleString();
  els.statChunks.textContent = state.metadata.chunk_count.toLocaleString();
  els.statAvgChunk.textContent = `${Math.round(state.metadata.average_chunk_words)} words`;
}

async function initializeVectors(embeddingPayload) {
  const byId = new Map((embeddingPayload.embeddings || []).map((item) => [item.chunk_id, item.vector]));
  if (byId.size && state.chunks.every((chunk) => byId.has(chunk.chunk_id))) {
    state.vectors = state.chunks.map((chunk) => byId.get(chunk.chunk_id));
    setMode("semantic", "Semantic mode · precomputed vectors");
    return;
  }

  const cached = loadCachedVectors();
  if (cached) {
    state.vectors = cached;
    setMode("semantic", "Semantic mode · cached vectors");
    return;
  }

  setMode("loading", "Creating one-time browser index…");
  els.modelProgress.hidden = false;
  els.progressText.textContent = "Loading Sentence-BERT and embedding the sample corpus…";
  const batchSize = 6;
  const vectors = [];
  for (let start = 0; start < state.chunks.length; start += batchSize) {
    const batch = state.chunks.slice(start, start + batchSize).map((item) => item.text);
    const batchVectors = await embedTexts(batch, updateModelProgress);
    vectors.push(...batchVectors);
    const percent = Math.round((vectors.length / state.chunks.length) * 100);
    els.progressBar.style.width = `${percent}%`;
    els.progressText.textContent = `Creating local semantic index: ${vectors.length}/${state.chunks.length} chunks`;
  }
  state.vectors = vectors;
  saveCachedVectors(vectors);
  els.modelProgress.hidden = true;
  setMode("semantic", "Semantic mode · browser-generated vectors");
}

function renderResults(results, query) {
  if (!results.length) {
    els.results.innerHTML = '<div class="empty-state">No results matched the active filters. Try a broader query or select all categories.</div>';
    return;
  }

  els.results.innerHTML = results.map((result) => {
    const tags = (result.tags || []).slice(0, 5).map((tag) => `<span>#${escapeHtml(tag)}</span>`).join("");
    const sourceLink = String(result.url_or_local_path || "").startsWith("http")
      ? `<a href="${escapeHtml(result.url_or_local_path)}" target="_blank" rel="noreferrer">Open source ↗</a>`
      : `<span>${escapeHtml(result.source_file)}</span>`;
    return `
      <article class="result-card">
        <div class="result-rank">#${result.rank}</div>
        <div>
          <div class="result-topline">
            <div>
              <h3>${escapeHtml(result.project_name)}</h3>
              <div class="result-section">${escapeHtml(result.project_category)} · ${escapeHtml(result.section_title)}</div>
            </div>
            <div class="score" title="Model-based similarity signal, not a probability">${(result.similarity_score * 100).toFixed(1)}%</div>
          </div>
          <p class="result-text">${highlightLexicalTerms(result.text, query)}</p>
          <div class="result-meta">
            <span>${escapeHtml(formatDocumentType(result.document_type))}</span>
            <span>${escapeHtml(result.source_file)}</span>
            ${tags}
            ${sourceLink}
          </div>
        </div>
      </article>`;
  }).join("");
}

async function runSearch() {
  const query = els.query.value.trim();
  if (!query) {
    els.query.focus();
    return;
  }
  els.searchButton.disabled = true;
  els.searchButton.textContent = "Searching…";
  const totalStart = performance.now();
  let embeddingMs = 0;
  let rankingStart;
  let results;

  try {
    const options = {
      topK: Number(els.topK.value),
      category: els.category.value,
      documentType: els.documentType.value,
    };
    if (state.mode === "semantic" && state.vectors.length === state.chunks.length) {
      const embeddingStart = performance.now();
      const [queryVector] = await embedTexts([query], updateModelProgress);
      embeddingMs = performance.now() - embeddingStart;
      rankingStart = performance.now();
      results = rankSemantic(queryVector, state.chunks, state.vectors, options);
    } else {
      rankingStart = performance.now();
      results = keywordSearch(query, state.chunks, options);
    }
    const rankingMs = performance.now() - rankingStart;
    const totalMs = performance.now() - totalStart;
    renderResults(results, query);
    els.summary.hidden = false;
    els.summary.innerHTML = `
      <span><strong>${results.length}</strong> results</span>
      <span>Mode: <strong>${state.mode === "semantic" ? "Sentence-BERT semantic" : "keyword fallback"}</strong></span>
      <span>Embedding: <strong>${embeddingMs.toFixed(1)} ms</strong></span>
      <span>Ranking: <strong>${rankingMs.toFixed(1)} ms</strong></span>
      <span>Total: <strong>${totalMs.toFixed(1)} ms</strong></span>`;
  } catch (error) {
    console.error(error);
    state.initializationError = error;
    setMode("fallback");
    const options = { topK: Number(els.topK.value), category: els.category.value, documentType: els.documentType.value };
    results = keywordSearch(query, state.chunks, options);
    renderResults(results, query);
    els.summary.hidden = false;
    els.summary.innerHTML = `<span>Semantic inference failed. Results are labeled and returned using the keyword fallback.</span>`;
  } finally {
    els.modelProgress.hidden = true;
    els.searchButton.disabled = false;
    els.searchButton.textContent = "Search";
  }
}

async function initialize() {
  try {
    const [chunks, metadata, embeddingPayload] = await Promise.all([
      fetchJson("./data/document_chunks.json"),
      fetchJson("./data/metadata.json"),
      fetchJson("./data/embeddings.json"),
    ]);
    state.chunks = chunks;
    state.metadata = metadata;
    populateFilters();
    updateStats();
    await loadEvaluationMetrics();
    await initializeVectors(embeddingPayload);
  } catch (error) {
    console.error(error);
    state.initializationError = error;
    setMode("fallback");
    els.modelProgress.hidden = true;
  }
}

els.searchButton.addEventListener("click", runSearch);
els.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    els.query.value = button.dataset.query;
    runSearch();
  });
});

initialize();
