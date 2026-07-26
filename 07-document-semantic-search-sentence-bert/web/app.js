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
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
  return response.json();
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
