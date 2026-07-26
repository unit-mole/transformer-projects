import { DEPLOYMENT, MODEL_IDS } from "./constants.js";
import { loadDemoData } from "./data-loader.js";
import { downloadJson } from "./export-results.js";
import { calculateQueryMetrics } from "./metrics.js";
import { BrowserRankingEngine } from "./ranking-engine.js";
import {
  clearError,
  completeProgress,
  elements,
  hideProgressSoon,
  populateSampleQueries,
  renderSearchResults,
  resolveQuery,
  selectedMode,
  setBusy,
  setRuntimeStatus,
  showError,
  syncModeControls,
  syncSliderOutputs,
  updateProgress,
} from "./ui.js";

const state = {
  data: null,
  engine: null,
  lastExport: null,
  isBusy: false,
};

function setApplicationBusy(isBusy) {
  state.isBusy = isBusy;
  setBusy(isBusy);
}

function buildExportPayload({
  query,
  mode,
  candidateK,
  rerankK,
  response,
  quality,
}) {
  return {
    generated_at: new Date().toISOString(),
    deployment: DEPLOYMENT,
    models: MODEL_IDS,
    query,
    settings: {
      mode,
      candidate_k: candidateK,
      rerank_k: mode === "two-stage" ? rerankK : 0,
    },
    latency_ms: Object.fromEntries(
      Object.entries(response.latency).map(
        ([key, value]) => [
          key,
          Number.isFinite(value)
            ? Math.round(value * 1000) / 1000
            : null,
        ],
      ),
    ),
    quality,
    bi_encoder_results: response.candidates.map(
      (document) => ({
        retrieval_rank: document.retrieval_rank,
        document_id: document.document_id,
        title: document.title,
        category: document.category,
        bi_encoder_score:
          Math.round(document.bi_encoder_score * 1_000_000) /
          1_000_000,
      }),
    ),
    cross_encoder_results:
      response.rerankedResults.map((document) => ({
        reranked_rank: document.reranked_rank,
        retrieval_rank: document.retrieval_rank,
        rank_movement: document.rank_movement,
        document_id: document.document_id,
        title: document.title,
        cross_encoder_score:
          Math.round(
            document.cross_encoder_score * 1_000_000,
          ) / 1_000_000,
      })),
  };
}

async function executeSearch(event) {
  event.preventDefault();

  if (state.isBusy || !state.data || !state.engine) {
    return;
  }

  clearError();
  setApplicationBusy(true);

  try {
    const query = resolveQuery(state.data.queryById);
    const mode = selectedMode();
    const candidateK = Number(elements.candidateK.value);
    const rerankK = Math.min(
      Number(elements.rerankK.value),
      candidateK,
    );

    setRuntimeStatus(
      "loading",
      "Preparing browser inference",
    );
    elements.progressArea.hidden = false;
    elements.progressLabel.textContent =
      "Preparing browser inference…";
    elements.progressPercent.textContent = "0%";
    elements.progressBar.style.width = "4%";

    const response = await state.engine.search({
      query: query.text,
      candidateK,
      rerankK,
      mode,
    });

    const quality = calculateQueryMetrics({
      queryId: query.queryId,
      qrelsByQuery: state.data.qrelsByQuery,
      candidates: response.candidates,
      rerankedResults: response.rerankedResults,
      candidateK,
      mode,
    });

    renderSearchResults({
      query: query.text,
      mode,
      candidateK,
      candidates: response.candidates,
      rerankedResults: response.rerankedResults,
      latency: response.latency,
      quality,
    });

    state.lastExport = buildExportPayload({
      query,
      mode,
      candidateK,
      rerankK,
      response,
      quality,
    });

    completeProgress("Search complete");
    setRuntimeStatus("ready", "Models ready in browser");
  } catch (error) {
    showError(error);
  } finally {
    setApplicationBusy(false);
    syncModeControls();
    hideProgressSoon();
  }
}

function exportResults() {
  try {
    downloadJson(state.lastExport);
  } catch (error) {
    showError(error);
  }
}

async function initialize() {
  try {
    setRuntimeStatus("loading", "Loading sample data");
    state.data = await loadDemoData();

    state.engine = new BrowserRankingEngine(
      state.data.documents,
      (event, stageName) =>
        updateProgress(event, stageName),
    );

    populateSampleQueries(
      state.data.queries,
      state.data.documents.length,
    );
    setRuntimeStatus(
      "idle",
      "Models load on first search",
    );
  } catch (error) {
    const localFileHint =
      window.location.protocol === "file:"
        ? " Run the Vite development server instead of opening index.html directly."
        : "";

    showError(
      new Error(
        `${error instanceof Error ? error.message : String(error)}${localFileHint}`,
      ),
    );
  }
}

elements.candidateK.addEventListener(
  "input",
  syncSliderOutputs,
);
elements.rerankK.addEventListener(
  "input",
  syncSliderOutputs,
);
document
  .querySelectorAll('input[name="searchMode"]')
  .forEach((input) =>
    input.addEventListener("change", syncModeControls),
  );
elements.form.addEventListener("submit", executeSearch);
elements.exportButton.addEventListener(
  "click",
  exportResults,
);

syncModeControls();
initialize();
