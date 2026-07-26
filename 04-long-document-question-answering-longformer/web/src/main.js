import './styles.css';

import { APP_CONFIG } from './config.js';
import { countWords, normalizeDocumentText } from './chunking.js';
import { readDocumentFile } from './document-parser.js';
import { escapeHtml } from './evidence.js';
import { answerLongDocument, loadBrowserModel } from './qa-engine.js';

const state = {
  sourceName: 'pasted-text',
  lastResult: null,
};

const element = (id) => document.getElementById(id);
const documentText = element('documentText');
const question = element('question');
const sampleSelect = element('sampleSelect');
const fileInput = element('fileInput');
const runtimeSelect = element('runtimeSelect');
const modelProgress = element('modelProgress');
const inferenceProgress = element('inferenceProgress');
const statusMessage = element('statusMessage');
const askButton = element('askButton');
const loadModelButton = element('loadModelButton');
const exportButton = element('exportButton');

function setStatus(message, kind = 'info') {
  statusMessage.textContent = message;
  statusMessage.dataset.kind = kind;
}

function setProgress(target, label, percent = null) {
  const progress = target.querySelector('progress');
  const text = target.querySelector('span');
  text.textContent = label;
  if (Number.isFinite(percent)) {
    progress.removeAttribute('indeterminate');
    progress.value = Math.max(0, Math.min(100, percent));
  } else {
    progress.value = 0;
    progress.setAttribute('indeterminate', 'true');
  }
}

function updateDocumentStats() {
  const text = normalizeDocumentText(documentText.value);
  element('inputCharacters').textContent = text.length.toLocaleString();
  element('inputWords').textContent = countWords(text).toLocaleString();
  element('sourceName').textContent = state.sourceName;
}

function renderResult(result) {
  element('answerOutput').textContent = result.answer;
  element('confidenceOutput').textContent = result.confidenceProxy.toFixed(6);
  element('supportingOutput').textContent = result.supportingParagraph;
  element('evidenceOutput').innerHTML = result.highlightedEvidenceHtml;
  element('metricChunks').textContent = result.totalChunks.toLocaleString();
  element('metricEvaluated').textContent = result.evaluatedChunks.toLocaleString();
  element('metricLatency').textContent = `${result.latencySeconds.toFixed(2)} s`;
  element('metricRuntime').textContent = result.runtime.toUpperCase();
  element('diagnosticsOutput').textContent = JSON.stringify({
    source_name: state.sourceName,
    browser_model: result.browserModelId,
    core_python_model: result.corePythonModelId,
    answer_character_span: [result.answerStart, result.answerEnd],
    paragraph_index: result.paragraphIndex,
    document_characters: result.documentCharacters,
    document_words: result.documentWords,
    total_chunks: result.totalChunks,
    evaluated_candidate_chunks: result.evaluatedChunks,
    browser_runtime: result.runtime,
    latency_seconds: Number(result.latencySeconds.toFixed(4)),
    warnings: result.warnings,
    top_candidates: result.candidateResults.map((item) => ({
      answer: item.answer,
      confidence_proxy: Number(item.confidenceProxy.toFixed(6)),
      retrieval_score: Number(item.retrievalScore.toFixed(4)),
      chunk_id: item.chunkId,
    })),
  }, null, 2);
  exportButton.disabled = false;
}

async function loadSamples() {
  const response = await fetch('./samples/index.json');
  if (!response.ok) throw new Error('Unable to load sample-document index.');
  const payload = await response.json();
  sampleSelect.innerHTML = '<option value="">Choose a sample document</option>';
  for (const item of payload.samples ?? []) {
    const option = document.createElement('option');
    option.value = item.file;
    option.textContent = item.name;
    option.dataset.question = item.question ?? '';
    sampleSelect.appendChild(option);
  }
}

async function useSelectedSample() {
  if (!sampleSelect.value) return;
  const response = await fetch(`./samples/${encodeURIComponent(sampleSelect.value)}`);
  if (!response.ok) throw new Error('Unable to load the selected sample document.');
  documentText.value = await response.text();
  question.value = sampleSelect.selectedOptions[0]?.dataset.question ?? '';
  state.sourceName = sampleSelect.value;
  updateDocumentStats();
  setStatus(`Loaded sample: ${sampleSelect.selectedOptions[0]?.textContent ?? sampleSelect.value}`);
}

async function useUploadedFile() {
  const file = fileInput.files?.[0];
  if (!file) return;
  setStatus(`Reading ${file.name}...`);
  const parsed = await readDocumentFile(file);
  documentText.value = parsed.text;
  state.sourceName = parsed.sourceName;
  sampleSelect.value = '';
  updateDocumentStats();
  setStatus(`Loaded ${parsed.sourceName}. Review the extracted text before asking a question.`, 'success');
}

async function loadModelOnly() {
  loadModelButton.disabled = true;
  setStatus('Loading the browser QA model. The first load downloads ONNX weights and may take time.');
  try {
    await loadBrowserModel({
      runtime: runtimeSelect.value,
      onProgress: ({ label, percent }) => setProgress(modelProgress, label, percent),
    });
    setProgress(modelProgress, 'Model ready', 100);
    setStatus('Browser QA model loaded and cached for this session.', 'success');
  } catch (error) {
    setStatus(`Model loading failed: ${error.message}`, 'error');
  } finally {
    loadModelButton.disabled = false;
  }
}

async function askDocument() {
  askButton.disabled = true;
  exportButton.disabled = true;
  element('answerOutput').textContent = 'Working...';
  element('confidenceOutput').textContent = '—';
  element('supportingOutput').textContent = '';
  element('evidenceOutput').innerHTML = '<div class="empty-evidence">Waiting for model output...</div>';
  setStatus('Preparing document chunks and loading the model.');

  try {
    const result = await answerLongDocument({
      question: question.value,
      documentText: documentText.value,
      chunkWords: Number(element('chunkWords').value),
      overlapWords: Number(element('overlapWords').value),
      candidateChunks: Number(element('candidateChunks').value),
      runtime: runtimeSelect.value,
      onModelProgress: ({ label, percent }) => setProgress(modelProgress, label, percent),
      onInferenceProgress: ({ completed, total, label }) => {
        const percent = total ? Math.round((completed / total) * 100) : null;
        setProgress(inferenceProgress, label, percent);
      },
    });
    state.lastResult = result;
    renderResult(result);
    setProgress(modelProgress, 'Model ready', 100);
    setProgress(inferenceProgress, 'Inference complete', 100);
    setStatus('Answer generated. Review the highlighted evidence and limitations before relying on it.', 'success');
  } catch (error) {
    element('answerOutput').textContent = 'Unable to answer the question.';
    element('diagnosticsOutput').textContent = JSON.stringify({ error: error.message }, null, 2);
    setStatus(error.message, 'error');
  } finally {
    askButton.disabled = false;
  }
}

function exportResult() {
  if (!state.lastResult) return;
  const payload = {
    exported_at: new Date().toISOString(),
    source_name: state.sourceName,
    question: question.value,
    ...state.lastResult,
    highlightedEvidenceHtml: undefined,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'long-document-qa-result.json';
  anchor.click();
  URL.revokeObjectURL(url);
}

function resetOutputs() {
  state.lastResult = null;
  element('answerOutput').textContent = 'No answer generated yet.';
  element('confidenceOutput').textContent = '—';
  element('supportingOutput').textContent = 'No supporting paragraph selected yet.';
  element('evidenceOutput').innerHTML = '<div class="empty-evidence">Highlighted evidence will appear here.</div>';
  element('diagnosticsOutput').textContent = '{}';
  for (const id of ['metricChunks', 'metricEvaluated', 'metricLatency', 'metricRuntime']) {
    element(id).textContent = '—';
  }
  exportButton.disabled = true;
}

function configureLinks() {
  const links = {
    githubLink: APP_CONFIG.githubUrl,
    gradioLink: APP_CONFIG.gradioSpaceUrl,
    staticLink: APP_CONFIG.staticSpaceUrl,
    modelCardLink: APP_CONFIG.modelCardUrl,
  };
  for (const [id, url] of Object.entries(links)) element(id).href = url;
  element('browserModelName').textContent = APP_CONFIG.browserModelId;
  element('pythonModelName').textContent = APP_CONFIG.pythonModelId;
}

function configureRuntime() {
  const webgpuOption = runtimeSelect.querySelector('option[value="webgpu"]');
  if (!('gpu' in navigator)) {
    webgpuOption.disabled = true;
    webgpuOption.textContent = 'WebGPU (not available in this browser)';
  }
}

sampleSelect.addEventListener('change', () => useSelectedSample().catch((error) => setStatus(error.message, 'error')));
fileInput.addEventListener('change', () => useUploadedFile().catch((error) => setStatus(error.message, 'error')));
documentText.addEventListener('input', () => {
  state.sourceName = 'pasted-text';
  updateDocumentStats();
});
loadModelButton.addEventListener('click', loadModelOnly);
askButton.addEventListener('click', askDocument);
exportButton.addEventListener('click', exportResult);
element('resetButton').addEventListener('click', resetOutputs);

configureLinks();
configureRuntime();
resetOutputs();
updateDocumentStats();
loadSamples()
  .then(() => setStatus('Choose a sample, upload a document, or paste text to begin.'))
  .catch((error) => setStatus(error.message, 'error'));

// Ensure user-supplied text is never interpolated as HTML outside the dedicated,
// escaped evidence renderer.
element('securityNote').innerHTML = escapeHtml('Uploaded documents remain in the browser session; this static app has no Python server.');
