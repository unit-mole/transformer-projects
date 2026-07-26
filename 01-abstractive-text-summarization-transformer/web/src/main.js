import './styles.css';
import { SAMPLES } from './samples.js';
import { SummarizerClient } from './summarizer-client.js';
import {
  buildDownloadFileName,
  countCharacters,
  countWords,
  formatCompressionRatio,
  formatDuration,
  normalizeText,
  validateGenerationSettings,
} from './text-utils.js';

const elements = {
  input: document.querySelector('#article-input'),
  sampleSelect: document.querySelector('#sample-select'),
  sourceWords: document.querySelector('#source-words'),
  sourceCharacters: document.querySelector('#source-characters'),
  sourceTokens: document.querySelector('#source-tokens'),
  tokenPreview: document.querySelector('#token-preview-text'),
  runtimeSelect: document.querySelector('#runtime-select'),
  minTokens: document.querySelector('#min-tokens'),
  minTokenOutput: document.querySelector('#min-token-output'),
  maxTokens: document.querySelector('#max-tokens'),
  maxTokenOutput: document.querySelector('#max-token-output'),
  numBeams: document.querySelector('#num-beams'),
  beamOutput: document.querySelector('#beam-output'),
  lengthPenalty: document.querySelector('#length-penalty'),
  penaltyOutput: document.querySelector('#penalty-output'),
  noRepeatNgram: document.querySelector('#no-repeat-ngram'),
  ngramOutput: document.querySelector('#ngram-output'),
  longDocumentMode: document.querySelector('#long-document-mode'),
  loadModelButton: document.querySelector('#load-model-button'),
  generateButton: document.querySelector('#generate-button'),
  compareButton: document.querySelector('#compare-button'),
  modelState: document.querySelector('#model-state'),
  progressPanel: document.querySelector('#progress-panel'),
  progressLabel: document.querySelector('#progress-label'),
  progressPercent: document.querySelector('#progress-percent'),
  progress: document.querySelector('#model-progress'),
  progressDetail: document.querySelector('#progress-detail'),
  errorPanel: document.querySelector('#error-panel'),
  summaryOutput: document.querySelector('#summary-output'),
  copyButton: document.querySelector('#copy-button'),
  downloadButton: document.querySelector('#download-button'),
  latency: document.querySelector('#latency-metric'),
  compression: document.querySelector('#compression-metric'),
  summaryWords: document.querySelector('#summary-words-metric'),
  chunks: document.querySelector('#chunks-metric'),
  runtime: document.querySelector('#runtime-metric'),
  dtype: document.querySelector('#dtype-metric'),
  beamComparison: document.querySelector('#beam-comparison'),
  beamOneSummary: document.querySelector('#beam-one-summary'),
  beamOneLatency: document.querySelector('#beam-one-latency'),
  selectedBeamTitle: document.querySelector('#selected-beam-title'),
  selectedBeamSummary: document.querySelector('#selected-beam-summary'),
  selectedBeamLatency: document.querySelector('#selected-beam-latency'),
  evaluationStatus: document.querySelector('#evaluation-status'),
};

let currentSummary = '';
let modelLoadedForRuntime = null;
let activeOperation = false;

function populateSamples() {
  for (const sample of SAMPLES) {
    const option = document.createElement('option');
    option.value = sample.id;
    option.textContent = sample.title;
    elements.sampleSelect.append(option);
  }
}

function updateTextStats() {
  const text = elements.input.value;
  elements.sourceWords.textContent = String(countWords(text));
  elements.sourceCharacters.textContent = String(countCharacters(text));
  if (!text.trim()) elements.sourceTokens.textContent = '—';
}

function updateControlOutputs() {
  elements.minTokenOutput.value = elements.minTokens.value;
  elements.maxTokenOutput.value = elements.maxTokens.value;
  elements.beamOutput.value = elements.numBeams.value;
  elements.penaltyOutput.value = Number(elements.lengthPenalty.value).toFixed(1);
  elements.ngramOutput.value = elements.noRepeatNgram.value;
}

function getSettings() {
  return validateGenerationSettings({
    minNewTokens: Number(elements.minTokens.value),
    maxNewTokens: Number(elements.maxTokens.value),
    numBeams: Number(elements.numBeams.value),
    lengthPenalty: Number(elements.lengthPenalty.value),
    noRepeatNgramSize: Number(elements.noRepeatNgram.value),
  });
}

function setBusy(busy, label = 'Working…') {
  activeOperation = busy;
  elements.loadModelButton.disabled = busy;
  elements.generateButton.disabled = busy;
  elements.compareButton.disabled = busy;
  elements.generateButton.textContent = busy ? label : 'Generate summary';
}

function setStatus({ state, message }) {
  elements.modelState.textContent = message;
  elements.modelState.className = `status-pill status-${state}`;
}

function showError(error) {
  elements.errorPanel.hidden = false;
  elements.errorPanel.textContent = error instanceof Error ? error.message : String(error);
}

function clearError() {
  elements.errorPanel.hidden = true;
  elements.errorPanel.textContent = '';
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function handleProgress(info) {
  elements.progressPanel.hidden = false;
  const value = Number.isFinite(info.progress) ? info.progress : 0;
  elements.progress.value = value;
  elements.progressPercent.textContent = Number.isFinite(info.progress) ? `${Math.round(value)}%` : '…';
  elements.progressLabel.textContent = info.status === 'ready' ? 'Model ready' : 'Downloading model files';
  const sizeText = info.loaded !== null && info.total !== null
    ? `${formatBytes(info.loaded)} of ${formatBytes(info.total)}`
    : '';
  elements.progressDetail.textContent = [info.file, sizeText].filter(Boolean).join(' · ') ||
    'The first model load downloads quantized ONNX weights and stores them in the browser cache.';
}

const client = new SummarizerClient({
  onProgress: handleProgress,
  onStatus: (status) => {
    setStatus(status);
    if (status.state === 'ready') {
      elements.progress.value = 100;
      elements.progressPercent.textContent = '100%';
      window.setTimeout(() => {
        if (!activeOperation) elements.progressPanel.hidden = true;
      }, 1200);
    }
  },
});

async function ensureModelLoaded() {
  const runtime = elements.runtimeSelect.value;
  if (modelLoadedForRuntime === runtime) return;
  setBusy(true, 'Loading model…');
  clearError();
  elements.progressPanel.hidden = false;
  try {
    const info = await client.load(runtime);
    modelLoadedForRuntime = runtime;
    elements.runtime.textContent = info.runtime.toUpperCase();
    elements.dtype.textContent = info.dtype;
  } catch (error) {
    showError(error);
    setStatus({ state: 'error', message: 'Model load failed' });
    throw error;
  } finally {
    setBusy(false);
  }
}

function renderResult(result) {
  currentSummary = result.summary;
  elements.summaryOutput.textContent = result.summary;
  elements.summaryOutput.classList.remove('summary-placeholder');
  elements.latency.textContent = formatDuration(result.latencyMs);
  elements.compression.textContent = formatCompressionRatio(elements.input.value, result.summary);
  elements.summaryWords.textContent = String(countWords(result.summary));
  elements.chunks.textContent = result.truncatedToFirstChunk
    ? `${result.chunks} (first chunk only)`
    : result.secondPass
      ? `${result.chunks} + combine`
      : String(result.chunks);
  elements.runtime.textContent = result.runtime.toUpperCase();
  elements.dtype.textContent = result.dtype;
  elements.sourceTokens.textContent = String(result.inputTokens);
  elements.tokenPreview.textContent = result.inputTokenPreview.length
    ? `First ${result.inputTokenPreview.length} token IDs: ${result.inputTokenPreview.join(', ')}`
    : 'Token IDs are unavailable for this runtime.';
  elements.copyButton.disabled = false;
  elements.downloadButton.disabled = false;
}

async function generateSummary() {
  clearError();
  const text = normalizeText(elements.input.value);
  if (countWords(text) < 25) {
    showError('Enter at least 25 words so the model has enough context to summarize.');
    return;
  }

  let settings;
  try {
    settings = getSettings();
  } catch (error) {
    showError(error);
    return;
  }

  setBusy(true, 'Generating…');
  elements.beamComparison.hidden = true;
  try {
    await ensureModelLoaded();
    setBusy(true, 'Generating…');
    const result = await client.summarize({
      text,
      settings,
      runtimePreference: elements.runtimeSelect.value,
      longDocumentMode: elements.longDocumentMode.checked,
    });
    renderResult(result);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function compareBeams() {
  clearError();
  const text = normalizeText(elements.input.value);
  if (countWords(text) < 25) {
    showError('Enter at least 25 words before running the beam comparison.');
    return;
  }

  let settings;
  try {
    settings = getSettings();
  } catch (error) {
    showError(error);
    return;
  }

  if (settings.numBeams === 1) {
    showError('Select a beam count greater than 1 to compare it with greedy decoding.');
    return;
  }

  setBusy(true, 'Comparing…');
  try {
    await ensureModelLoaded();
    setBusy(true, 'Comparing…');
    const result = await client.compareBeams({
      text,
      settings,
      runtimePreference: elements.runtimeSelect.value,
      longDocumentMode: elements.longDocumentMode.checked,
    });
    elements.beamOneSummary.textContent = result.greedy.summary;
    elements.beamOneLatency.textContent = formatDuration(result.greedy.latencyMs);
    elements.selectedBeamTitle.textContent = `Beam search — beam ${settings.numBeams}`;
    elements.selectedBeamSummary.textContent = result.selected.summary;
    elements.selectedBeamLatency.textContent = formatDuration(result.selected.latencyMs);
    elements.beamComparison.hidden = false;
    elements.beamComparison.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function copySummary() {
  if (!currentSummary) return;
  try {
    await navigator.clipboard.writeText(currentSummary);
    const previous = elements.copyButton.textContent;
    elements.copyButton.textContent = 'Copied';
    window.setTimeout(() => { elements.copyButton.textContent = previous; }, 1400);
  } catch {
    showError('Clipboard access was blocked. Select the summary text and copy it manually.');
  }
}

function downloadSummary() {
  if (!currentSummary) return;
  const content = [
    'Abstractive Text Summarization Transformer',
    '',
    currentSummary,
    '',
    `Generated with: Xenova/distilbart-cnn-12-6`,
    `Runtime: ${elements.runtime.textContent}`,
    `Quantization: ${elements.dtype.textContent}`,
    `Latency: ${elements.latency.textContent}`,
    `Compression ratio: ${elements.compression.textContent}`,
    '',
    'Review the summary against the source text before use.',
  ].join('\n');
  const url = URL.createObjectURL(new Blob([content], { type: 'text/plain;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = buildDownloadFileName();
  link.click();
  URL.revokeObjectURL(url);
}

async function loadEvaluationStatus() {
  try {
    const response = await fetch('./evaluation-results.json');
    if (!response.ok) throw new Error('Evaluation file unavailable.');
    const data = await response.json();
    const status = data.status === 'completed' ? 'Completed' : 'Not run';
    elements.evaluationStatus.textContent = `${status} · ROUGE and BERTScore values are published only after an actual Python evaluation run.`;
    elements.evaluationStatus.classList.add(data.status === 'completed' ? 'evaluation-complete' : 'evaluation-pending');
  } catch (error) {
    elements.evaluationStatus.textContent = error.message;
    elements.evaluationStatus.classList.add('evaluation-pending');
  }
}

function bindEvents() {
  elements.input.addEventListener('input', updateTextStats);
  elements.sampleSelect.addEventListener('change', () => {
    const sample = SAMPLES.find((item) => item.id === elements.sampleSelect.value);
    if (sample) {
      elements.input.value = sample.text;
      updateTextStats();
      elements.input.focus();
    }
  });

  for (const input of [elements.minTokens, elements.maxTokens, elements.numBeams, elements.lengthPenalty, elements.noRepeatNgram]) {
    input.addEventListener('input', updateControlOutputs);
  }

  elements.runtimeSelect.addEventListener('change', () => {
    client.reset();
    modelLoadedForRuntime = null;
    setStatus({ state: 'idle', message: 'Model reload required' });
  });

  elements.loadModelButton.addEventListener('click', async () => {
    try {
      await ensureModelLoaded();
    } catch {
      // Error is already displayed by ensureModelLoaded.
    }
  });
  elements.generateButton.addEventListener('click', generateSummary);
  elements.compareButton.addEventListener('click', compareBeams);
  elements.copyButton.addEventListener('click', copySummary);
  elements.downloadButton.addEventListener('click', downloadSummary);
}

populateSamples();
updateTextStats();
updateControlOutputs();
bindEvents();
void loadEvaluationStatus();
