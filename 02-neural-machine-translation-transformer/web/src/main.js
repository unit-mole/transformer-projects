import { resolveDirection, directionLabel } from './language-detection.js';
import { confidenceProxy } from './confidence.js';
import { parseCsv, toCsv, downloadText } from './csv.js';

const worker = new Worker('./src/translation.worker.js', { type: 'module' });
let requestCounter = 0;
const pending = new Map();
let csvRecords = [];
let batchOutput = [];

const byId = (id) => document.getElementById(id);
const sentenceStatus = byId('sentence-status');
const progressWrap = byId('progress-wrap');
const progressBar = byId('model-progress');
const progressText = byId('progress-text');
const progressValue = byId('progress-value');

function setSentenceStatus(message, kind = '') {
  sentenceStatus.textContent = message;
  sentenceStatus.className = `status ${kind}`.trim();
}

function setProgress(message, value = null) {
  progressWrap.hidden = false;
  progressText.textContent = message;
  if (Number.isFinite(value)) {
    progressBar.value = value;
    progressValue.textContent = `${Math.round(value)}%`;
  } else {
    progressBar.removeAttribute('value');
    progressValue.textContent = '';
  }
}

function hideProgress() {
  progressWrap.hidden = true;
  progressBar.value = 0;
  progressValue.textContent = '';
}

worker.addEventListener('message', (event) => {
  const message = event.data ?? {};
  if (message.type === 'progress') {
    const filename = message.file ? ` · ${message.file.split('/').at(-1)}` : '';
    setProgress(`${message.status ?? 'Loading'}${filename}`, message.progress);
    return;
  }
  if (message.type === 'model-status') {
    setProgress(message.message, null);
    return;
  }
  if (message.type === 'inference-status') {
    setProgress(message.message, null);
    return;
  }

  const job = pending.get(message.id);
  if (!job) return;
  pending.delete(message.id);
  if (message.type === 'translation-result') job.resolve(message);
  else job.reject(new Error(message.message || 'Translation failed'));
});

function requestTranslation(text, direction, options = {}) {
  const id = ++requestCounter;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({
      type: 'translate',
      id,
      text,
      direction,
      numBeams: options.numBeams ?? 4,
      maxNewTokens: options.maxNewTokens ?? 128,
    });
  });
}

function formatLanguage(language) {
  return ({ english: 'English', hindi: 'Hindi', mixed: 'Mixed', uncertain: 'Uncertain' })[language] ?? language;
}

function renderSentenceResult(source, directionInfo, result) {
  const proxy = confidenceProxy(source, result.translatedText, result.direction);
  byId('translated-text').value = result.translatedText;
  byId('metric-language').textContent = formatLanguage(directionInfo.detection.language);
  byId('metric-direction').textContent = directionLabel(result.direction);
  byId('metric-source-tokens').textContent = String(result.sourceTokenCount);
  byId('metric-target-tokens').textContent = String(result.targetTokenCount);
  byId('metric-latency').textContent = `${(result.latencyMs / 1000).toFixed(2)} s`;
  byId('metric-confidence').textContent = `${(proxy.score * 100).toFixed(0)}% · ${proxy.label}`;
  const sourcePreview = result.sourceTokens.join(' · ');
  const targetPreview = result.targetTokens.join(' · ');
  byId('token-preview').textContent = `Source: ${sourcePreview}\nTarget: ${targetPreview}`;
}

async function translateSentence() {
  const source = byId('source-text').value.trim();
  const selected = byId('direction').value;
  const info = resolveDirection(source, selected);
  if (!source) {
    setSentenceStatus('Enter text before translating.', 'error');
    return;
  }
  if (!info.direction) {
    setSentenceStatus(`Automatic detection returned ${formatLanguage(info.detection.language)}. Select a manual direction for mixed or uncertain text.`, 'error');
    return;
  }

  const button = byId('translate-button');
  button.disabled = true;
  byId('translated-text').value = '';
  setSentenceStatus(`Preparing ${directionLabel(info.direction)} model…`);
  setProgress('Preparing model…', null);
  try {
    const result = await requestTranslation(source, info.direction, {
      numBeams: Number(byId('beam-size').value),
      maxNewTokens: Number(byId('max-tokens').value),
    });
    renderSentenceResult(source, info, result);
    setSentenceStatus(`Translation completed locally with ${result.model}.`, 'success');
  } catch (error) {
    setSentenceStatus(`Translation failed: ${error.message}. Check browser memory, network access, or model availability.`, 'error');
  } finally {
    button.disabled = false;
    hideProgress();
  }
}

function activateTab(name) {
  document.querySelectorAll('.tab').forEach((button) => button.classList.toggle('active', button.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    const active = panel.id === `${name}-panel`;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  });
}

document.querySelectorAll('.tab').forEach((button) => button.addEventListener('click', () => activateTab(button.dataset.tab)));
byId('translate-button').addEventListener('click', translateSentence);
byId('source-text').addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') translateSentence();
});
byId('clear-button').addEventListener('click', () => {
  byId('source-text').value = '';
  byId('translated-text').value = '';
  setSentenceStatus('');
});

document.querySelectorAll('.example').forEach((button) => button.addEventListener('click', () => {
  if (button.dataset.example === 'hi') {
    byId('source-text').value = 'मशीन लर्निंग हमें डेटा में उपयोगी पैटर्न खोजने में मदद करती है।';
  } else {
    byId('source-text').value = 'Machine learning helps us discover useful patterns in data.';
  }
  byId('direction').value = 'auto';
}));

byId('csv-file').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const parsed = parseCsv(await file.text());
    csvRecords = parsed.records;
    const select = byId('csv-column');
    select.replaceChildren(...parsed.headers.map((header) => {
      const option = document.createElement('option');
      option.value = header;
      option.textContent = header;
      return option;
    }));
    byId('batch-button').disabled = !parsed.records.length || !parsed.headers.length;
    byId('batch-status').textContent = `Loaded ${parsed.records.length} rows and ${parsed.headers.length} columns. Up to 25 rows will be processed.`;
    renderBatchRows(parsed.records.slice(0, 5).map((record, index) => ({ index: index + 1, original_text: record[parsed.headers[0]], status: 'Preview' })));
  } catch (error) {
    byId('batch-status').textContent = `CSV could not be read: ${error.message}`;
  }
});

function renderBatchRows(rows) {
  const body = byId('batch-results');
  body.replaceChildren();
  if (!rows.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 8;
    cell.textContent = 'No rows available.';
    return;
  }
  for (const item of rows) {
    const row = body.insertRow();
    [item.index, item.original_text, item.detected_language ?? '—', item.translation_direction ?? '—', item.translated_text ?? '—', item.confidence_score ?? '—', item.latency_seconds ?? '—', item.status ?? '—'].forEach((value) => {
      const cell = row.insertCell();
      cell.textContent = String(value ?? '');
    });
  }
}

async function runBatch() {
  const column = byId('csv-column').value;
  const selectedDirection = byId('batch-direction').value;
  const rows = csvRecords.slice(0, 25);
  if (!column || !rows.length) return;

  const button = byId('batch-button');
  button.disabled = true;
  batchOutput = [];
  byId('batch-status').textContent = `Starting batch translation for ${rows.length} rows…`;

  for (let index = 0; index < rows.length; index += 1) {
    const source = String(rows[index][column] ?? '').trim();
    const info = resolveDirection(source, selectedDirection);
    const item = {
      index: index + 1,
      original_text: source,
      detected_language: formatLanguage(info.detection.language),
      translation_direction: directionLabel(info.direction),
      translated_text: '',
      confidence_score: '',
      latency_seconds: '',
      status: 'Pending',
    };

    if (!source) {
      item.status = 'Skipped: empty text';
    } else if (!info.direction) {
      item.status = 'Needs manual direction';
    } else {
      try {
        byId('batch-status').textContent = `Translating row ${index + 1} of ${rows.length}…`;
        const result = await requestTranslation(source, info.direction, { numBeams: 4, maxNewTokens: 128 });
        const proxy = confidenceProxy(source, result.translatedText, info.direction);
        item.translated_text = result.translatedText;
        item.confidence_score = proxy.score.toFixed(3);
        item.latency_seconds = (result.latencyMs / 1000).toFixed(3);
        item.status = 'Success';
      } catch (error) {
        item.status = `Error: ${error.message}`;
      }
    }
    batchOutput.push({ ...rows[index], ...item });
    renderBatchRows(batchOutput);
  }

  byId('batch-status').textContent = `Completed ${batchOutput.filter((row) => row.status === 'Success').length} of ${rows.length} rows.`;
  byId('download-results-button').disabled = !batchOutput.length;
  button.disabled = false;
}

byId('batch-button').addEventListener('click', runBatch);
byId('download-results-button').addEventListener('click', () => downloadText('project02_browser_translations.csv', toCsv(batchOutput), 'text/csv;charset=utf-8'));
byId('sample-csv-button').addEventListener('click', () => {
  const sample = 'text\r\n"Quality data supports better decisions."\r\n"गुणवत्ता डेटा बेहतर निर्णय लेने में मदद करता है।"\r\n';
  downloadText('sample_batch_translation.csv', sample, 'text/csv;charset=utf-8');
});

async function loadEvaluation() {
  try {
    const response = await fetch('./data/evaluation-results.json');
    if (!response.ok) return;
    const data = await response.json();
    const set = (id, value) => { byId(id).textContent = Number.isFinite(value) ? Number(value).toFixed(2) : 'Not run'; };
    set('eval-enhi-bleu', data?.en_hi?.sacrebleu);
    set('eval-enhi-chrf', data?.en_hi?.chrf);
    set('eval-hien-bleu', data?.hi_en?.sacrebleu);
    set('eval-hien-chrf', data?.hi_en?.chrf);
  } catch {
    // The static demo remains usable when optional evaluation data is absent.
  }
}
loadEvaluation();
