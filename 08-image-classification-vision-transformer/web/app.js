import { classifyImage } from './inference.js';
import { clearSensitivityOverlay, computePatchSensitivity, renderSensitivityOverlay } from './attention.js';
import { fileFromUrl, formatBytes, loadImage, validateImageFile } from './image_preprocessing.js';

const state = { metadata: null, file: null, objectUrl: null, image: null, lastResult: null, busy: false };
const $ = (id) => document.getElementById(id);
const elements = {
  input: $('image-input'), drop: $('drop-zone'), preview: $('image-preview'), placeholder: $('image-placeholder'),
  fileName: $('file-name'), fileSize: $('file-size'), predict: $('predict-button'), message: $('input-message'),
  modelStatus: $('model-status'), statusDetail: $('status-detail'), device: $('device-value'), empty: $('empty-result'),
  result: $('prediction-result'), topLabel: $('top-label'), topConfidence: $('top-confidence'), bars: $('probability-bars'),
  latency: $('latency-pill'), sensitivityButton: $('sensitivity-button'), sensitivityStatus: $('sensitivity-status'),
  sensitivityCanvas: $('sensitivity-canvas'), gallery: $('sample-gallery'), repo: $('repo-link')
};

async function loadMetadata() {
  const response = await fetch('./metadata.json');
  if (!response.ok) throw new Error('Could not load metadata.json. Serve the web folder over HTTP.');
  state.metadata = await response.json();
  elements.repo.href = state.metadata.links.repository;
}

function setStatus(title, detail, device = null) {
  elements.modelStatus.textContent = title;
  elements.statusDetail.textContent = detail;
  if (device) elements.device.textContent = device.toUpperCase();
}

async function selectFile(file) {
  try {
    validateImageFile(file, state.metadata.prediction.max_upload_mb);
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);
    state.image = await loadImage(state.objectUrl);
    elements.preview.src = state.objectUrl;
    elements.preview.style.display = 'block';
    elements.placeholder.style.display = 'none';
    elements.fileName.textContent = file.name || 'Selected image';
    elements.fileSize.textContent = formatBytes(file.size);
    elements.predict.disabled = false;
    elements.sensitivityButton.disabled = true;
    elements.message.textContent = 'Ready for inference.';
    elements.sensitivityStatus.textContent = '';
    clearSensitivityOverlay(elements.sensitivityCanvas);
    state.lastResult = null;
  } catch (error) {
    elements.message.textContent = error.message;
  }
}

function renderPredictions(result) {
  const predictions = result.predictions;
  if (!Array.isArray(predictions) || !predictions.length) throw new Error('The model returned no predictions.');
  const top = predictions[0];
  elements.empty.hidden = true;
  elements.result.hidden = false;
  elements.topLabel.textContent = top.label;
  elements.topConfidence.textContent = `${(top.score * 100).toFixed(2)}% confidence`;
  elements.latency.textContent = `Latency ${result.latencyMs.toFixed(1)} ms`;
  elements.bars.innerHTML = '';
  predictions.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'probability-row';
    row.innerHTML = `<span title="${item.label}">${item.label}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(item.score * 100, 0.5)}%"></div></div><span class="probability-score">${(item.score * 100).toFixed(2)}%</span>`;
    elements.bars.append(row);
  });
}

async function predict(source = state.objectUrl, updateUi = true) {
  if (!source || state.busy) return null;
  if (updateUi) {
    state.busy = true;
    elements.predict.disabled = true;
    elements.sensitivityButton.disabled = true;
    elements.message.textContent = 'Preparing model…';
    setStatus('Loading', 'The first model load can take longer.');
  }
  try {
    const result = await classifyImage(source, state.metadata, (message) => {
      if (updateUi) { elements.message.textContent = message; setStatus('Loading', message); }
    });
    if (updateUi) {
      state.lastResult = result;
      renderPredictions(result);
      setStatus('Ready', 'Inference is running locally in this browser.', result.device);
      elements.message.textContent = 'Prediction complete.';
      elements.sensitivityButton.disabled = false;
    }
    return result;
  } catch (error) {
    if (updateUi) {
      console.error(error);
      setStatus('Load failed', 'Check the browser console and network connection.');
      elements.message.textContent = error.message || 'Inference failed.';
    }
    throw error;
  } finally {
    if (updateUi) { state.busy = false; elements.predict.disabled = !state.file; }
  }
}

async function generateSensitivity() {
  if (!state.lastResult || !state.image || state.busy) return;
  state.busy = true;
  elements.predict.disabled = true;
  elements.sensitivityButton.disabled = true;
  const target = state.lastResult.predictions[0];
  try {
    const sensitivity = await computePatchSensitivity({
      image: state.image,
      gridSize: state.metadata.explainability.grid_size,
      targetLabel: target.label,
      baselineScore: target.score,
      classify: (canvas) => classifyImage(canvas, state.metadata),
      onProgress: (done, total) => { elements.sensitivityStatus.textContent = `Running masked inference ${done}/${total}…`; }
    });
    renderSensitivityOverlay(elements.sensitivityCanvas, elements.preview, sensitivity);
    elements.sensitivityStatus.textContent = 'Patch sensitivity complete. Redder patches produced a larger drop in the selected class score when masked. This is not raw attention.';
  } catch (error) {
    console.error(error);
    elements.sensitivityStatus.textContent = error.message || 'Sensitivity analysis failed.';
  } finally {
    state.busy = false;
    elements.predict.disabled = false;
    elements.sensitivityButton.disabled = false;
  }
}

function wireEvents() {
  elements.input.addEventListener('change', (event) => event.target.files[0] && selectFile(event.target.files[0]));
  elements.drop.addEventListener('dragover', (event) => { event.preventDefault(); elements.drop.classList.add('dragover'); });
  elements.drop.addEventListener('dragleave', () => elements.drop.classList.remove('dragover'));
  elements.drop.addEventListener('drop', (event) => { event.preventDefault(); elements.drop.classList.remove('dragover'); if (event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]); });
  elements.predict.addEventListener('click', () => predict());
  elements.sensitivityButton.addEventListener('click', generateSensitivity);
  window.addEventListener('resize', () => state.lastResult && clearSensitivityOverlay(elements.sensitivityCanvas));
}

async function buildSamples() {
  const samples = [
    { name: 'Geometric bird', file: 'sample_bird.svg' },
    { name: 'Stylized car', file: 'sample_car.svg' },
    { name: 'Flower study', file: 'sample_flower.svg' }
  ];
  for (const sample of samples) {
    const button = document.createElement('button');
    button.className = 'sample-button';
    button.type = 'button';
    button.innerHTML = `<img src="./sample_images/${sample.file}" alt="${sample.name}"><span>${sample.name}</span>`;
    button.addEventListener('click', async () => {
      const file = await fileFromUrl(`./sample_images/${sample.file}`, sample.file);
      await selectFile(file);
    });
    elements.gallery.append(button);
  }
}

async function init() {
  try {
    await loadMetadata();
    wireEvents();
    await buildSamples();
    setStatus('Not loaded', state.metadata.browser_model.first_load_note);
  } catch (error) {
    console.error(error);
    setStatus('Configuration error', error.message);
    elements.message.textContent = error.message;
  }
}

init();
