import { cleanText, parseCandidateLabels, validateImageFile } from './clip_preprocessing.js';
import { ClipRuntime, CLIP_CONFIG } from './clip_inference.js';
import { captionBaseline, parseBundledEmbeddings, rankByEmbedding } from './retrieval.js';
import { classifyZeroShot } from './zero_shot.js';

const state = {
  gallery: [],
  metadata: null,
  bundledEmbeddings: null,
  galleryEmbeddings: null,
  runtime: null,
  uploadedFile: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function setStatus(message, tone = 'neutral') {
  const element = $('#model-status');
  element.textContent = message;
  element.dataset.tone = tone;
}

function setProgress(event) {
  if (!event) return;
  if (event.status === 'progress' && Number.isFinite(event.progress)) {
    setStatus(`Loading ${event.file ?? 'model'} — ${Math.round(event.progress)}%`, 'loading');
  } else if (event.status === 'ready') {
    setStatus(`${event.file} ready`, 'success');
  } else if (event.status === 'initiate') {
    setStatus(`Preparing ${event.file ?? 'model'}…`, 'loading');
  }
}

function getRuntime() {
  if (!state.runtime) state.runtime = new ClipRuntime({ progressCallback: setProgress });
  return state.runtime;
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '—';
}

function formatMs(value) {
  return `${Math.round(value)} ms`;
}

function showMessage(target, message, tone = 'info') {
  const box = $(target);
  box.textContent = message;
  box.className = `message ${tone}`;
  box.hidden = false;
}

function hideMessage(target) {
  $(target).hidden = true;
}

function renderResults(results, mode, elapsed) {
  const container = $('#results-grid');
  container.innerHTML = '';
  $('#retrieval-mode').textContent = mode;
  $('#retrieval-latency').textContent = formatMs(elapsed);
  for (const item of results) {
    const card = document.createElement('article');
    card.className = 'result-card';
    const tags = item.tags.map((tag) => `<span>${tag}</span>`).join('');
    card.innerHTML = `
      <div class="image-wrap">
        <img src="${item.image_path}" alt="${item.caption}" loading="lazy">
        <span class="rank-badge">#${item.rank}</span>
      </div>
      <div class="result-body">
        <div class="score-row"><strong>${formatScore(item.score)}</strong><span>${mode === 'CLIP' ? 'cosine similarity' : 'baseline score'}</span></div>
        <p>${item.caption}</p>
        <div class="meta-row"><span>${item.category}</span><span>${item.image_id}</span></div>
        <div class="tag-list">${tags}</div>
      </div>`;
    container.appendChild(card);
  }
}

async function ensureGalleryEmbeddings() {
  if (state.galleryEmbeddings) return state.galleryEmbeddings;
  const bundled = parseBundledEmbeddings(state.bundledEmbeddings, state.gallery);
  if (bundled) {
    state.galleryEmbeddings = bundled;
    setStatus('Using bundled CLIP gallery embeddings', 'success');
    return bundled;
  }
  const cached = ClipRuntime.loadCachedGallery(state.gallery);
  if (cached) {
    state.galleryEmbeddings = cached;
    setStatus('Using browser-cached CLIP gallery embeddings', 'success');
    return cached;
  }
  const runtime = getRuntime();
  setStatus('Building gallery embeddings in this browser…', 'loading');
  const generated = await runtime.encodeGallery(state.gallery, ({ index, total }) => {
    setStatus(`Embedding gallery image ${index} of ${total}…`, 'loading');
  });
  ClipRuntime.saveCachedGallery(generated);
  state.galleryEmbeddings = generated;
  setStatus('CLIP gallery embeddings ready and cached', 'success');
  return generated;
}

async function runRetrieval() {
  hideMessage('#retrieval-message');
  const button = $('#search-button');
  button.disabled = true;
  const started = performance.now();
  try {
    const query = cleanText($('#search-query').value, { fieldName: 'Search query', maxLength: 240 });
    const topK = Number($('#top-k').value);
    const runtime = getRuntime();
    const galleryEmbeddings = await ensureGalleryEmbeddings();
    setStatus('Encoding query with CLIP text encoder…', 'loading');
    const [queryEmbedding] = await runtime.encodeText(query);
    const results = rankByEmbedding(queryEmbedding, state.gallery, galleryEmbeddings, topK);
    renderResults(results, 'CLIP', performance.now() - started);
    setStatus('CLIP retrieval ready', 'success');
  } catch (error) {
    console.error(error);
    try {
      const query = cleanText($('#search-query').value, { fieldName: 'Search query', maxLength: 240 });
      const results = captionBaseline(query, state.gallery, Number($('#top-k').value));
      renderResults(results, 'Caption TF-IDF baseline', performance.now() - started);
      setStatus('CLIP unavailable — using labeled caption baseline', 'warning');
      showMessage('#retrieval-message', `CLIP could not initialize in this browser. A caption-search baseline is shown instead. Details: ${error.message}`, 'warning');
    } catch (fallbackError) {
      showMessage('#retrieval-message', fallbackError.message, 'error');
      setStatus('Input or model error', 'error');
    }
  } finally {
    button.disabled = false;
  }
}

function renderPredictions(predictions, elapsed) {
  const body = $('#prediction-body');
  body.innerHTML = '';
  for (const item of predictions) {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${item.rank}</td><td><strong>${item.label}</strong><small>${item.prompt}</small></td><td>${formatScore(item.score)}</td><td>${(item.probability * 100).toFixed(1)}%</td>`;
    body.appendChild(row);
  }
  $('#classification-latency').textContent = formatMs(elapsed);
  $('#prediction-summary').textContent = predictions.length ? `Top prediction: ${predictions[0].label}` : 'No prediction';
}

async function runClassification() {
  hideMessage('#classification-message');
  const button = $('#classify-button');
  button.disabled = true;
  const started = performance.now();
  try {
    const file = validateImageFile(state.uploadedFile);
    const labels = parseCandidateLabels($('#candidate-labels').value);
    const runtime = getRuntime();
    setStatus('Running CLIP zero-shot classification…', 'loading');
    const predictions = await classifyZeroShot(runtime, file, labels, state.metadata.prompt_template);
    renderPredictions(predictions, performance.now() - started);
    setStatus('Zero-shot classification ready', 'success');
  } catch (error) {
    console.error(error);
    showMessage('#classification-message', error.message, 'error');
    setStatus('Classification error', 'error');
  } finally {
    button.disabled = false;
  }
}

function wireTabs() {
  $$('.tab-button').forEach((button) => {
    button.addEventListener('click', () => {
      $$('.tab-button').forEach((item) => item.setAttribute('aria-selected', 'false'));
      $$('.tab-panel').forEach((panel) => panel.hidden = true);
      button.setAttribute('aria-selected', 'true');
      $(`#${button.dataset.target}`).hidden = false;
    });
  });
}

function wireInputs() {
  $('#search-button').addEventListener('click', runRetrieval);
  $('#search-query').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') runRetrieval();
  });
  $('#top-k').addEventListener('input', (event) => $('#top-k-value').textContent = event.target.value);
  $$('.sample-query').forEach((button) => button.addEventListener('click', () => {
    $('#search-query').value = button.dataset.query;
    runRetrieval();
  }));
  $('#image-upload').addEventListener('change', (event) => {
    const [file] = event.target.files;
    state.uploadedFile = file ?? null;
    if (!file) return;
    try {
      validateImageFile(file);
      $('#upload-preview').src = URL.createObjectURL(file);
      $('#upload-preview').hidden = false;
      $('#upload-placeholder').hidden = true;
      hideMessage('#classification-message');
    } catch (error) {
      state.uploadedFile = null;
      showMessage('#classification-message', error.message, 'error');
    }
  });
  $('#classify-button').addEventListener('click', runClassification);
}

function renderStats() {
  $('#gallery-count').textContent = state.gallery.length;
  $('#category-count').textContent = new Set(state.gallery.map((item) => item.category)).size;
  $('#embedding-dim').textContent = state.metadata.embedding_dimension;
  $('#model-name').textContent = state.metadata.base_model;
  $('#runtime-detail').textContent = `Transformers.js + ONNX Runtime Web · ${CLIP_CONFIG.dtype} · ${CLIP_CONFIG.device}`;
}

async function initialize() {
  wireTabs();
  wireInputs();
  try {
    const [galleryPayload, metadata, bundledEmbeddings] = await Promise.all([
      fetch('./data/image_gallery.json').then((response) => response.json()),
      fetch('./metadata.json').then((response) => response.json()),
      fetch('./data/image_embeddings.json').then((response) => response.json()),
    ]);
    state.gallery = galleryPayload.images;
    state.metadata = metadata;
    state.bundledEmbeddings = bundledEmbeddings;
    renderStats();
    renderResults(state.gallery.slice(0, 6).map((item, index) => ({ ...item, rank: index + 1, score: NaN })), 'Gallery preview', 0);
    setStatus('Ready — CLIP loads on first use', 'neutral');
  } catch (error) {
    console.error(error);
    setStatus('Static data failed to load', 'error');
    showMessage('#retrieval-message', 'The gallery data could not be loaded. Serve the web folder over HTTP rather than opening index.html directly.', 'error');
  }
}

initialize();
