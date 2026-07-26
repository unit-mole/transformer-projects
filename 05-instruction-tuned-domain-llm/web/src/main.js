import './styles.css';
import { APP_CONFIG, runtimeVariables } from './config.js';
import { demoEvaluation } from './evaluation.js';
import { EXAMPLES } from './examples.js';
import { BrowserModelClient } from './model-client.js';
import { buildPrompt, PROMPT_CATEGORIES } from './prompt-templates.js';
import { detectWebGpu, escapeHtml, formatMilliseconds, safeModelId, setHidden } from './utils.js';

const hfVariables = runtimeVariables();
const customFromSpace = hfVariables.DOMAIN_MODEL_ID || APP_CONFIG.defaultCustomModelId;
const hasWebGpu = detectWebGpu();

const app = document.querySelector('#app');
app.innerHTML = `
  <header class="hero shell">
    <div>
      <p class="eyebrow">Transformer Portfolio · Project ${APP_CONFIG.projectNumber}</p>
      <h1>${APP_CONFIG.title}</h1>
      <p class="lead">Run a FLAN-T5 encoder-decoder Transformer directly in your browser with Transformers.js and ONNX Runtime Web.</p>
      <div class="badges">
        <span class="badge success">Real browser inference</span>
        <span class="badge">No Python server</span>
        <span class="badge">WebGPU + WASM fallback</span>
        <span class="badge">LoRA → merged ONNX pathway</span>
      </div>
    </div>
    <div class="hero-card">
      <strong>Runtime detected</strong>
      <span>${hasWebGpu ? 'WebGPU available' : 'WASM fallback required'}</span>
      <small>Model files download once and are cached by the browser.</small>
    </div>
  </header>

  <main class="shell layout">
    <section class="panel controls-panel">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Interactive assistant</p>
          <h2>Ask an ML or Data Science question</h2>
        </div>
        <span id="model-state" class="status-pill idle">Not loaded</span>
      </div>

      <div class="notice responsible"><strong>Responsible use:</strong> ${APP_CONFIG.responsibleUse}</div>

      <div class="form-grid two">
        <label>
          Model profile
          <select id="model-profile">
            <option value="base">Base FLAN-T5 demo</option>
            <option value="custom">Custom merged domain model</option>
          </select>
        </label>
        <label id="custom-model-wrap" hidden>
          Hugging Face ONNX model ID
          <input id="custom-model-id" value="${escapeHtml(customFromSpace)}" placeholder="YOUR_HF_USERNAME/ml-ds-flan-t5-small-onnx" />
        </label>
      </div>
      <p id="model-claim" class="model-claim">${APP_CONFIG.baseModel.claim}</p>

      <div class="form-grid two">
        <label>
          Prompt category
          <select id="category">${PROMPT_CATEGORIES.map((item) => `<option>${escapeHtml(item)}</option>`).join('')}</select>
        </label>
        <label>
          Execution preference
          <select id="device">
            <option value="auto">Auto (${hasWebGpu ? 'WebGPU preferred' : 'WASM'})</option>
            <option value="wasm">WASM / CPU</option>
            ${hasWebGpu ? '<option value="webgpu">WebGPU</option>' : ''}
          </select>
        </label>
      </div>

      <label>
        Question
        <textarea id="instruction" maxlength="${APP_CONFIG.maxInputCharacters}" rows="5" placeholder="Explain precision vs recall with a quality analytics example."></textarea>
        <span class="counter"><span id="instruction-count">0</span> / ${APP_CONFIG.maxInputCharacters}</span>
      </label>

      <label>
        Optional supporting context
        <textarea id="context" maxlength="${APP_CONFIG.maxContextCharacters}" rows="3" placeholder="Add constraints, a synthetic scenario, or the desired audience."></textarea>
        <span class="counter"><span id="context-count">0</span> / ${APP_CONFIG.maxContextCharacters}</span>
      </label>

      <div class="examples" id="examples"></div>

      <details class="generation-controls">
        <summary>Generation controls</summary>
        <div class="form-grid four">
          <label>Max new tokens <input id="max-tokens" type="number" min="32" max="256" step="8" value="160" /></label>
          <label>Temperature <input id="temperature" type="number" min="0" max="1" step="0.05" value="0.30" /></label>
          <label>Top-p <input id="top-p" type="number" min="0.5" max="1" step="0.05" value="0.90" /></label>
          <label>Repetition penalty <input id="repetition" type="number" min="1" max="1.5" step="0.05" value="1.10" /></label>
        </div>
      </details>

      <div class="actions">
        <button id="load-model" class="secondary">Load model</button>
        <button id="generate" class="primary">Generate response</button>
        <button id="clear" class="ghost">Clear</button>
      </div>

      <div class="progress-card">
        <div class="progress-row"><span id="progress-label">Model has not been loaded.</span><span id="progress-percent">0%</span></div>
        <progress id="progress" value="0" max="100"></progress>
      </div>
    </section>

    <section class="panel output-panel">
      <div class="section-heading">
        <div><p class="section-kicker">Generated output</p><h2>Assistant response</h2></div>
        <button id="copy" class="icon-button" disabled>Copy</button>
      </div>
      <div id="response" class="response empty">Your generated educational response will appear here.</div>
      <div id="error" class="notice error" hidden></div>

      <div class="metrics-grid">
        <article><span>Source tokens</span><strong id="source-tokens">—</strong></article>
        <article><span>Output tokens</span><strong id="target-tokens">—</strong></article>
        <article><span>Inference latency</span><strong id="latency">—</strong></article>
        <article><span>Runtime</span><strong id="runtime">—</strong></article>
      </div>

      <details open>
        <summary>Tokenizer preview</summary>
        <div id="token-preview" class="token-box">Load a model and generate a response to inspect actual tokenizer output.</div>
      </details>

      <details>
        <summary>Transparent demo diagnostics</summary>
        <div class="diagnostics">
          <div><span>Adherence heuristic</span><strong id="adherence">—</strong></div>
          <div><span>Relevance heuristic</span><strong id="relevance">—</strong></div>
          <div><span>Hallucination flag</span><strong id="hallucination">—</strong></div>
        </div>
        <ul id="warnings" class="warnings"><li>No response evaluated.</li></ul>
        <p class="fine-print">These browser heuristics are not the project's official BERTScore, human review, or hallucination evaluation.</p>
      </details>
    </section>
  </main>

  <section class="shell info-grid">
    <article class="panel info-card">
      <p class="section-kicker">Architecture</p>
      <h2>LoRA to browser deployment</h2>
      <img src="./architecture.svg" alt="Architecture showing LoRA training, adapter merging, ONNX export, and browser inference" />
      <ol>
        <li>Fine-tune FLAN-T5-small using LoRA / PEFT.</li>
        <li>Merge the trained adapter into the base model.</li>
        <li>Export and quantize the merged model as ONNX.</li>
        <li>Run text-to-text generation locally in the browser.</li>
      </ol>
    </article>

    <article class="panel info-card">
      <p class="section-kicker">Evaluation</p>
      <h2>No invented results</h2>
      <p>The public project includes instruction adherence, BERTScore, response relevance, manual review, latency, and hallucination analysis. Values remain marked <strong>not run</strong> until evaluation is executed against a trained adapter.</p>
      <div id="evaluation-summary" class="evaluation-summary"></div>
    </article>

    <article class="panel info-card">
      <p class="section-kicker">Project links</p>
      <h2>Engineering evidence</h2>
      <nav class="link-list">
        <a href="${APP_CONFIG.githubUrl}" target="_blank" rel="noreferrer">Python implementation</a>
        <a href="${APP_CONFIG.modelCardUrl}" target="_blank" rel="noreferrer">Model card</a>
        <a href="${APP_CONFIG.datasetCardUrl}" target="_blank" rel="noreferrer">Dataset card</a>
        <a href="${APP_CONFIG.evaluationUrl}" target="_blank" rel="noreferrer">Evaluation outputs</a>
      </nav>
      <p class="fine-print">Replace the placeholder GitHub username in <code>src/config.js</code> after publishing.</p>
    </article>
  </section>

  <footer class="shell footer">
    <strong>Project 05 · Instruction-Tuned Domain LLM</strong>
    <span>Python + PEFT + LoRA + Transformers.js + ONNX Runtime Web</span>
  </footer>
`;

const elements = Object.fromEntries([
  'model-profile', 'custom-model-wrap', 'custom-model-id', 'model-claim', 'category', 'device',
  'instruction', 'context', 'instruction-count', 'context-count', 'examples', 'max-tokens',
  'temperature', 'top-p', 'repetition', 'load-model', 'generate', 'clear', 'progress-label',
  'progress-percent', 'progress', 'model-state', 'response', 'error', 'copy', 'source-tokens',
  'target-tokens', 'latency', 'runtime', 'token-preview', 'adherence', 'relevance',
  'hallucination', 'warnings', 'evaluation-summary',
].map((id) => [id, document.getElementById(id)]));

let loadedRuntime = null;
let isBusy = false;

const client = new BrowserModelClient({
  onStatus: (message) => setProgress(message, null),
  onProgress: (progress) => {
    const percent = Number(progress?.progress ?? 0);
    const file = progress?.file ? ` · ${progress.file}` : '';
    setProgress(`${progress?.status ?? 'Loading'}${file}`, percent);
  },
});

function setProgress(message, percent) {
  elements['progress-label'].textContent = message;
  if (Number.isFinite(percent)) {
    elements.progress.value = percent;
    elements['progress-percent'].textContent = `${Math.round(percent)}%`;
  }
}

function setState(label, mode) {
  elements['model-state'].textContent = label;
  elements['model-state'].className = `status-pill ${mode}`;
}

function runtimeSelection() {
  const isCustom = elements['model-profile'].value === 'custom';
  const modelId = isCustom ? safeModelId(elements['custom-model-id'].value) : APP_CONFIG.baseModel.id;
  if (isCustom && !modelId) throw new Error('Enter your merged ONNX domain-model repository ID.');
  const requestedDevice = elements.device.value === 'auto' ? (hasWebGpu ? 'webgpu' : 'wasm') : elements.device.value;
  return {
    modelId,
    device: requestedDevice,
    dtype: requestedDevice === 'webgpu' ? 'q4' : 'q8',
  };
}

function renderExamples() {
  elements.examples.innerHTML = EXAMPLES.map((example, index) => `
    <button class="example-chip" data-example="${index}" title="${escapeHtml(example.prompt)}">${escapeHtml(example.category)}</button>
  `).join('');
  elements.examples.querySelectorAll('[data-example]').forEach((button) => {
    button.addEventListener('click', () => {
      const example = EXAMPLES[Number(button.dataset.example)];
      elements.category.value = example.category;
      elements.instruction.value = example.prompt;
      elements.context.value = example.context;
      updateCounters();
    });
  });
}

function updateCounters() {
  elements['instruction-count'].textContent = elements.instruction.value.length;
  elements['context-count'].textContent = elements.context.value.length;
}

function updateModelProfile() {
  const custom = elements['model-profile'].value === 'custom';
  setHidden(elements['custom-model-wrap'], !custom);
  elements['model-claim'].textContent = custom
    ? 'Custom mode is valid only for a merged, tested, ONNX-exported model repository with Transformers.js-compatible files.'
    : APP_CONFIG.baseModel.claim;
  loadedRuntime = null;
  setState('Not loaded', 'idle');
}

function setBusy(busy) {
  isBusy = busy;
  elements['load-model'].disabled = busy;
  elements.generate.disabled = busy;
  elements['model-profile'].disabled = busy;
  elements.device.disabled = busy;
}

function showError(error) {
  elements.error.textContent = error.message ?? String(error);
  setHidden(elements.error, false);
  setState('Error', 'error');
}

async function loadModel() {
  setHidden(elements.error, true);
  setBusy(true);
  setState('Loading', 'loading');
  try {
    const runtime = runtimeSelection();
    setProgress(`Preparing ${runtime.modelId}…`, 0);
    loadedRuntime = await client.load(runtime);
    setState('Ready', 'ready');
    setProgress(`Ready: ${loadedRuntime.modelId} · ${loadedRuntime.device.toUpperCase()} · ${loadedRuntime.dtype}`, 100);
    return loadedRuntime;
  } catch (error) {
    loadedRuntime = null;
    showError(error);
    throw error;
  } finally {
    setBusy(false);
  }
}

async function generate() {
  if (isBusy) return;
  const instruction = elements.instruction.value.trim();
  if (!instruction) {
    showError(new Error('Enter an ML or Data Science question before generating.'));
    return;
  }

  setHidden(elements.error, true);
  setBusy(true);
  setState('Generating', 'loading');
  elements.response.className = 'response loading-response';
  elements.response.textContent = 'Generating with the Transformer model…';

  try {
    if (!loadedRuntime) await loadModel();
    setBusy(true);
    const prompt = buildPrompt({
      category: elements.category.value,
      instruction,
      context: elements.context.value,
    });
    const settings = {
      maxNewTokens: Number(elements['max-tokens'].value),
      temperature: Number(elements.temperature.value),
      topP: Number(elements['top-p'].value),
      repetitionPenalty: Number(elements.repetition.value),
    };
    const result = await client.generate(prompt, settings);
    const diagnostics = demoEvaluation(instruction, result.generatedText);

    elements.response.className = 'response';
    elements.response.textContent = result.generatedText || 'The model returned an empty response.';
    elements.copy.disabled = !result.generatedText;
    elements['source-tokens'].textContent = result.sourceTokenCount;
    elements['target-tokens'].textContent = result.targetTokenCount;
    elements.latency.textContent = formatMilliseconds(result.latencyMs);
    elements.runtime.textContent = `${result.device.toUpperCase()} · ${result.dtype}`;
    elements['token-preview'].innerHTML = result.tokenPreview
      .map((token, index) => `<span class="token" title="Token ID ${escapeHtml(result.tokenIdsPreview[index] ?? '')}">${escapeHtml(token)}</span>`)
      .join('');
    elements.adherence.textContent = diagnostics.adherenceScore.toFixed(2);
    elements.relevance.textContent = diagnostics.relevanceScore.toFixed(2);
    elements.hallucination.textContent = diagnostics.hallucinationRisk;
    elements.warnings.innerHTML = diagnostics.warnings.length
      ? diagnostics.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')
      : '<li>No simple overconfidence pattern was flagged. Manual factual review is still required.</li>';
    setState('Ready', 'ready');
    setProgress(`Generated with ${result.modelId}`, 100);
  } catch (error) {
    elements.response.className = 'response empty';
    elements.response.textContent = 'Generation did not complete.';
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function loadEvaluationSummary() {
  try {
    const response = await fetch('./evaluation-summary.json');
    const data = await response.json();
    elements['evaluation-summary'].innerHTML = Object.entries(data.metrics)
      .map(([metric, value]) => `<div><span>${escapeHtml(metric)}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join('');
  } catch {
    elements['evaluation-summary'].textContent = 'Evaluation summary could not be loaded.';
  }
}

elements['model-profile'].addEventListener('change', updateModelProfile);
elements['custom-model-id'].addEventListener('input', () => {
  loadedRuntime = null;
  setState('Not loaded', 'idle');
});
elements.device.addEventListener('change', () => {
  loadedRuntime = null;
  setState('Not loaded', 'idle');
});
elements.instruction.addEventListener('input', updateCounters);
elements.context.addEventListener('input', updateCounters);
elements['load-model'].addEventListener('click', () => loadModel().catch(() => {}));
elements.generate.addEventListener('click', generate);
elements.clear.addEventListener('click', () => {
  elements.instruction.value = '';
  elements.context.value = '';
  elements.response.className = 'response empty';
  elements.response.textContent = 'Your generated educational response will appear here.';
  updateCounters();
});
elements.copy.addEventListener('click', async () => {
  await navigator.clipboard.writeText(elements.response.textContent);
  elements.copy.textContent = 'Copied';
  setTimeout(() => { elements.copy.textContent = 'Copy'; }, 1200);
});

renderExamples();
updateModelProfile();
updateCounters();
loadEvaluationSummary();
