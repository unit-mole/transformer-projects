import { env, pipeline } from '@huggingface/transformers';
import { normalizeText, splitIntoSentences } from './text-utils.js';

const MODEL_ID = 'Xenova/distilbart-cnn-12-6';
const MAX_INPUT_TOKENS = 900;
const TOKEN_OVERLAP_SENTENCES = 1;

env.allowLocalModels = false;
env.useBrowserCache = true;

let summarizer = null;
let activeRuntime = null;
let activeDtype = null;
let loadPromise = null;

function postStatus(state, message) {
  self.postMessage({ type: 'status', payload: { state, message, runtime: activeRuntime, dtype: activeDtype } });
}

function normalizeProgress(info) {
  const progress = Number(info?.progress);
  const loaded = Number(info?.loaded);
  const total = Number(info?.total);
  const normalizedReportedProgress = Number.isFinite(progress) && progress >= 0 && progress <= 1 ? progress * 100 : progress;
  const ratio = Number.isFinite(normalizedReportedProgress)
    ? normalizedReportedProgress
    : Number.isFinite(loaded) && Number.isFinite(total) && total > 0
      ? (loaded / total) * 100
      : null;

  return {
    status: info?.status ?? 'loading',
    file: info?.file ?? info?.name ?? '',
    progress: ratio === null ? null : Math.max(0, Math.min(100, ratio)),
    loaded: Number.isFinite(loaded) ? loaded : null,
    total: Number.isFinite(total) ? total : null,
  };
}

function supportsWebGPU() {
  return Boolean(self.navigator && 'gpu' in self.navigator);
}

function runtimePlan(preference) {
  if (preference === 'wasm') return [{ device: 'wasm', dtype: 'q8' }];
  if (preference === 'webgpu') {
    return supportsWebGPU()
      ? [{ device: 'webgpu', dtype: 'q4f16' }, { device: 'wasm', dtype: 'q8' }]
      : [{ device: 'wasm', dtype: 'q8' }];
  }
  return supportsWebGPU()
    ? [{ device: 'webgpu', dtype: 'q4f16' }, { device: 'wasm', dtype: 'q8' }]
    : [{ device: 'wasm', dtype: 'q8' }];
}

async function loadModel(runtimePreference = 'auto') {
  if (summarizer && activeRuntime) return { modelId: MODEL_ID, runtime: activeRuntime, dtype: activeDtype };
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    let lastError = null;
    for (const candidate of runtimePlan(runtimePreference)) {
      try {
        activeRuntime = candidate.device;
        activeDtype = candidate.dtype;
        postStatus('loading', `Loading ${MODEL_ID} with ${candidate.device.toUpperCase()}…`);
        summarizer = await pipeline('summarization', MODEL_ID, {
          device: candidate.device,
          dtype: candidate.dtype,
          progress_callback: (info) => {
            self.postMessage({ type: 'progress', payload: normalizeProgress(info) });
          },
        });
        postStatus('ready', `Model ready on ${candidate.device.toUpperCase()} (${candidate.dtype}).`);
        return { modelId: MODEL_ID, runtime: activeRuntime, dtype: activeDtype };
      } catch (error) {
        lastError = error;
        summarizer = null;
        postStatus('fallback', `${candidate.device.toUpperCase()} load failed; trying a compatible fallback.`);
      }
    }
    activeRuntime = null;
    activeDtype = null;
    throw lastError ?? new Error('Unable to load the browser model.');
  })();

  try {
    return await loadPromise;
  } finally {
    loadPromise = null;
  }
}

function tensorData(inputIds) {
  if (!inputIds) return [];
  if (Array.isArray(inputIds)) return inputIds.flat(Infinity);
  if (inputIds.data) return Array.from(inputIds.data);
  if (typeof inputIds.tolist === 'function') return inputIds.tolist().flat(Infinity);
  return [];
}

async function tokenize(text) {
  if (!summarizer?.tokenizer) return { count: Math.ceil(text.length / 4), preview: [] };
  const encoded = await Promise.resolve(summarizer.tokenizer(text, { add_special_tokens: true }));
  const ids = tensorData(encoded?.input_ids);
  return { count: ids.length, preview: ids.slice(0, 40) };
}

async function splitOversizedSentence(sentence, limit) {
  const words = sentence.split(/\s+/u).filter(Boolean);
  const chunks = [];
  let current = [];

  for (const word of words) {
    const candidate = [...current, word].join(' ');
    const { count } = await tokenize(candidate);
    if (current.length > 0 && count > limit) {
      chunks.push(current.join(' '));
      current = [word];
    } else {
      current.push(word);
    }
  }
  if (current.length) chunks.push(current.join(' '));
  return chunks;
}

async function chunkText(text, limit = MAX_INPUT_TOKENS) {
  const sentences = splitIntoSentences(text);
  if (!sentences.length) return [];

  const chunks = [];
  let current = [];

  for (const sentence of sentences) {
    const sentenceTokens = await tokenize(sentence);
    if (sentenceTokens.count > limit) {
      if (current.length) {
        chunks.push(current.join(' '));
        current = [];
      }
      chunks.push(...(await splitOversizedSentence(sentence, limit)));
      continue;
    }

    const candidate = [...current, sentence].join(' ');
    const candidateTokens = await tokenize(candidate);
    if (current.length > 0 && candidateTokens.count > limit) {
      chunks.push(current.join(' '));
      current = current.slice(-TOKEN_OVERLAP_SENTENCES);
    }
    current.push(sentence);
  }

  if (current.length) chunks.push(current.join(' '));
  return chunks;
}

function extractSummary(output) {
  const first = Array.isArray(output) ? output[0] : output;
  const value = first?.summary_text ?? first?.generated_text ?? '';
  return String(value).trim();
}

async function generateOne(text, settings) {
  const started = performance.now();
  const output = await summarizer(text, {
    min_new_tokens: settings.minNewTokens,
    max_new_tokens: settings.maxNewTokens,
    num_beams: settings.numBeams,
    length_penalty: settings.lengthPenalty,
    no_repeat_ngram_size: settings.noRepeatNgramSize,
    early_stopping: settings.earlyStopping,
  });
  return { summary: extractSummary(output), latencyMs: performance.now() - started };
}

async function summarizeDocument({ text, settings, longDocumentMode = true }) {
  const normalized = normalizeText(text);
  if (!normalized) throw new Error('Enter source text before generating a summary.');

  const inputTokenInfo = await tokenize(normalized);
  let chunks = [normalized];
  let truncatedToFirstChunk = false;
  if (inputTokenInfo.count > MAX_INPUT_TOKENS) {
    const safeChunks = await chunkText(normalized, MAX_INPUT_TOKENS);
    if (longDocumentMode) {
      chunks = safeChunks;
    } else {
      chunks = safeChunks.slice(0, 1);
      truncatedToFirstChunk = safeChunks.length > 1;
    }
  }

  const partials = [];
  let totalLatencyMs = 0;
  for (let index = 0; index < chunks.length; index += 1) {
    postStatus('inference', `Summarizing chunk ${index + 1} of ${chunks.length}…`);
    const result = await generateOne(chunks[index], settings);
    partials.push(result.summary);
    totalLatencyMs += result.latencyMs;
  }

  let summary = partials.join(' ');
  let secondPass = false;
  if (partials.length > 1) {
    const combinedTokenInfo = await tokenize(summary);
    if (combinedTokenInfo.count > settings.maxNewTokens || partials.length > 2) {
      postStatus('inference', 'Combining chunk summaries in a second pass…');
      const combinedChunks = combinedTokenInfo.count > MAX_INPUT_TOKENS
        ? await chunkText(summary, MAX_INPUT_TOKENS)
        : [summary];
      const finalParts = [];
      for (const chunk of combinedChunks) {
        const result = await generateOne(chunk, settings);
        finalParts.push(result.summary);
        totalLatencyMs += result.latencyMs;
      }
      summary = finalParts.join(' ');
      secondPass = true;
    }
  }

  const summaryTokenInfo = await tokenize(summary);
  postStatus('ready', `Summary generated on ${activeRuntime.toUpperCase()} (${activeDtype}).`);

  return {
    summary,
    latencyMs: totalLatencyMs,
    inputTokens: inputTokenInfo.count,
    inputTokenPreview: inputTokenInfo.preview,
    summaryTokens: summaryTokenInfo.count,
    chunks: chunks.length,
    secondPass,
    truncatedToFirstChunk,
    runtime: activeRuntime,
    dtype: activeDtype,
    modelId: MODEL_ID,
  };
}

async function handleRequest(message) {
  const { requestId, action, payload } = message;
  try {
    if (action === 'load') {
      const result = await loadModel(payload.runtimePreference);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }

    await loadModel(payload.runtimePreference);

    if (action === 'summarize') {
      const result = await summarizeDocument(payload);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }

    if (action === 'compare-beams') {
      const greedySettings = { ...payload.settings, numBeams: 1 };
      const selectedSettings = { ...payload.settings };
      const greedy = await summarizeDocument({ ...payload, settings: greedySettings });
      const selected = await summarizeDocument({ ...payload, settings: selectedSettings });
      self.postMessage({ type: 'result', requestId, payload: { greedy, selected } });
      return;
    }

    throw new Error(`Unsupported worker action: ${action}`);
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId,
      payload: {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : '',
      },
    });
  }
}

self.addEventListener('message', (event) => {
  void handleRequest(event.data);
});
